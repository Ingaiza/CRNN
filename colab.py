import time
import glob
import shutil
import requests
import torch
import soundfile as sf
import numpy as np
import xgboost as xgb
import itertools
import torchaudio.transforms as T
import torch.nn.functional as F
from unittest.mock import MagicMock

# --- 1. MOCKING & ALIASING (Crucial for loading checkpoint) ---
# sys.modules["tensorboardX"] = MagicMock()

# Mock the 'utils' module structure expected by the checkpoint
# We assume the repo is at /content/CRNN, so we point to the real utils there
try:
    import utils 
    sys.modules["utils"] = utils
    sys.modules["utils.logger"] = utils.logger
    sys.modules["utils.util"] = utils.util
except ImportError:
    # Fallback if import fails (e.g. repo structure changed)
    utils_mock = MagicMock()
    utils_mock.logger = MagicMock()
    # Define a dummy class for the pickle loader
    class MockLogger:
        def __init__(self, *args, **kwargs): pass
    utils_mock.logger.Logger = MockLogger
    sys.modules["utils"] = utils_mock
    sys.modules["utils.logger"] = utils_mock.logger

from net.model import AudioCRNN

# --- 2. CONFIGURATION ---
CRNN_MODEL_PATH = "/content/drive/MyDrive/CRNN_training_output2-20251112T081448Z-1-001/CRNN_training_output2/1111_140703/checkpoints/model_best.pth"
XGB_MODEL_PATH = "/content/drive/MyDrive/xgboost_audio_classifier.json"
CFG_PATH = "/content/CRNN/crnn.cfg"

# Your Ngrok URL (UPDATE THIS EVERY TIME YOU RESTART NGROK)
WEBHOOK_URL = "https://YOUR-NGROK-URL.ngrok-free.app/alert" 

CLASS_MAP = {0: "Natural", 1: "Unnatural", 2: "Human Sound"}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 3. VALIDATION LAYER (Truth Table Logic) ---
def validate_prediction(probs):
    """
    Applies sensitivity analysis using a truth table of weights (0.76, 1.24).
    Returns: (is_valid, pass_ratio)
    """
    original_winner_idx = np.argmax(probs)
    original_winner_class = CLASS_MAP[original_winner_idx]
    
    # Weights
    w_low = 0.76
    w_high = 1.24
    
    # Generate Truth Table (8 combinations for 3 classes)
    multipliers = [w_low, w_high]
    combinations = list(itertools.product(multipliers, repeat=3))
    
    wins = 0
    total_scenarios = len(combinations)

    for coeffs in combinations:
        # Apply weights: [P_N * w1,  P_U * w2,  P_H * w3]
        weighted_probs = np.array(probs) * np.array(coeffs)
        round_winner_idx = np.argmax(weighted_probs)
        
        if round_winner_idx == original_winner_idx:
            wins += 1
            
    is_valid = wins > (total_scenarios / 2) # Majority rule (>4/8)
    pass_ratio = wins / total_scenarios
    
    return is_valid, pass_ratio

# --- 4. INITIALIZATION ---
print("Loading Models...")
# Load CRNN
config = {'cfg': CFG_PATH, 'transforms': {'args': {'channels': 'mono'}}}
crnn = AudioCRNN(classes=CLASS_MAP.values(), config=config)
checkpoint = torch.load(CRNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    crnn.load_state_dict(state_dict)
else:
    crnn.load_state_dict(checkpoint)
crnn.eval().to(DEVICE)

# Load XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)
print("Models Loaded. Watching 'Ambience' folder...")

# --- 5. MAIN LOOP ---
while True:
    wav_files = glob.glob(os.path.join(WATCH_FOLDER, "*.wav"))
    
    if not wav_files:
        time.sleep(1)
        continue
        
    # Process oldest file first
    wav_files.sort(key=os.path.getmtime)
    current_file = wav_files[0]
    filename = os.path.basename(current_file)
    
    print(f"\nProcessing: {filename}")
    
    try:
        # A. Load Audio
        data, sr = sf.read(current_file)
        waveform = torch.from_numpy(data).float()
        
        # Preprocessing (Standardize to 15s / 16kHz / Mono)
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        else: waveform = waveform.permute(1, 0) # (channels, time)
        
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Prepare Batch (Whole File Strategy is best for 15s)
        seqs = waveform.permute(1, 0).unsqueeze(0) # (1, time, 1)
        lengths = torch.tensor([seqs.shape[1]]).long()
        srs = torch.tensor([16000]).long()
        batch = (seqs.to(DEVICE), lengths.to(DEVICE), srs.to(DEVICE))

        # B. Hybrid Inference
        with torch.no_grad():
            # Tier 1: CRNN Features
            features = crnn(batch, return_features=True)
            features_np = features.cpu().numpy()
            
            # Tier 2: XGBoost Probabilities
            probs = xgb_model.predict_proba(features_np)[0]

        # C. Validation Layer
        is_valid, pass_ratio = validate_prediction(probs)
        
        pred_idx = np.argmax(probs)
        pred_class = CLASS_MAP[pred_idx]
        
        print(f"-> Raw Prediction: {pred_class} ({probs[pred_idx]:.2%})")
        print(f"-> Validation: Won {pass_ratio:.0%} of scenarios.")

        if is_valid:
            print("-> STATUS: VALID ALERT. Sending to UI...")
            payload = {
                "class": pred_class,
                "probs": [float(p) for p in probs], # Convert numpy float to py float
                "filename": filename
            }
            try:
                requests.post(WEBHOOK_URL, json=payload, timeout=2)
                print("   Success: Webhook sent.")
            except Exception as e:
                print(f"   Error sending webhook: {e}")
        else:
            print(f"-> STATUS: BLOCKED. Prediction '{pred_class}' too weak/noisy.")

        # D. Cleanup (Move to Processed)
        shutil.move(current_file, os.path.join(PROCESSED_FOLDER, filename))
        
    except Exception as e:
        print(f"CRITICAL ERROR processing {filename}: {e}")
        # Move to processed even on error to prevent infinite loops
        error_folder = os.path.join(PROCESSED_FOLDER, "Errors")
        if not os.path.exists(error_folder): os.makedirs(error_folder)
        shutil.move(current_file, os.path.join(error_folder, filename))import time
import glob
import shutil
import requests
import torch
import soundfile as sf
import numpy as np
import xgboost as xgb
import itertools
import torchaudio.transforms as T
import torch.nn.functional as F
from unittest.mock import MagicMock

# --- 1. MOCKING & ALIASING (Crucial for loading checkpoint) ---
sys.modules["tensorboardX"] = MagicMock()

# Mock the 'utils' module structure expected by the checkpoint
# We assume the repo is at /content/CRNN, so we point to the real utils there
try:
    import utils 
    sys.modules["utils"] = utils
    sys.modules["utils.logger"] = utils.logger
    sys.modules["utils.util"] = utils.util
except ImportError:
    # Fallback if import fails (e.g. repo structure changed)
    utils_mock = MagicMock()
    utils_mock.logger = MagicMock()
    # Define a dummy class for the pickle loader
    class MockLogger:
        def __init__(self, *args, **kwargs): pass
    utils_mock.logger.Logger = MockLogger
    sys.modules["utils"] = utils_mock
    sys.modules["utils.logger"] = utils_mock.logger

from net.model import AudioCRNN

# --- 2. CONFIGURATION ---
CRNN_MODEL_PATH = "/content/drive/MyDrive/CRNN_training_output2-20251112T081448Z-1-001/CRNN_training_output2/1111_140703/checkpoints/model_best.pth"
XGB_MODEL_PATH = "/content/drive/MyDrive/xgboost_audio_classifier.json"
CFG_PATH = "/content/CRNN/crnn.cfg"

# Your Ngrok URL (UPDATE THIS EVERY TIME YOU RESTART NGROK)
WEBHOOK_URL = "https://YOUR-NGROK-URL.ngrok-free.app/alert" 

CLASS_MAP = {0: "Natural", 1: "Unnatural", 2: "Human Sound"}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 3. VALIDATION LAYER (Truth Table Logic) ---
def validate_prediction(probs):
    """
    Applies sensitivity analysis using a truth table of weights (0.76, 1.24).
    Returns: (is_valid, pass_ratio)
    """
    original_winner_idx = np.argmax(probs)
    original_winner_class = CLASS_MAP[original_winner_idx]
    
    # Weights
    w_low = 0.76
    w_high = 1.24
    
    # Generate Truth Table (8 combinations for 3 classes)
    multipliers = [w_low, w_high]
    combinations = list(itertools.product(multipliers, repeat=3))
    
    wins = 0
    total_scenarios = len(combinations)

    for coeffs in combinations:
        # Apply weights: [P_N * w1,  P_U * w2,  P_H * w3]
        weighted_probs = np.array(probs) * np.array(coeffs)
        round_winner_idx = np.argmax(weighted_probs)
        
        if round_winner_idx == original_winner_idx:
            wins += 1
            
    is_valid = wins > (total_scenarios / 2) # Majority rule (>4/8)
    pass_ratio = wins / total_scenarios
    
    return is_valid, pass_ratio

# --- 4. INITIALIZATION ---
print("Loading Models...")
# Load CRNN
config = {'cfg': CFG_PATH, 'transforms': {'args': {'channels': 'mono'}}}
crnn = AudioCRNN(classes=CLASS_MAP.values(), config=config)
checkpoint = torch.load(CRNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    crnn.load_state_dict(state_dict)
else:
    crnn.load_state_dict(checkpoint)
crnn.eval().to(DEVICE)

# Load XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)
print("Models Loaded. Watching 'Ambience' folder...")

# --- 5. MAIN LOOP ---
while True:
    wav_files = glob.glob(os.path.join(WATCH_FOLDER, "*.wav"))
    
    if not wav_files:
        time.sleep(1)
        continue
        
    # Process oldest file first
    wav_files.sort(key=os.path.getmtime)
    current_file = wav_files[0]
    filename = os.path.basename(current_file)
    
    print(f"\nProcessing: {filename}")
    
    try:
        # A. Load Audio
        data, sr = sf.read(current_file)
        waveform = torch.from_numpy(data).float()
        
        # Preprocessing (Standardize to 15s / 16kHz / Mono)
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        else: waveform = waveform.permute(1, 0) # (channels, time)
        
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Prepare Batch (Whole File Strategy is best for 15s)
        seqs = waveform.permute(1, 0).unsqueeze(0) # (1, time, 1)
        lengths = torch.tensor([seqs.shape[1]]).long()
        srs = torch.tensor([16000]).long()
        batch = (seqs.to(DEVICE), lengths.to(DEVICE), srs.to(DEVICE))

        # B. Hybrid Inference
        with torch.no_grad():
            # Tier 1: CRNN Features
            features = crnn(batch, return_features=True)
            features_np = features.cpu().numpy()
            
            # Tier 2: XGBoost Probabilities
            probs = xgb_model.predict_proba(features_np)[0]

        # C. Validation Layer
        is_valid, pass_ratio = validate_prediction(probs)
        
        pred_idx = np.argmax(probs)
        pred_class = CLASS_MAP[pred_idx]
        
        print(f"-> Raw Prediction: {pred_class} ({probs[pred_idx]:.2%})")
        print(f"-> Validation: Won {pass_ratio:.0%} of scenarios.")

        if is_valid:
            print("-> STATUS: VALID ALERT. Sending to UI...")
            payload = {
                "class": pred_class,
                "probs": [float(p) for p in probs], # Convert numpy float to py float
                "filename": filename
            }
            try:
                requests.post(WEBHOOK_URL, json=payload, timeout=2)
                print("   Success: Webhook sent.")
            except Exception as e:
                print(f"   Error sending webhook: {e}")
        else:
            print(f"-> STATUS: BLOCKED. Prediction '{pred_class}' too weak/noisy.")

        # D. Cleanup (Move to Processed)
        shutil.move(current_file, os.path.join(PROCESSED_FOLDER, filename))
        
    except Exception as e:
        print(f"CRITICAL ERROR processing {filename}: {e}")
        # Move to processed even on error to prevent infinite loops
        error_folder = os.path.join(PROCESSED_FOLDER, "Errors")
        if not os.path.exists(error_folder): os.makedirs(error_folder)
        shutil.move(current_file, os.path.join(error_folder, filename))