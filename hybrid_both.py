import sys
from unittest.mock import MagicMock
import itertools # <--- Added for Truth Table generation

# 1. Setup Environment
sys.modules["tensorboardX"] = MagicMock()
import torch
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
import torch.nn.functional as F
import xgboost as xgb
import os

# Add your repo to path if needed
if "/home/ingaiza/CRNN" not in sys.path:
    sys.path.append("/home/ingaiza/CRNN")

from net.model import AudioCRNN

# --- CONFIGURATION ---
CRNN_MODEL_PATH = "/home/ingaiza/CRNN/models/CRNN_training_output2-20251112T081448Z-1-001/CRNN_training_output2/1111_140703/checkpoints/model_best.pth"
XGB_MODEL_PATH = "/home/ingaiza/CRNN/models/xgboost_mixed_model.json"
CFG_PATH = "/home/ingaiza/CRNN/crnn.cfg" 

# CHANGE THIS TO TEST DIFFERENT FILES
TEST_FILE = "/home/ingaiza/GoogleDrive/Ambience20251119_103522.wav"

CLASS_MAP = {
    0: "Natural",
    1: "Unnatural",
    2: "Human Sound"
}

# --- VALIDATION LAYER ---
def validate_prediction(probs):
    """
    Applies sensitivity analysis using a truth table of weights (0.76, 1.24).
    Returns: (is_valid, pass_ratio)
    """
    original_winner_idx = np.argmax(probs)
    
    # Weights (Accuracy Allowance)
    w_low = 0.76
    w_high = 1.24
    
    # Generate Truth Table (8 combinations for 3 classes)
    # Cartesian product of [Low, High] repeated 3 times
    multipliers = [w_low, w_high]
    combinations = list(itertools.product(multipliers, repeat=3))
    
    wins = 0
    total_scenarios = len(combinations)

    for coeffs in combinations:
        # Apply weights: [P_N * w1,  P_U * w2,  P_H * w3]
        weighted_probs = np.array(probs) * np.array(coeffs)
        round_winner_idx = np.argmax(weighted_probs)
        
        # Check if the original winner still wins this scenario
        if round_winner_idx == original_winner_idx:
            wins += 1
            
    # Majority Rule: Must win more than half (e.g., 5 out of 8)
    is_valid = wins > (total_scenarios / 2) 
    pass_ratio = wins / total_scenarios
    
    return is_valid, pass_ratio
# -----------------------

def predict_hybrid():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Tier 1 (CRNN)
    print("Loading Tier 1 (CRNN)...")
    config = {'cfg': CFG_PATH, 'transforms': {'args': {'channels': 'mono'}}}
    crnn = AudioCRNN(classes=CLASS_MAP.values(), config=config)
    
    try:
        checkpoint = torch.load(CRNN_MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint: crnn.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint: crnn.load_state_dict(checkpoint['state_dict'])
            else: crnn.load_state_dict(checkpoint)
        else:
            crnn.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading CRNN: {e}")
        return
    
    crnn.eval().to(device)

    # 2. Load Tier 2 (XGBoost)
    print("Loading Tier 2 (XGBoost)...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)

    # 3. Process Audio
    print(f"Processing: {TEST_FILE}")
    try:
        data, sr = sf.read(TEST_FILE)
        waveform = torch.from_numpy(data).float()
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        else: waveform = waveform.permute(1, 0)

        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # --- ADAPTIVE STRATEGY ---
        num_samples = waveform.shape[1]
        THRESHOLD_SAMPLES = 240000 # 15 seconds * 16000 Hz
        
        final_probs = None
        strategy_name = ""

        # STRATEGY A: WHOLE FILE (For > 15s)
        if num_samples > THRESHOLD_SAMPLES:
            strategy_name = "Whole File (Long-Term Context)"
            
            # Just feed the whole thing (Model handles long sequences well)
            seqs = waveform.permute(1, 0).unsqueeze(0)
            lengths = torch.tensor([seqs.shape[1]]).long()
            srs = torch.tensor([16000]).long()
            batch = (seqs.to(device), lengths.to(device), srs.to(device))
            
            with torch.no_grad():
                features = crnn(batch, return_features=True)
                features_np = features.cpu().numpy()
                final_probs = xgb_model.predict_proba(features_np)[0]

        # STRATEGY B: SLIDING WINDOW (For <= 15s)
        else:
            strategy_name = "Sliding Window (Stabilization)"
            
            # Use 6-second windows to ensure no "output size too small" errors
            WINDOW_SIZE = 96000  
            STRIDE = 48000       
            
            windows = []
            
            # Case 1: Shorter than 6s -> Pad once
            if num_samples < WINDOW_SIZE:
                pad_amount = WINDOW_SIZE - num_samples
                chunk = F.pad(waveform, (0, pad_amount))
                windows.append(chunk)
            # Case 2: Between 6s and 15s -> Slice
            else:
                for i in range(0, num_samples, STRIDE):
                    chunk = waveform[:, i : i + WINDOW_SIZE]
                    if chunk.shape[1] < WINDOW_SIZE:
                        if chunk.shape[1] < 8000: continue # Skip tiny tails
                        pad_amount = WINDOW_SIZE - chunk.shape[1]
                        chunk = F.pad(chunk, (0, pad_amount))
                    windows.append(chunk)
            
            print(f"DEBUG: Using {len(windows)} windows.")
            window_probs = []
            
            for chunk in windows:
                seqs = chunk.permute(1, 0).unsqueeze(0) 
                lengths = torch.tensor([seqs.shape[1]]).long()
                srs = torch.tensor([16000]).long()
                batch = (seqs.to(device), lengths.to(device), srs.to(device))

                with torch.no_grad():
                    features = crnn(batch, return_features=True)
                    features_np = features.cpu().numpy()
                    probs = xgb_model.predict_proba(features_np)[0]
                    window_probs.append(probs)
            
            final_probs = np.mean(window_probs, axis=0)

        # --- VALIDATION CHECK ---
        is_valid, pass_ratio = validate_prediction(final_probs)
        pred_idx = np.argmax(final_probs)

    except Exception as e:
        print(f"Audio Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- HYBRID Prediction Complete ---")
    print(f"Strategy Used: \033[1m{strategy_name}\033[0m")
    print(f"File: {os.path.basename(TEST_FILE)}")
    
    pred_class = CLASS_MAP[pred_idx]
    print(f"Raw Prediction: \033[1m{pred_class}\033[0m ({final_probs[pred_idx]:.2%})")
    
    print(f"Validation Score: Won {pass_ratio:.0%} of scenarios.")
    
    if is_valid:
        print("-> STATUS: \033[92mVALID ALERT\033[0m (Passed Sensitivity Check)")
    else:
        print("-> STATUS: \033[91mBLOCKED\033[0m (Prediction too weak/noisy)")
    
    print("\nFull Probabilities:")
    for i, name in CLASS_MAP.items():
        print(f"  {name}: {final_probs[i]:.2%}")

if __name__ == "__main__":
    predict_hybrid()