import sys
from unittest.mock import MagicMock

# 1. Setup Environment
sys.modules["tensorboardX"] = MagicMock()
import torch
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
import xgboost as xgb
import os

from net.model import AudioCRNN

# --- CONFIGURATION ---
# 1. Paths
CRNN_MODEL_PATH = "/home/ingaiza/CRNN/models/CRNN_training_output2-20251112T081448Z-1-001/CRNN_training_output2/1111_140703/checkpoints/model_best.pth"
XGB_MODEL_PATH = "/home/ingaiza/CRNN/models/xgboost_audio_classifier.json" # The file you just created
CFG_PATH = "/home/ingaiza/CRNN/crnn.cfg" 

# 2. The File You Want to Test
# (Ensure this file exists in your Colab audio_data folder)
# TEST_FILE = "/home/ingaiza/CRNN/dataset/audio/fold7/20_12054.wav"   
TEST_FILE = "/home/ingaiza/CRNN/dataset/audio/val/chainsaw15s.wav"

CLASS_MAP = {
    0: "Natural",
    1: "Unnatural",
    2: "Human Sound"
}

def predict_hybrid():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load CRNN (Feature Extractor)
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

    # 2. Load XGBoost (Classifier)
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
        
        # Looping Fix
        MIN_SAMPLES = 96000 
        if waveform.shape[1] < MIN_SAMPLES:
            print("Looping initiated")
            n_repeats = int(np.ceil(MIN_SAMPLES / waveform.shape[1]))
            waveform = waveform.repeat(1, n_repeats)
            waveform = waveform[:, :MIN_SAMPLES]

        seqs = waveform.permute(1, 0).unsqueeze(0) # (1, time, 1)
        
        lengths = torch.tensor([seqs.shape[1]]).long()
        srs = torch.tensor([16000]).long()
        batch = (seqs.to(device), lengths.to(device), srs.to(device))

    except Exception as e:
        print(f"Audio Error: {e}")
        return

    # 4. HYBRID INFERENCE
    with torch.no_grad():
        # Step A: Get Features from CRNN
        features = crnn(batch, return_features=True)
        features_np = features.cpu().numpy() # Shape (1, 128)

        # Step B: Get Prediction from XGBoost
        probs = xgb_model.predict_proba(features_np)[0]
        pred_idx = np.argmax(probs)

    print("\n--- HYBRID Prediction Complete ---")
    print(f"File: {os.path.basename(TEST_FILE)}")
    print(f"Predicted Class: \033[1m{CLASS_MAP[pred_idx]}\033[0m")
    print(f"Confidence: {probs[pred_idx]:.2%}")
    
    print("\nFull Probabilities:")
    for i, name in CLASS_MAP.items():
        print(f"  {name}: {probs[i]:.2%}")

if __name__ == "__main__":
    predict_hybrid()