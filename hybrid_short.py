import sys
from unittest.mock import MagicMock

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
XGB_MODEL_PATH = "/home/ingaiza/CRNN/models/xgboost_audio_classifier.json"
CFG_PATH = "/home/ingaiza/CRNN/crnn.cfg" 
TEST_FILE = "/home/ingaiza/CRNN/dataset/audio/val/group-talking-29731.wav"

CLASS_MAP = {
    0: "Natural",
    1: "Unnatural",
    2: "Human Sound"
}

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
        
        # --- FIX: INCREASE WINDOW SIZE TO 6 SECONDS ---
        # 4 seconds (64k) was too small and caused the crash.
        # 6 seconds (96k) ensures the tensor is large enough for the model.
        WINDOW_SIZE = 96000  
        STRIDE = 48000       # 50% overlap
        
        windows = []
        
        # Case 1: File is shorter than 6 seconds -> Pad it once
        if waveform.shape[1] < WINDOW_SIZE:
            pad_amount = WINDOW_SIZE - waveform.shape[1]
            chunk = F.pad(waveform, (0, pad_amount))
            windows.append(chunk)
            
        # Case 2: File is longer -> Slice it up
        else:
            for i in range(0, waveform.shape[1], STRIDE):
                chunk = waveform[:, i : i + WINDOW_SIZE]
                
                # Handle the tail end of the file
                if chunk.shape[1] < WINDOW_SIZE:
                    # Skip tiny fragments (< 0.5s)
                    if chunk.shape[1] < 8000: 
                        continue
                    
                    # Pad the tail to match WINDOW_SIZE
                    pad_amount = WINDOW_SIZE - chunk.shape[1]
                    chunk = F.pad(chunk, (0, pad_amount))
                
                windows.append(chunk)

        print(f"DEBUG: Generated {len(windows)} windows for analysis.")
        
        window_probs = []

        for chunk in windows:
            # Prepare batch for CRNN
            seqs = chunk.permute(1, 0).unsqueeze(0) 
            
            lengths = torch.tensor([seqs.shape[1]]).long()
            srs = torch.tensor([16000]).long()
            batch = (seqs.to(device), lengths.to(device), srs.to(device))

            with torch.no_grad():
                # Step A: Get Features from CRNN
                features = crnn(batch, return_features=True)
                features_np = features.cpu().numpy() # Shape (1, 128)

                # Step B: Get Prediction from XGBoost
                probs = xgb_model.predict_proba(features_np)[0]
                window_probs.append(probs)

        # Average the probabilities across all windows
        avg_probs = np.mean(window_probs, axis=0)
        pred_idx = np.argmax(avg_probs)

    except Exception as e:
        print(f"Audio Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- HYBRID Prediction Complete ---")
    print(f"File: {os.path.basename(TEST_FILE)}")
    print(f"Predicted Class: \033[1m{CLASS_MAP[pred_idx]}\033[0m")
    print(f"Confidence: {avg_probs[pred_idx]:.2%}")
    
    print("\nFull Probabilities:")
    for i, name in CLASS_MAP.items():
        print(f"  {name}: {avg_probs[i]:.2%}")

if __name__ == "__main__":
    predict_hybrid()