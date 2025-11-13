import sys
from unittest.mock import MagicMock

# 1. Mock tensorboardX
sys.modules["tensorboardX"] = MagicMock()

import torch
import os
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
import torch.nn.functional as F
from net.model import AudioCRNN

MODEL_PATH = "/home/ingaiza/CRNN/CRNN_training_output2-20251112T081448Z-1-001/CRNN_training_output2/1111_140703/checkpoints/model_best.pth" 
AUDIO_PATH = "/home/ingaiza/CRNN/dataset/audio/fold7/17_11740.wav" 
CFG_PATH = "/home/ingaiza/CRNN/crnn.cfg" 

CLASS_MAP = {
    0: "Natural",
    1: "Unnatural",
    2: "Human Sound"
}

def predict(audio_path, model_path, cfg_path, class_map):
    print(f"Loading model from {model_path}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = {
        'cfg': cfg_path,
        'transforms': {'args': {'channels': 'mono'}}
    }
    
    model = AudioCRNN(classes=class_map.values(), config=config)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    model.to(device)
    
    print(f"Loading and preprocessing audio: {audio_path}")
    
    try:
        # Read audio
        data, sr = sf.read(audio_path)
        waveform = torch.from_numpy(data).float()
        
        print(f"DEBUG: Original raw shape: {waveform.shape}, Sample Rate: {sr}")

        # Standardize shape to (1, time)
        if waveform.ndim == 1:
            # Mono (time) -> (1, time)
            waveform = waveform.unsqueeze(0)
        else:
            # Stereo/Multi (time, channels) -> (channels, time)
            waveform = waveform.permute(1, 0)

        # Resample to 16000Hz
        if sr != 16000:
            print("DEBUG: Resampling to 16000Hz...")
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        # Mix to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # --- LOOPING FIX (Better for Accuracy) ---
        # Instead of adding silence, we repeat the audio to fill the time.
        MIN_SAMPLES = 96000 
        if waveform.shape[1] < MIN_SAMPLES:
            print(f"DEBUG: Audio short ({waveform.shape[1]} samples). Looping to fill...")
            
            # Calculate how many times we need to repeat
            n_repeats = int(np.ceil(MIN_SAMPLES / waveform.shape[1]))
            
            # Repeat the waveform
            waveform = waveform.repeat(1, n_repeats)
            
            # Trim off any excess if we went slightly over MIN_SAMPLES
            waveform = waveform[:, :MIN_SAMPLES]
        
        # Prepare for model: (1, time, 1)
        # Current: (1, time)
        # Permute to (time, 1)
        waveform = waveform.permute(1, 0)
        # Unsqueeze to (1, time, 1)
        seqs = waveform.unsqueeze(0)
        
        print(f"DEBUG: Final input shape entering model: {seqs.shape}")

    except Exception as e:
        print(f"Error loading audio file: {e}")
        import traceback
        traceback.print_exc()
        return

    lengths = torch.tensor([seqs.shape[1]]).long()
    srs = torch.tensor([sr]).long()
    
    batch = (
        seqs.to(device),
        lengths.to(device),
        srs.to(device)
    )

    print("Running prediction...")
    with torch.no_grad():
        output = model(batch)
        probabilities = torch.exp(output).cpu().numpy().flatten()
        
        prediction_index = np.argmax(probabilities)
        predicted_class = class_map[prediction_index]
        confidence = probabilities[prediction_index]

    print("\n--- Prediction Complete ---")
    print(f"File: {os.path.basename(audio_path)}")
    print(f"Predicted Class: \033[1m{predicted_class}\033[0m")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nFull Probabilities:")
    for i, class_name in class_map.items():
        print(f"  {class_name}: {probabilities[i]:.2%}")

if __name__ == "__main__":
    predict(AUDIO_PATH, MODEL_PATH, CFG_PATH, CLASS_MAP)