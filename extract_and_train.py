import sys
from unittest.mock import MagicMock

# 1. Mock tensorboardX (Required for your specific checkpoint)
sys.modules["tensorboardX"] = MagicMock()

import torch
import os
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
import torch.nn.functional as F
from net.model import AudioCRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pandas as pd

# CONFIG
MODEL_PATH = "/home/ingaiza/CRNN/CRNN_training_output2-20251112T081448Z-1-001/CRNN_training_output2/1111_140703/checkpoints/model_best.pth"
DATA_DIR = "/home/ingaiza/CRNN/dataset/audio" 
CFG_PATH = "/home/ingaiza/CRNN/crnn.cfg"

def preprocess_audio(audio_path):
    try:
        data, sr = sf.read(audio_path)
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
            n_repeats = int(np.ceil(MIN_SAMPLES / waveform.shape[1]))
            waveform = waveform.repeat(1, n_repeats)
            waveform = waveform[:, :MIN_SAMPLES]

        waveform = waveform.permute(1, 0).unsqueeze(0) # (1, time, 1)
        return waveform
    except:
        return None

def main():
    print("1. Loading CRNN Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    classes = ["Natural", "Unnatural", "Human Sound"] 
    config = {'cfg': CFG_PATH, 'transforms': {'args': {'channels': 'mono'}}}
    
    model = AudioCRNN(classes=classes, config=config)
    
    # --- FIX 1: ROBUST CHECKPOINT LOADING ---
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print("   Model loaded successfully.")
    except Exception as e:
        print(f"   CRITICAL ERROR Loading Model: {e}")
        return
    
    model.eval().to(device)

    print("2. Extracting Features from Dataset...")
    X_features = []
    y_labels = []

    # Load CSV
    csv_path = "/home/ingaiza/CRNN/dataset/metadata/UrbanSound8K.csv" 
    df = pd.read_csv(csv_path)
    
    total_files = len(df)
    print(f"   Found {total_files} entries in CSV.")
    
    for idx, row in df.iterrows():
        filename = row['slice_file_name'] 
        label = row['classID'] 
        fold = row['fold']
        
        # --- FIX 2: CLEANER PATH LOGIC ---
        # Construct path using the 'fold' column directly
        full_path = os.path.join(DATA_DIR, f"fold{fold}", filename)
        
        if not os.path.exists(full_path):
            # Optional: print missing files occasionally
            # print(f"Missing: {full_path}")
            continue

        # Process Audio
        seqs = preprocess_audio(full_path)
        if seqs is None: continue

        lengths = torch.tensor([seqs.shape[1]]).long()
        srs = torch.tensor([16000]).long()
        batch = (seqs.to(device), lengths.to(device), srs.to(device))

        # Extract Features
        with torch.no_grad():
            # features is (1, 64)
            features = model(batch, return_features=True)
            X_features.append(features.cpu().numpy().flatten())
            y_labels.append(label)
            
        if idx % 500 == 0:
            print(f"   Processed {idx}/{total_files}")

    X = np.array(X_features)
    y = np.array(y_labels)
    
    print(f"Extraction Complete. Feature Shape: {X.shape}")

    print("3. Training XGBoost Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        objective='multi:softprob',
        num_class=3
    )
    
    clf.fit(X_train, y_train)

    print("4. Evaluation:")
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"   XGBoost Accuracy: {acc:.2%}")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    
    clf.save_model("xgboost_audio_classifier.json")
    print("Saved second-tier model to xgboost_audio_classifier.json")

if __name__ == "__main__":
    main()