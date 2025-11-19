import sys
import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
import torch.nn.functional as F
import xgboost as xgb
import random
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from unittest.mock import MagicMock

# --- 1. CONFIGURATION & PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assumes script is in CRNN root
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset/audio")
CSV_PATH = os.path.join(PROJECT_ROOT, "dataset/metadata/mydataset.csv") # Ensure this path is correct
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/model_best.pth")
CFG_PATH = os.path.join(PROJECT_ROOT, "crnn.cfg")
OUTPUT_MODEL_PATH = "xgboost_mixed_model.json"

# How many synthetic samples to create?
NUM_AUGMENTED_SAMPLES = 6000 
TARGET_SAMPLE_RATE = 16000
TARGET_LENGTH = 240000 # 15 seconds

# --- 2. MOCKING (To load CRNN) ---
# sys.modules["tensorboardX"] = MagicMock()
# try:
#     import utils
#     sys.modules["utils"] = utils
#     sys.modules["utils.logger"] = utils.logger
#     sys.modules["utils.util"] = utils.util
# except ImportError:
#     # Fallback if utils folder isn't found (Colab/etc)
#     pass 

# Add 'net' to path if needed
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from net.model import AudioCRNN

# --- 3. DATA LOADING & HELPERS ---

def load_dataset_map(csv_path):
    """Reads CSV and returns a dict: {classID: [list_of_file_paths]}"""
    df = pd.read_csv(csv_path)
    files_by_class = {0: [], 1: [], 2: []}
    
    print(f"Loading dataset from {csv_path}...")
    for _, row in df.iterrows():
        # Construct full path: dataset/audio/foldX/filename.wav
        fold_dir = f"fold{row['fold']}"
        filename = row['slice_file_name']
        full_path = os.path.join(DATA_DIR, fold_dir, filename)
        
        cls_id = row['classID']
        if os.path.exists(full_path):
            files_by_class[cls_id].append(full_path)
            
    for cid, files in files_by_class.items():
        print(f"  Class {cid}: {len(files)} files")
    return files_by_class

def preprocess_audio(path):
    """Loads, resamples, mono-mixes, and loops audio to 15s."""
    try:
        data, sr = sf.read(path)
        waveform = torch.from_numpy(data).float()
        
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        else: waveform = waveform.permute(1, 0)
        
        if sr != TARGET_SAMPLE_RATE:
            resampler = T.Resample(sr, TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Loop/Pad to target length
        if waveform.shape[1] < TARGET_LENGTH:
            n_repeats = int(np.ceil(TARGET_LENGTH / waveform.shape[1]))
            waveform = waveform.repeat(1, n_repeats)
        
        return waveform[:, :TARGET_LENGTH] # Trim to exact 15s
    except Exception as e:
        return None

def create_mix(class_files_map, presence_vector):
    """
    Creates a mixed audio tensor based on presence vector [N, U, H].
    presence_vector: tuple (0/1, 0/1, 0/1) indicating which classes are present.
    """
    signals = []
    
    # presence_vector indices: 0=Natural, 1=Unnatural, 2=Human
    for class_id, is_present in enumerate(presence_vector):
        if is_present:
            # Pick a random file from this class
            file_path = random.choice(class_files_map[class_id])
            wav = preprocess_audio(file_path)
            if wav is not None:
                # Random gain (0.4 to 1.0) to simulate depth
                gain = random.uniform(0.4, 1.0)
                signals.append(wav * gain)
    
    # Handle empty case (0,0,0) -> Treat as low-volume Natural
    if not signals:
        file_path = random.choice(class_files_map[0]) # Natural
        wav = preprocess_audio(file_path)
        signals.append(wav * 0.1) # Very quiet
        
    # Stack and Sum
    if len(signals) > 1:
        # Pad all to max length in list (they should all be 15s though)
        mixed = torch.stack(signals).sum(dim=0)
    else:
        mixed = signals[0]

    # Normalize to [-1, 1]
    max_amp = torch.max(torch.abs(mixed))
    if max_amp > 0:
        mixed = mixed / max_amp
        
    return mixed.unsqueeze(0) # Add batch dim -> (1, 1, 240000)

def determine_label(presence_vector):
    """
    Truth Table Logic:
    Priority: Unnatural > Human > Natural
    [N, U, H]
    """
    n, u, h = presence_vector
    if u == 1: return 1 # Unnatural (Overpowers everything)
    if h == 1: return 2 # Human (Overpowers Natural)
    return 0            # Natural (Default)

# --- 4. MAIN SCRIPT ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load CRNN Feature Extractor
    print("Loading CRNN...")
    class_map_values = ["Natural", "Unnatural", "Human Sound"]
    config = {'cfg': CFG_PATH, 'transforms': {'args': {'channels': 'mono'}}}
    crnn = AudioCRNN(classes=class_map_values, config=config)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    crnn.load_state_dict(state_dict)
    crnn.eval().to(device)
    
    # 2. Load Dataset Map
    files_map = load_dataset_map(CSV_PATH)
    
    # 3. Generate Augmented Dataset
    print(f"Generating {NUM_AUGMENTED_SAMPLES} mixed samples...")
    X_features = []
    y_labels = []
    
    # Truth table combinations: (0,0,0) to (1,1,1)
    combinations = list(itertools.product([0, 1], repeat=3)) # 8 combinations
    
    for i in range(NUM_AUGMENTED_SAMPLES):
        # Pick a random scenario from the 8 truth table rows
        scenario = random.choice(combinations) # e.g., (1, 0, 1)
        
        # Create Audio Mix
        mixed_wav = create_mix(files_map, scenario)
        
        # Determine Ground Truth Label based on priority
        label = determine_label(scenario)
        
        # Extract Features using CRNN
        # Prepare Batch: (1, time, 1)
        seqs = mixed_wav.permute(0, 2, 1) # (1, 240000, 1)
        lengths = torch.tensor([seqs.shape[1]]).long()
        srs = torch.tensor([16000]).long()
        batch = (seqs.to(device), lengths.to(device), srs.to(device))
        
        with torch.no_grad():
            feats = crnn(batch, return_features=True) # (1, 128)
            X_features.append(feats.cpu().numpy().flatten())
            y_labels.append(label)
            
        if (i+1) % 500 == 0:
            print(f"  Generated {i+1}/{NUM_AUGMENTED_SAMPLES} samples")

    X = np.array(X_features)
    y = np.array(y_labels)
    
    print(f"Dataset ready. Shape: {X.shape}")
    
    # 4. Train XGBoost
    print("Training XGBoost...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        objective='multi:softprob',
        num_class=3,
        tree_method='hist', # Use 'gpu_hist' if you have GPU for XGB
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = clf.predict(X_test)
    print("\n--- Evaluation on Mixed Data ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
    print(classification_report(y_test, preds, target_names=["Natural", "Unnatural", "Human"]))
    
    # 6. Save
    clf.save_model(OUTPUT_MODEL_PATH)
    print(f"Model saved to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()