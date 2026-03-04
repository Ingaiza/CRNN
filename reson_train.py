import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchaudio.transforms as T
from net.model import AudioCRNN
from data.data_manager import CSVDataManager

# --- 1. CONFIGURATION ---
CFG_PATH = "/home/ingaiza/CRNN/crnn.cfg"  # REMEMBER: Ensure [dense_module] out_features = 10
DATA_DIR = "/home/ingaiza/CRNN/dataset"
MODEL_SAVE_PATH = "haptihear_crnn_1sec_best.pth"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
SAMPLE_RATE = 16000
WINDOW_SIZE = 16000 # 1 second of audio at 16kHz

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. CUSTOM 1-SECOND TRANSFORM ---
class OneSecondAudioTransform:
    """
    Forces all incoming audio from the DataLoader into exactly 1-second windows.
    Pads short audio and truncates long audio.
    """
    def __init__(self, target_samples=WINDOW_SIZE, target_sr=SAMPLE_RATE):
        self.target_samples = target_samples
        self.target_sr = target_sr

    def __call__(self, waveform):
        # waveform expected shape: (channels, time)
        
        # 1. Convert to Mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 2. Pad or Truncate to exactly target_samples (1 second)
        current_samples = waveform.shape[1]
        
        if current_samples < self.target_samples:
            # Pad with repeating audio (looping fix)
            n_repeats = int(np.ceil(self.target_samples / current_samples))
            waveform = waveform.repeat(1, n_repeats)
            waveform = waveform[:, :self.target_samples]
        elif current_samples > self.target_samples:
            # Truncate
            waveform = waveform[:, :self.target_samples]
            
        # Return expected shape for the CRNN batching: (time, channels)
        return waveform.permute(1, 0)

# --- 3. INITIALIZATION ---
def main():
    print(f"Running on device: {DEVICE}")

    # Set up the DataManager configuration
    data_config = {
        'path': DATA_DIR,
        'format': 'audio',
        'splits': {
            'train': [1, 2, 3, 4, 5, 6, 7, 8, 9], # Folds 1-9 for training
            'val': [10]                           # Fold 10 for validation
        },
        'loader': {
            'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': 4
        }
    }

    print("Loading DataManagers...")
    data_manager = CSVDataManager(data_config)
    audio_transform = OneSecondAudioTransform()
    
    train_loader = data_manager.get_loader('train', transfs=audio_transform)
    val_loader = data_manager.get_loader('val', transfs=audio_transform)

    # Initialize Model from Scratch (No Checkpoint)
    print("Initializing CRNN from scratch...")
    # Passing dummy classes list of length 10 to match UrbanSound8K
    dummy_classes = [str(i) for i in range(10)] 
    model_config = {'cfg': CFG_PATH, 'transforms': {'args': {'channels': 'mono'}}}
    
    model = AudioCRNN(classes=dummy_classes, config=model_config).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. TRAINING LOOP ---
    
    
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, batch in enumerate(train_loader):
            # CSVDataManager's pad_seq returns: seqs_pad, lengths, srs, labels
            seqs, lengths, srs, labels = batch
            
            # Format batch for AudioCRNN's forward method
            crnn_batch = (seqs.to(DEVICE), lengths.to(DEVICE), srs.to(DEVICE))
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            
            # Forward pass (end-to-end, no feature extraction)
            outputs = model(crnn_batch) 
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        
        # --- 5. VALIDATION ---
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                seqs, lengths, srs, labels = batch
                crnn_batch = (seqs.to(DEVICE), lengths.to(DEVICE), srs.to(DEVICE))
                labels = labels.to(DEVICE)

                outputs = model(crnn_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   -> New best model saved to {MODEL_SAVE_PATH}")

    print("Training Complete!")

if __name__ == "__main__":
    main()