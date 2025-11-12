import torch
import librosa
import os
import numpy as np
from net.model import AudioCRNN
from torchparse import parse_cfg
import torch.nn.functional as F


MODEL_PATH = "/home/ingaiza/CRNN/CRNN_training_output2-20251112T081448Z-1-001/CRNN_training_output2/1111_140703/checkpoints/model_best.pth" 

AUDIO_PATH = "/home/ingaiza/CRNN/dataset/audio/fold7/10_11001.wav" # UNNATURAL

CFG_PATH = "/home/ingaiza/CRNN/crnn.cfg" 

CLASS_MAP = {
    0: "Natural",
    1: "Unnatural",
    2: "Human Sound"
}

def predict(audio_path, model_path, cfg_path, class_map):
    """
    Loads a trained CRNN model and predicts the class of a single audio file.
    """
    
    print(f"Loading model from {model_path}...")
    
    # Check if we're on a GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the architecture config file
    config = {'cfg': parse_cfg(cfg_path)}
    
    # Instantiate the model with your 3 classes
    model = AudioCRNN(classes=class_map.values(), config=config)
    
    # Load the trained weights from the checkpoint
    # map_location=device ensures it loads correctly whether you're on CPU or GPU
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        print("Please make sure 'MODEL_PATH' is set correctly.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure your local code files (model.py, crnn.cfg) are patched.")
        return

    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    model.to(device)
    
    # --- 4. LOAD AND PREPROCESS THE AUDIO FILE ---
    print(f"Loading and preprocessing audio: {audio_path}")
    
    try:
        # Load audio file with librosa, resample to 16kHz
        # We use 16000 Hz as it's a common standard and the default
        # for torchaudio's MelSpectrogram.
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    except FileNotFoundError:
        print(f"ERROR: Audio file not found at {audio_path}")
        print("Please make sure 'AUDIO_PATH' is set correctly.")
        return
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Convert to PyTorch tensor
    waveform = torch.tensor(waveform).float()
    
    # The model expects a "batch" of 4 items:
    # 1. Waveform tensor: (batch, time, channel) -> (1, num_samples, 1)
    seqs = waveform.unsqueeze(0).unsqueeze(-1)
    
    # 2. Lengths tensor: (batch)
    lengths = torch.tensor([seqs.shape[1]]).long()
    
    # 3. Sample rate tensor: (batch)
    srs = torch.tensor([sr]).long()
    
    # 4. Dummy labels tensor: (batch)
    labels = torch.tensor().long() # Not used for inference

    # Create the batch tuple and move to device
    batch = (
        seqs.to(device),
        lengths.to(device),
        srs.to(device),
        labels.to(device)
    )

    # --- 5. RUN INFERENCE ---
    print("Running prediction...")
    with torch.no_grad():
        # The model's forward pass returns log_softmax values
        output = model(batch)
        
        # Convert log_softmax to probabilities (0.0 to 1.0)
        probabilities = torch.exp(output).cpu().numpy()
        
        # Get the index of the highest probability
        prediction_index = np.argmax(probabilities)
        
        # Get the class name and confidence
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
    if AUDIO_PATH == "path/to/your/test_file.wav":
        print("="*50)
        print("ERROR: Please open predict.py and change the")
        print("       'AUDIO_PATH' variable to point to your audio file.")
        print("="*50)
    else:
        predict(AUDIO_PATH, MODEL_PATH, CFG_PATH, CLASS_MAP)