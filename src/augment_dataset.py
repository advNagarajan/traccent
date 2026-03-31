import librosa
import soundfile as sf
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

INPUT_PATH = "data/controlled"
OUTPUT_PATH = "data/augmented"

# Augmentation strategies to generate "synthetic" speakers
def augment_audio(y, sr):
    augmented_versions = []
    
    # 1. Pitch Shift (synthesizes new vocal cord structure)
    y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2.0)
    y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2.0)
    augmented_versions.extend([y_pitch_up, y_pitch_down])
    
    # 2. Time Stretch (synthesizes speech pacing changes)
    y_fast = librosa.effects.time_stretch(y, rate=1.2)
    y_slow = librosa.effects.time_stretch(y, rate=0.8)
    augmented_versions.extend([y_fast, y_slow])
    
    # 3. Add Background White Noise
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y + noise_amp * np.random.normal(size=y.shape[0])
    augmented_versions.append(y_noise)
    
    return augmented_versions

def generate_dataset():
    print("Initializing synthetic speaker augmentation pipeline...")
    
    for accent in os.listdir(INPUT_PATH):
        input_folder = os.path.join(INPUT_PATH, accent)
        output_folder = os.path.join(OUTPUT_PATH, accent)
        
        if not os.path.isdir(input_folder): continue
        os.makedirs(output_folder, exist_ok=True)
        
        files = os.listdir(input_folder)
        print(f"Processing {accent} ({len(files)} original files)...")
        
        for file in files:
            file_path = os.path.join(input_folder, file)
            
            # Save original file
            y, sr = librosa.load(file_path, sr=22050)
            sf.write(os.path.join(output_folder, f"{file}"), y, sr)
            
            # Generate and save synthetically augmented speakers
            aug_audios = augment_audio(y, sr)
            for i, aug_y in enumerate(aug_audios):
                aug_name = f"aug_{i}_{file}"
                sf.write(os.path.join(output_folder, aug_name), aug_y, sr)
                
    print(f"\nCompleted! Dataset successfully expanded from {len(files)} to {len(files) * 6} speakers per class in {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_dataset()
