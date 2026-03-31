import os
from preprocess import load_audio_slices
from features import extract_features
import numpy as np

def build_dataset(base_path):
    X = []
    y = []
    groups = []  # file identifier

    for label in os.listdir(base_path):
        folder = os.path.join(base_path, label)

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)

            try:
                slices = load_audio_slices(file_path)

                for s in slices:
                    features = extract_features(s, 22050)

                    X.append(features)
                    y.append(label)
                    
                    # Extract original speaker ID to prevent augmentation leakage
                    original_id = file.split("aug_")[-1] if "aug_" in file else file
                    groups.append(original_id)

            except:
                continue

    return np.array(X), np.array(y), np.array(groups)