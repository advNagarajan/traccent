import joblib
import numpy as np
import os

from preprocess import load_audio_slices
from features import extract_features

# =========================
# LOAD MODEL + SCALER
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


import librosa

# =========================
# SINGLE FILE PREDICTION (DENSE SLIDING WINDOW)
# =========================
def predict_file(file_path):
    # Load the raw audio
    audio, sr = librosa.load(file_path, sr=22050)
    
    # Normalize volume and strip total digital silence
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    audio, _ = librosa.effects.trim(audio, top_db=30)
    
    window_length = sr * 5  # 5 seconds
    step_size = sr * 1      # 1 second stride
    
    all_probs = []
    
    # If file is too short, just pad and process once
    if len(audio) < window_length:
        padded = np.pad(audio, (0, window_length - len(audio)))
        features = extract_features(padded, sr).reshape(1, -1)
        features = scaler.transform(features)
        all_probs.append(model.predict_proba(features)[0])
    else:
        # Sliding dense window
        for start in range(0, len(audio) - window_length + 1, step_size):
            window = audio[start:start + window_length]
            features = extract_features(window, sr).reshape(1, -1)
            features = scaler.transform(features)
            
            probs = model.predict_proba(features)[0]
            
            # Confidence gating: Only keep this window if the model is decently confident (>40%) 
            # This throws out pure static or heavy breathing windows
            if np.max(probs) > 0.40:
                all_probs.append(probs)
                
    # If the file was so bad everything was dropped, fall back
    if len(all_probs) == 0:
        return "Unknown", 0.0, {}

    # Average the high-confidence windows
    avg_probs = np.mean(all_probs, axis=0)

    pred = model.classes_[np.argmax(avg_probs)]
    confidence = np.max(avg_probs)

    # create readable breakdown
    probs_dict = dict(zip(model.classes_, avg_probs))

    return pred, confidence, probs_dict


# =========================
# FOLDER PREDICTION
# =========================
def predict_folder(folder_path):
    results = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        # skip non-audio files
        if not file.lower().endswith((".wav", ".mp3")):
            continue

        try:
            pred, conf, probs = predict_file(file_path)

            results.append({
                "file": file,
                "prediction": pred,
                "confidence": round(conf, 3),
                "probabilities": probs
            })

        except Exception as e:
            print("Error processing:", file)

    return results


# =========================
# TEST / CLI RUN
# =========================
if __name__ == "__main__":
    
    folder = r"C:\Users\aadha\Desktop\College Stuff\mllab\traccent\input"  # put your test audio files here

    results = predict_folder(folder)

    for r in results:
        print(f"\n=> File: {r['file']}")
        print(f"Prediction: {r['prediction'].upper()}")
        print("Probabilities:")
        for accent, prob in r['probabilities'].items():
            print(f"  - {accent.capitalize()}: {prob * 100:.1f}%")