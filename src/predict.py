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


# =========================
# SINGLE FILE PREDICTION
# =========================
def predict_file(file_path):
    slices = load_audio_slices(file_path)

    all_probs = []

    for s in slices:
        features = extract_features(s, 22050).reshape(1, -1)

        # IMPORTANT: scale features
        features = scaler.transform(features)

        probs = model.predict_proba(features)[0]
        all_probs.append(probs)

    # average probabilities across slices
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
    
    folder = r"C:\Users\Aadhav Nagarajan\OneDrive\Desktop\College Stuff\MLAccent\input"  # put your test audio files here

    results = predict_folder(folder)

    for r in results:
        print("\nFile:", r["file"])
        print("Prediction:", r["prediction"])
        print("Confidence:", r["confidence"])
        print("Probabilities:", r["probabilities"])