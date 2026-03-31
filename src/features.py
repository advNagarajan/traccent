import librosa
import numpy as np

def extract_features(audio, sr):

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)

    # MFCC stats
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_min = np.min(mfcc, axis=1)
    mfcc_max = np.max(mfcc, axis=1)

    # Delta stats
    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    # NEW FEATURES (small but useful)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    features = np.hstack([
        mfcc_mean,
        mfcc_std,
        mfcc_min,
        mfcc_max,
        delta_mean,
        delta_std,
        zcr,
        centroid
    ])

    return features