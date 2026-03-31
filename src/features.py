import librosa
import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(audio, sr):

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    
    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    
    features = []
    
    # helper for stats
    def get_stats(mat):
        return [
            np.mean(mat, axis=1),
            np.std(mat, axis=1),
            np.min(mat, axis=1),
            np.max(mat, axis=1),
            np.median(mat, axis=1)
        ]
        
    for mat in [mfcc, delta, chroma, contrast]:
        for stat in get_stats(mat):
            features.extend(stat)
            
    # 1D features
    features.extend([
        np.mean(centroid), np.std(centroid),
        np.mean(rolloff), np.std(rolloff),
        np.mean(zcr), np.std(zcr)
    ])
    
    return np.array(features)