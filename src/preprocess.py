import librosa
import numpy as np

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 22050
DURATION = 5  # seconds
SLICE_LENGTH = SAMPLE_RATE * DURATION


# =========================
# LOAD + DISTRIBUTED SLICING
# =========================
def load_audio_slices(file_path):
    """
    Load audio and return multiple slices using distributed sampling:
    positions = [0.2, 0.5, 0.8]

    Each slice is fixed length (5 sec)
    """

    # load audio
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # normalize (avoid volume differences)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # trim silence
    audio, _ = librosa.effects.trim(audio)

    # if audio is too short → pad and return single slice
    if len(audio) < SLICE_LENGTH:
        padded = np.pad(audio, (0, SLICE_LENGTH - len(audio)))
        return [padded]

    slices = []

    # distributed positions (avoid start/end bias)
    positions = [0.2, 0.5, 0.8]

    for p in positions:
        center = int(len(audio) * p)

        start = center - SLICE_LENGTH // 2
        end = start + SLICE_LENGTH

        # fix boundaries
        if start < 0:
            start = 0
            end = SLICE_LENGTH

        if end > len(audio):
            end = len(audio)
            start = end - SLICE_LENGTH

        slice_audio = audio[start:end]

        # final safety check (pad if needed)
        if len(slice_audio) < SLICE_LENGTH:
            slice_audio = np.pad(slice_audio, (0, SLICE_LENGTH - len(slice_audio)))

        slices.append(slice_audio)

    return slices


# =========================
# DEBUG / TEST
# =========================
if __name__ == "__main__":
    test_file = "data/controlled/indian/sample.wav"  # change this

    slices = load_audio_slices(test_file)

    print("Number of slices:", len(slices))
    print("Slice length:", len(slices[0]))