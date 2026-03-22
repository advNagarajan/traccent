import os
from pydub import AudioSegment

def convert_folder(folder_path):
    for file in os.listdir(folder_path):

        # ONLY process files that actually exist and are mp3
        if not file.lower().endswith(".mp3"):
            continue

        mp3_path = os.path.join(folder_path, file)

        # double-check existence
        if not os.path.isfile(mp3_path):
            continue

        wav_path = mp3_path.replace(".mp3", ".wav")

        try:
            sound = AudioSegment.from_mp3(mp3_path)
            sound.export(wav_path, format="wav")

            os.remove(mp3_path)

            print("Converted:", file)

        except Exception as e:
            print("Skipping:", file)


def convert_dataset(base_path):
    for label in ["indian", "american", "british"]:
        folder = os.path.join(base_path, label)
        convert_folder(folder)


if __name__ == "__main__":
    convert_dataset("data/controlled")