import pandas as pd
import os
import shutil
import random

# paths (UPDATE THESE)
CSV_PATH = "speakers_all.csv"
AUDIO_PATH = r"C:\Users\Aadhav Nagarajan\OneDrive\Desktop\College Stuff\MLAccent\recordings\recordings"
OUTPUT_PATH = "data/controlled"

df = pd.read_csv(CSV_PATH)

# clean NaNs
df = df.dropna(subset=["filename", "country", "native_language"])

def get_label(row):
    country = str(row['country']).lower()
    lang = str(row['native_language']).lower()

    # Indian
    if "india" in country or lang in [
        "hindi", "tamil", "telugu", "malayalam", "kannada",
        "punjabi", "bengali", "gujarati"
    ]:
        return "indian"

    # American
    elif country.strip() in ["usa", "united states"]:
        return "american"

    # British
    elif country.strip() in [
        "uk", "england", "scotland", "wales", "northern ireland"
    ]:
        return "british"

    return None


# apply filter
data = {"indian": [], "american": [], "british": []}

for _, row in df.iterrows():
    label = get_label(row)
    if label:
        data[label].append(row["filename"])

# print counts BEFORE balancing
print("Before balancing:")
for k in data:
    print(k, len(data[k]))

# balance dataset
min_samples = min(len(data["indian"]), len(data["american"]), len(data["british"]))
print("\nBalancing to:", min_samples)

for k in data:
    data[k] = random.sample(data[k], min_samples)

# create folders
for k in data:
    os.makedirs(os.path.join(OUTPUT_PATH, k), exist_ok=True)

# copy files
for k in data:
    for file in data[k]:

        # try possible extensions
        possible_files = [
            file + ".wav",
            file + ".mp3"
        ]

        src = None

        # find correct file
        for f in possible_files:
            temp_path = os.path.join(AUDIO_PATH, f)
            if os.path.exists(temp_path):
                src = temp_path
                break

        # if file not found
        if src is None:
            print("File not found:", file)
            continue

        # destination path (preserve extension)
        dst = os.path.join(OUTPUT_PATH, k, os.path.basename(src))

        try:
            shutil.copy(src, dst)
        except Exception as e:
            print("Error copying:", file, "|", e)

print("\nDone!")