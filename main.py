import os

folder = "data/controlled/indian"

files = os.listdir(folder)

print("Total files:", len(files))
print("Sample files:", files[:10])