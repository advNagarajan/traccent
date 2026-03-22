from dataset import build_dataset

X, y = build_dataset("data/controlled")

print(X.shape)
print(y[:10])