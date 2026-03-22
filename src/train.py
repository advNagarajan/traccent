import joblib
import numpy as np
import os

from dataset import build_dataset
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# =========================
# LOAD DATA (WITH GROUPS)
# =========================
X, y, groups = build_dataset("data/controlled")

print("Dataset shape:", X.shape)

# =========================
# GROUP-BASED SPLIT (CRITICAL)
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# GRID SEARCH (SVM)
# =========================
'''param_grid = {
    'C': [5, 10, 20],
    'gamma': ['scale', 0.01]
}
grid = GridSearchCV(
    SVC(kernel='linear', probability=True, class_weight='balanced'),
    param_grid,
    cv=3,
    verbose=1,
    n_jobs=-1
)

print("\nRunning GridSearch...")
grid.fit(X_train, y_train)

model = grid.best_estimator_'''

model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,
    class_weight='balanced'
)

model.fit(X_train, y_train)

#print("\nBest Parameters:", grid.best_params_)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL + SCALER
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as model.pkl")
print("Scaler saved as scaler.pkl")