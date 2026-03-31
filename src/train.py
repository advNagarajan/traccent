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
# STANDARD SPLIT (FOR METRICS)
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
# MODEL TRAINING (ENSEMBLE + FEATURE SELECTION)
# =========================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced', random_state=42)

voting = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
    voting='soft',
    n_jobs=-1
)

model = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif, k=150)),
    ('classifier', voting)
])

print("\nRunning Ensemble Training with Feature Selection...")
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
from sklearn.metrics import roc_auc_score

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Calculate Multi-Class AUC
auc_score = roc_auc_score(y_test, y_probs, multi_class='ovr')
print(f"ROC AUC Score (OVR): {auc_score:.4f}\n")

# =========================
# SAVE MODEL + SCALER
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as model.pkl")
print("Scaler saved as scaler.pkl")