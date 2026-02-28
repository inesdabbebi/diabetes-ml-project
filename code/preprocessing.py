import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings("ignore")

DATA_PATH   = "data/Diabetes_Dataset.csv"
FIGURES_DIR = "reports/figures"
MODELS_DIR  = "models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────
print("=" * 55)
print("  PREPROCESSING & IMBALANCE HANDLING")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
print(f"\n✔ Loaded dataset: {df.shape}")

# ── 2. Encode Target ─────────────────────────────────────
df["Outcome"] = (df["Outcome"] == "Diabetic").astype(int)
print(f"\nTarget encoding: Diabetic=1, Non-diabetic=0")
print(df["Outcome"].value_counts())

# ── 3. Encode Categorical Features ──────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"\nCategorical columns to encode: {cat_cols}")

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Save label encoders
joblib.dump(label_encoders, f"{MODELS_DIR}/label_encoders.pkl")
print(f"\n✔ Label encoders saved → models/label_encoders.pkl")

# ── 4. Feature / Target Split ────────────────────────────
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
print(f"\nFeatures shape : {X.shape}")
print(f"Target shape   : {y.shape}")
print(f"Feature list   : {X.columns.tolist()}")

# ── 5. Train / Test Split ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Train class dist:\n{y_train.value_counts()}")
print(f"Test  class dist:\n{y_test.value_counts()}")

# ── 6. Feature Scaling ───────────────────────────────────
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
print(f"\n✔ StandardScaler fitted & saved → models/scaler.pkl")

# ── 7. SMOTE — Handle Class Imbalance ───────────────────
print("\n── SMOTE Oversampling ────────────────────────────────")
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
y_train_resampled = pd.Series(y_train_resampled, name="Outcome")

print(f"After  SMOTE: {y_train_resampled.value_counts().to_dict()}")
print(f"Resampled train shape: {X_train_resampled.shape}")

# ── 8. Visualize Imbalance Before / After ───────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ["#4C9BE8", "#E8694C"]

# Before
before_counts = y_train.value_counts()
axes[0].bar(["d (0)", "Diabetic (1)"], before_counts.values, color=colors, edgecolor="white")
axes[0].set_title("Before SMOTE (Training Set)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count")
for bar, val in zip(axes[0].patches, before_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{val:,}", ha="center", fontsize=11)

# After
after_counts = y_train_resampled.value_counts()
axes[1].bar(["d (0)", "Diabetic (1)"], after_counts.values, color=colors, edgecolor="white")
axes[1].set_title("After SMOTE (Balanced Training Set)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Count")
for bar, val in zip(axes[1].patches, after_counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{val:,}", ha="center", fontsize=11)

plt.suptitle("Class Imbalance — Before vs After SMOTE", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/08_class_balance_before_after.png", dpi=150)
plt.close()
print(f"\n✔ Saved → reports/figures/08_class_balance_before_after.png")

# ── 9. Save Processed Data ───────────────────────────────
X_train_scaled.to_csv("data/X_train.csv", index=False)
X_test_scaled.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
X_train_resampled.to_csv("data/X_train_resampled.csv", index=False)
y_train_resampled.to_csv("data/y_train_resampled.csv", index=False)

print("\n Saved processed data files:")
print("   data/X_train.csv, X_test.csv")
print("   data/y_train.csv, y_test.csv")
print("   data/X_train_resampled.csv, y_train_resampled.csv")

print("\n  Preprocessing complete!")