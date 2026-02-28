import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model  import LogisticRegression
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.ensemble      import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
FIGURES_DIR = "reports/figures"
MODELS_DIR  = "models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Load Data ────────────────────────────────────────────
print("=" * 55)
print("  MODELING — CLASSIFICATION")
print("=" * 55)

X_train = pd.read_csv("data/X_train_resampled.csv")
y_train = pd.read_csv("data/y_train_resampled.csv").squeeze()
X_test  = pd.read_csv("data/X_test.csv")
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print(f"\n✔ Train: {X_train.shape}  |  Test: {X_test.shape}")

# ════════════════════════════════════════════════════════
#  CLASSIFICATION
# ════════════════════════════════════════════════════════
print("\n" + "═" * 55)
print("  CLASSIFICATION")
print("═" * 55)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.01),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
}

mlflow.set_experiment("Diabetes_Classification")

results    = {}
fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
colors_roc = ["#4C9BE8", "#E8694C", "#2ECC71"]

for (name, model), color in zip(models.items(), colors_roc):
    print(f"\n── Training: {name} ─────────────────────────────────")

    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred, average="weighted")
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Non-diabetic','Diabetic'])}")

        # Log to MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, name.replace(" ", "_"))

        results[name] = {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}

        # Save model
        joblib.dump(model, f"{MODELS_DIR}/{name.replace(' ', '_')}.pkl")
        print(f"  ✔ Model saved → models/{name.replace(' ', '_')}.pkl")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["Non-diabetic", "Diabetic"],
                    yticklabels=["Non-diabetic", "Diabetic"])
        ax_cm.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_xlabel("Predicted")
        plt.tight_layout()
        cm_path = f"{FIGURES_DIR}/cm_{name.replace(' ', '_')}.png"
        plt.savefig(cm_path, dpi=150)
        mlflow.log_artifact(cm_path)
        plt.close()
        print(f"  ✔ Saved confusion matrix")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC = {roc_auc:.3f})")

# ROC combined
ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax_roc.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/09_roc_curves_all.png", dpi=150)
plt.close()
print(f"\n✔ Saved → reports/figures/09_roc_curves_all.png")

# Model Comparison
results_df = pd.DataFrame(results).T.round(4)
print("\n\n── Model Comparison ──────────────────────────────────")
print(results_df.to_string())
results_df.to_csv("reports/model_comparison.csv")

fig, ax = plt.subplots(figsize=(10, 5))
results_df[["accuracy", "f1", "roc_auc"]].plot(
    kind="bar", ax=ax,
    color=["#4C9BE8", "#E8694C", "#2ECC71"],
    edgecolor="white", width=0.65
)
ax.set_title("Model Comparison — Classification Metrics", fontsize=13, fontweight="bold")
ax.set_ylabel("Score")
ax.set_xticklabels(results_df.index, rotation=15, ha="right")
ax.set_ylim(0.5, 1.05)
ax.legend(["Accuracy", "F1-Score", "ROC-AUC"])
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/10_model_comparison.png", dpi=150)
plt.close()
print(f"✔ Saved → reports/figures/10_model_comparison.png")

# Feature Importance — Random Forest
rf = joblib.load(f"{MODELS_DIR}/Random_Forest.pkl")
importances = pd.Series(
    rf.feature_importances_, index=X_train.columns
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
importances.plot(kind="barh", color="#4C9BE8", edgecolor="white", ax=ax)
ax.set_title("Feature Importances — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/11_feature_importance_rf.png", dpi=150)
plt.close()
print(f"✔ Saved → reports/figures/11_feature_importance_rf.png")

# KNN: Finding Best K
print("\n── Finding Best K for KNN ────────────────────────────")
k_scores = []
k_range  = range(1, 21)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    k_scores.append(accuracy_score(y_test, knn.predict(X_test)))
    print(f"  k={k:2d} | Accuracy={k_scores[-1]:.4f}")

best_k_knn = list(k_range)[np.argmax(k_scores)]
print(f"\n  Best k = {best_k_knn} (Accuracy={max(k_scores):.4f})")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_range, k_scores, "o-", color="#4C9BE8", lw=2, markersize=6)
ax.axvline(best_k_knn, color="#E8694C", linestyle="--", label=f"Best k={best_k_knn}")
ax.set_title("KNN — Accuracy vs Number of Neighbors (k)", fontsize=13, fontweight="bold")
ax.set_xlabel("k (Number of Neighbors)")
ax.set_ylabel("Accuracy")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/12_knn_best_k.png", dpi=150)
plt.close()
print(f"✔ Saved → reports/figures/12_knn_best_k.png")

# ── Final Summary ────────────────────────────────────────
print("\n" + "═" * 55)
print("  FINAL SUMMARY")
print("═" * 55)
best_model = results_df["roc_auc"].idxmax()
print(f"\n🏆 Best Model   : {best_model}")
print(f"   Accuracy     : {results_df.loc[best_model, 'accuracy']:.4f}")
print(f"   F1-Score     : {results_df.loc[best_model, 'f1']:.4f}")
print(f"   ROC-AUC      : {results_df.loc[best_model, 'roc_auc']:.4f}")
print(f"\n   MLflow UI    : mlflow ui  →  http://127.0.0.1:5000")
print(f"\n✔  All done! Check reports/figures/ for all plots.")