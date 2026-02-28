import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────
DATA_PATH   = "data/Diabetes_Dataset.csv"
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("=" * 55)
print("  DIABETES DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 55)
print(f"\n Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\n Columns:")
print(df.columns.tolist())

# ── 2. Basic Info ────────────────────────────────────────
print("\n\n── Data Types & Missing Values ──────────────────────")
info = pd.DataFrame({
    "dtype"   : df.dtypes,
    "non_null": df.notnull().sum(),
    "missing" : df.isnull().sum(),
    "missing%": (df.isnull().mean() * 100).round(2),
})
print(info.to_string())

# ── 3. Target Distribution ───────────────────────────────
print("\n\n── Target: Outcome ───────────────────────────────────")
print(df["Outcome"].value_counts())
print(df["Outcome"].value_counts(normalize=True).mul(100).round(2).astype(str) + " %")

fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#4C9BE8", "#E8694C"]
df["Outcome"].value_counts().plot(kind="bar", color=colors, ax=ax, edgecolor="white")
ax.set_title("Target Distribution — Diabetic vs Non-diabetic", fontsize=13, fontweight="bold")
ax.set_xlabel("Outcome")
ax.set_ylabel("Count")
ax.set_xticklabels(["Non-diabetic (0)", "Diabetic (1)"], rotation=0)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/01_target_distribution.png", dpi=150)
plt.close()
print(f"  ✔ Saved 01_target_distribution.png")

# ── 4. Numeric Feature Distributions ────────────────────
num_cols = df.select_dtypes(include=np.number).columns.tolist()
if "Outcome" in num_cols:
    num_cols.remove("Outcome")

print(f"\n\n── Numeric Features ({len(num_cols)}) ──────────────────────────")
print(df[num_cols].describe().round(2).to_string())

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[df["Outcome"] == "Non-diabetic"][col], bins=30, alpha=0.6,
                 color="#4C9BE8", label="Non-diabetic", edgecolor="white")
    axes[i].hist(df[df["Outcome"] == "Diabetic"][col], bins=30, alpha=0.6,
                 color="#E8694C", label="Diabetic", edgecolor="white")
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].legend(fontsize=8)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Numeric Feature Distributions by Outcome", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/02_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔ Saved 02_numeric_distributions.png")

# ── 5. Categorical Features ──────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
if "Outcome" in cat_cols:
    cat_cols.remove("Outcome")

print(f"\n\n── Categorical Features ({len(cat_cols)}) ────────────────────────")
for col in cat_cols:
    print(f"\n  {col}: {df[col].value_counts().to_dict()}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df["Outcome"], normalize="index") * 100
    ct.plot(kind="bar", ax=axes[i], color=["#4C9BE8", "#E8694C"],
            edgecolor="white", rot=30)
    axes[i].set_title(f"{col} vs Outcome (%)", fontsize=11, fontweight="bold")
    axes[i].set_ylabel("% within group")
    axes[i].legend(title="Outcome", fontsize=8)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Categorical Features vs Outcome", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/03_categorical_vs_outcome.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔ Saved 03_categorical_vs_outcome.png")

# ── 6. Correlation Heatmap ───────────────────────────────
# encode outcome for correlation
df_corr = df.copy()
df_corr["Outcome_bin"] = (df_corr["Outcome"] == "Diabetic").astype(int)
corr = df_corr[num_cols + ["Outcome_bin"]].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
ax.set_title("Correlation Heatmap (Numeric Features + Target)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/04_correlation_heatmap.png", dpi=150)
plt.close()
print(f"  ✔ Saved 04_correlation_heatmap.png")

# ── 7. Boxplots ──────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(data=df, x="Outcome", y=col, ax=axes[i],
                palette={"Non-diabetic": "#4C9BE8", "Diabetic": "#E8694C"})
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Boxplots: Numeric Features by Outcome", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/05_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔ Saved 05_boxplots.png")

# ── 8. Age Distribution ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for outcome, color in [("Diabetic", "#E8694C"), ("Non-diabetic", "#4C9BE8")]:
    sns.kdeplot(df[df["Outcome"] == outcome]["Age"], label=outcome,
                fill=True, alpha=0.4, color=color, ax=ax)
ax.set_title("Age Distribution by Diabetic Status", fontsize=13, fontweight="bold")
ax.set_xlabel("Age")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/06_age_distribution.png", dpi=150)
plt.close()
print(f"  ✔ Saved 06_age_distribution.png")

# ── 9. BMI vs Glucose Scatter ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for outcome, color, marker in [("Diabetic", "#E8694C", "o"), ("Non-diabetic", "#4C9BE8", "s")]:
    sub = df[df["Outcome"] == outcome]
    ax.scatter(sub["BMI"], sub["Glucose"], c=color, label=outcome,
               alpha=0.3, s=15, marker=marker)
ax.set_xlabel("BMI")
ax.set_ylabel("Glucose")
ax.set_title("BMI vs Glucose by Diabetic Status", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/07_bmi_vs_glucose.png", dpi=150)
plt.close()
print(f"  ✔ Saved 07_bmi_vs_glucose.png")

# ── Summary ──────────────────────────────────────────────
summary = f"""
DIABETES DATASET — EDA SUMMARY
================================
Rows           : {df.shape[0]:,}
Columns        : {df.shape[1]}
Missing Values : {df.isnull().sum().sum()}
Numeric Cols   : {len(num_cols)}
Categorical    : {len(cat_cols)}

Target (Outcome):
  Diabetic     : {(df['Outcome']=='Diabetic').sum():,} ({(df['Outcome']=='Diabetic').mean()*100:.1f}%)
  Non-diabetic : {(df['Outcome']=='Non-diabetic').sum():,} ({(df['Outcome']=='Non-diabetic').mean()*100:.1f}%)
  Imbalance Ratio: {(df['Outcome']=='Non-diabetic').sum() / (df['Outcome']=='Diabetic').sum():.2f}:1

Top Correlated Features with Outcome:
{corr['Outcome_bin'].drop('Outcome_bin').abs().sort_values(ascending=False).round(3).to_string()}
"""
with open("reports/eda_summary.txt", "w") as f:
    f.write(summary)

print("\n" + summary)
print("\n  EDA complete! All figures saved to reports/figures/")