"""
Binary Classification: Vermögensverzeichnis Detection
======================================================
Target 0: Document is NOT vermögensverzeichnis
Target 1: Document IS vermögensverzeichnis

Uses TF-IDF features with a Logistic Regression baseline and optional
Random Forest / Linear SVC comparison.
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import dill
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Make the intent_recognition submodule importable so dill can resolve the
# ClassificationSpacyLemmaTokenizer class referenced inside the pickled vectorizer.
INTENT_RECOGNITION_PATH = Path(__file__).resolve().parents[3] / "intent_recognition"
if str(INTENT_RECOGNITION_PATH) not in sys.path:
    sys.path.insert(0, str(INTENT_RECOGNITION_PATH))

from src.services.models.aftercourt_tokenizer import ClassificationSpacyLemmaTokenizer  # noqa: F401, E402

# ── 1. Load & prepare data ──────────────────────────────────────────────────

DATA_PATH = Path("/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/data/raw/final_raw_data.csv")
print(f"Loading data from {DATA_PATH}")

df = pd.read_csv(DATA_PATH, usecols=["text", "cleaned_text", "document_type"])

# Use cleaned_text where available, fall back to text
df["input_text"] = df["cleaned_text"].fillna(df["text"])
df = df.dropna(subset=["input_text"]).reset_index(drop=True)

# Binary target
df["label"] = (df["document_type"] == "vermögensverzeichnis").astype(int)

print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts().rename({0: 'NOT vermögensverzeichnis', 1: 'vermögensverzeichnis'})}\n")

# ── 2. Train / test split (stratified) ──────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    df["input_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
print(f"Train label distribution:\n{y_train.value_counts()}")
print(f"Test  label distribution:\n{y_test.value_counts()}\n")

# ── 3. TF-IDF Vectorizer (pre-trained, loaded from dill) ───────────────────

VEC_PATH = Path(
    "/Users/melih.gorgulu/Desktop/Projects/intent_recognition/notebooks/after-court/models/vectorizers/05-12-2026_tf_idf_vectorizer_v1.dill"
)
print(f"Loading pre-trained TF-IDF vectorizer from {VEC_PATH}")
with open(VEC_PATH, "rb") as f:
    tfidf = dill.load(f)
print(f"Vectorizer: {type(tfidf).__name__}  |  vocab size: {len(getattr(tfidf, 'vocabulary_', {}))}")

# ── 4. Define models ────────────────────────────────────────────────────────

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

# ── 5. Cross-validation + evaluation ────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, clf in models.items():
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])

    # Cross-val on train set
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    print(f"[{name}] CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit on full train, evaluate on test
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n{'='*60}")
    print(f"  {name} — Test Set Results")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["NOT vermögensverz.", "vermögensverz."]))

    # ROC-AUC (where probability is available)
    if hasattr(clf, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"  ROC-AUC: {auc:.4f}\n")
    elif hasattr(clf, "decision_function"):
        y_scores = pipe.decision_function(X_test)
        auc = roc_auc_score(y_test, y_scores)
        print(f"  ROC-AUC: {auc:.4f}\n")
    else:
        auc = None

    results[name] = {"pipe": pipe, "y_pred": y_pred, "auc": auc}

# ── 6. Feature importance — top TF-IDF keywords (Random Forest) ────────────

best_model_name = "RandomForest"
best_pipe = results[best_model_name]["pipe"]
feature_names = best_pipe.named_steps["tfidf"].get_feature_names_out()
importances = best_pipe.named_steps["clf"].feature_importances_

top_k = 30
top_idx = np.argsort(importances)[-top_k:][::-1]

print(f"\n{'='*60}")
print(f"  Top {top_k} most important features (Random Forest)")
print(f"{'='*60}")
for i in top_idx:
    print(f"  {feature_names[i]:40s}  importance={importances[i]:.4f}")

# ── 6b. Save the trained Random Forest classifier ──────────────────────────

MODEL_DIR = Path("/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/models/classification")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RF_MODEL_PATH = MODEL_DIR / "15-05-2026_binary_va_rf_classifier_v1.dill"

rf_clf = results["RandomForest"]["pipe"].named_steps["clf"]
with open(RF_MODEL_PATH, "wb") as f:
    dill.dump(rf_clf, f)
print(f"\nRandom Forest classifier saved to {RF_MODEL_PATH}")

# ── 7. Plots ────────────────────────────────────────────────────────────────

PLOT_DIR = Path(__file__).resolve().parents[2] / "assets" / "data_analysis_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 7a. Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    results[best_model_name]["y_pred"],
    display_labels=["NOT vermögensverz.", "vermögensverz."],
    cmap="Blues",
    ax=ax,
)
ax.set_title(f"{best_model_name} — Confusion Matrix")
fig.tight_layout()
fig.savefig(PLOT_DIR / "vermogensverzeichnis_confusion_matrix.png", dpi=150)
print(f"\nConfusion matrix saved to {PLOT_DIR / 'vermogensverzeichnis_confusion_matrix.png'}")

# 7b. ROC curve (Logistic Regression)
if results[best_model_name]["auc"] is not None:
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, label=f"AUC = {results[best_model_name]['auc']:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"{best_model_name} — ROC Curve")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "vermogensverzeichnis_roc_curve.png", dpi=150)
    print(f"ROC curve saved to {PLOT_DIR / 'vermogensverzeichnis_roc_curve.png'}")

# 7c. Top features bar chart (Random Forest importances)
fig3, ax3 = plt.subplots(figsize=(10, 8))
top_names = [feature_names[i] for i in top_idx[:15]]
top_imps = [importances[i] for i in top_idx[:15]]
ax3.barh(top_names[::-1], top_imps[::-1], color="steelblue")
ax3.set_title("Top 15 most important features (Random Forest)")
ax3.set_xlabel("Feature importance")
fig3.tight_layout()
fig3.savefig(PLOT_DIR / "vermogensverzeichnis_top_features.png", dpi=150)
print(f"Top features plot saved to {PLOT_DIR / 'vermogensverzeichnis_top_features.png'}")

plt.close("all")
print("\nDone.")
