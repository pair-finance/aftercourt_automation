import json
import logging
import sys
import os

# Add the project root so "intent_recognition" is found as a package
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import dill
import yaml
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")


if __name__ == "__main__":
    all_params = yaml.safe_load(open("params.yaml"))
    eval_params = all_params["evaluate"]
    threshold = eval_params["threshold"]

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/eval.py model-path featurized-dir output-dir\n")
        sys.exit(1)

    model_path = sys.argv[1]
    featurized_dir = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        rf_classifier = dill.load(f)

    # Load test data
    test_features_path = os.path.join(featurized_dir, "test_features.npz")
    test_labels_path = os.path.join(featurized_dir, "test_labels.npy")

    logger.info(f"Loading test features from {test_features_path}")
    X_test = sp.load_npz(test_features_path)

    logger.info(f"Loading test labels from {test_labels_path}")
    y_test = np.load(test_labels_path)

    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Using threshold: {threshold}")

    # Predict
    y_pred_probs = rf_classifier.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "threshold": threshold,
        "confusion_matrix": {
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
        },
    }

    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"Confusion Matrix: {cm}")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save prediction probabilities for DVC plots (e.g. ROC curve)
    preds_path = os.path.join(output_dir, "predictions.json")
    preds = [
        {"y_true": int(yt), "y_pred_prob": float(yp)}
        for yt, yp in zip(y_test, y_pred_probs)
    ]
    with open(preds_path, "w") as f:
        json.dump(preds, f, indent=2)
    logger.info(f"Saved predictions to {preds_path}")

    logger.info("Done!")
