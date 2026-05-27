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
    average_precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)
import matplotlib
matplotlib.use("Agg")  # headless backend for CI/servers
import matplotlib.pyplot as plt
import mlflow

from src.mlflow_utils import init_mlflow, get_or_create_run, finish_pipeline_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("STAGE: EVALUATE")


if __name__ == "__main__":
    all_params = yaml.safe_load(open("params.yaml"))
    eval_params = all_params["evaluate"]

    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/eval/eval.py model-path featurized-dir output-dir target\n")
        sys.exit(1)

    model_path = sys.argv[1]
    featurized_dir = sys.argv[2]
    output_dir = sys.argv[3]
    target = sys.argv[4]          # e.g. "ladung" or "pfub" — passed by DVC foreach
    threshold = eval_params[target]["threshold"]

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
    pr_auc = average_precision_score(y_test, y_pred_probs)
    cm = confusion_matrix(y_test, y_pred).tolist()
    cls_report_str = classification_report(y_test, y_pred, zero_division=0, digits=4)
    cls_report_dict = classification_report(
        y_test, y_pred, zero_division=0, output_dict=True
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "confusion_matrix": {
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
        },
        "classification_report": cls_report_dict,
    }

    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"PR-AUC:    {pr_auc:.4f}")
    logger.info(f"Confusion Matrix: {cm}")
    logger.info(f"Classification report:\n{cls_report_str}")

    # Save metrics JSON (kept for DVC compatibility)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save human-readable classification report alongside JSON metrics
    cls_report_path = os.path.join(output_dir, "classification_report.txt")
    with open(cls_report_path, "w") as f:
        f.write(cls_report_str)
    logger.info(f"Saved classification report to {cls_report_path}")

    # Save prediction probabilities for DVC plots (e.g. ROC curve)
    preds_path = os.path.join(output_dir, "predictions.json")
    preds = [
        {"y_true": int(yt), "y_pred_prob": float(yp)}
        for yt, yp in zip(y_test, y_pred_probs)
    ]
    with open(preds_path, "w") as f:
        json.dump(preds, f, indent=2)
    logger.info(f"Saved predictions to {preds_path}")

    # ── MLflow: log evaluation metrics, artifacts & plots ──────────
    init_mlflow()
    with get_or_create_run("evaluate", target=target, run_name=f"{target}-{threshold}"):
        # Scalar metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "threshold": threshold,
            "test_samples": int(X_test.shape[0]),
            "confusion_tn": cm[0][0],
            "confusion_fp": cm[0][1],
            "confusion_fn": cm[1][0],
            "confusion_tp": cm[1][1],
        })

        # Log the JSON artefacts
        mlflow.log_artifact(metrics_path, artifact_path="evaluation")
        mlflow.log_artifact(preds_path, artifact_path="evaluation")
        mlflow.log_artifact(cls_report_path, artifact_path="evaluation")

        # ── Generate & log plots ──────────────────────────────────
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # ROC curve
        roc_display = RocCurveDisplay.from_predictions(y_test, y_pred_probs)
        roc_display.ax_.set_title("ROC Curve")
        roc_fig_path = os.path.join(plots_dir, "roc_curve.png")
        roc_display.figure_.savefig(roc_fig_path, dpi=150, bbox_inches="tight")
        plt.close(roc_display.figure_)
        mlflow.log_artifact(roc_fig_path, artifact_path="evaluation/plots")

        # Confusion matrix
        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        cm_display.ax_.set_title("Confusion Matrix")
        cm_fig_path = os.path.join(plots_dir, "confusion_matrix.png")
        cm_display.figure_.savefig(cm_fig_path, dpi=150, bbox_inches="tight")
        plt.close(cm_display.figure_)
        mlflow.log_artifact(cm_fig_path, artifact_path="evaluation/plots")

        # Precision-Recall curve
        pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred_probs)
        pr_display.ax_.set_title("Precision-Recall Curve")
        pr_fig_path = os.path.join(plots_dir, "precision_recall_curve.png")
        pr_display.figure_.savefig(pr_fig_path, dpi=150, bbox_inches="tight")
        plt.close(pr_display.figure_)
        mlflow.log_artifact(pr_fig_path, artifact_path="evaluation/plots")

        # Predicted probability distribution (overlaid by true class)
        probs_pos = y_pred_probs[y_test == 1]
        probs_neg = y_pred_probs[y_test == 0]
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0.0, 1.0, 51)
        ax.hist(probs_neg, bins=bins, alpha=0.6, label=f"y_true=0 (n={len(probs_neg)})", color="tab:blue")
        ax.hist(probs_pos, bins=bins, alpha=0.6, label=f"y_true=1 (n={len(probs_pos)})", color="tab:orange")
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1, label=f"threshold={threshold}")
        ax.set_xlabel("Predicted probability (positive class)")
        ax.set_ylabel("Count")
        ax.set_title(f"Predicted Probability Distribution — {target}")
        ax.legend()
        prob_dist_path = os.path.join(plots_dir, "probability_distribution.png")
        fig.savefig(prob_dist_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(prob_dist_path, artifact_path="evaluation/plots")

        # Same plot, log-scaled y-axis (helps when one class dominates)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(probs_neg, bins=bins, alpha=0.6, label=f"y_true=0 (n={len(probs_neg)})", color="tab:blue")
        ax.hist(probs_pos, bins=bins, alpha=0.6, label=f"y_true=1 (n={len(probs_pos)})", color="tab:orange")
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1, label=f"threshold={threshold}")
        ax.set_yscale("log")
        ax.set_xlabel("Predicted probability (positive class)")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(f"Predicted Probability Distribution (log) — {target}")
        ax.legend()
        prob_dist_log_path = os.path.join(plots_dir, "probability_distribution_log.png")
        fig.savefig(prob_dist_log_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(prob_dist_log_path, artifact_path="evaluation/plots")

        logger.info("Logged metrics, artifacts & plots to MLflow")

    # The evaluate stage is the last stage — clean up the active run marker
    # so the next `dvc repro` creates a fresh MLflow run.
    finish_pipeline_run(target)

    logger.info("Done!")
