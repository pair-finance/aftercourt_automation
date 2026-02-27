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
from sklearn.ensemble import RandomForestClassifier
import mlflow

from src.mlflow_utils import init_mlflow, get_or_create_run, log_params_flat

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")


if __name__ == "__main__":
    all_params = yaml.safe_load(open("params.yaml"))
    prepare_params = all_params["prepare"]
    train_params = all_params["train"]
    target = prepare_params["target"]
    clf_params = train_params["classifier"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/training/train.py featurized-dir output-model-path\n")
        sys.exit(1)

    featurized_dir = sys.argv[1]
    output_model_path = sys.argv[2]

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    train_features_path = os.path.join(featurized_dir, "train_features.npz")
    train_labels_path = os.path.join(featurized_dir, "train_labels.npy")

    logger.info(f"Loading training features from {train_features_path}")
    X_train = sp.load_npz(train_features_path)

    logger.info(f"Loading training labels from {train_labels_path}")
    y_train = np.load(train_labels_path)

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Labels distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ── MLflow: initialise & open run ──────────────────────────────
    init_mlflow()
    with get_or_create_run("train", run_name=f"train-{target}"):
        # Log all pipeline parameters (only the first stage to open the run
        # should log the full param set; subsequent stages add their own).
        log_params_flat(prepare_params, prefix="prepare")
        log_params_flat(all_params.get("fit_vectorizer", {}), prefix="fit_vectorizer")
        log_params_flat(train_params, prefix="train")
        log_params_flat(all_params.get("evaluate", {}), prefix="evaluate")

        mlflow.log_metric("train_samples", X_train.shape[0])
        mlflow.log_metric("train_features", X_train.shape[1])

        label_counts = dict(zip(*np.unique(y_train, return_counts=True)))
        for label, count in label_counts.items():
            mlflow.log_metric(f"train_label_{int(label)}_count", int(count))

        # ── Train ──────────────────────────────────────────────────
        logger.info("Training RandomForestClassifier")
        rf_classifier = RandomForestClassifier(
            n_estimators=clf_params["n_estimators"],
            random_state=clf_params["random_state"],
        )
        rf_classifier.fit(X_train, y_train)

        # Save model via dill (DVC artifact)
        logger.info(f"Saving model to {output_model_path}")
        with open(output_model_path, "wb") as f:
            dill.dump(rf_classifier, f)

        # Log the serialised model file to MLflow as an artifact
        mlflow.log_artifact(output_model_path, artifact_path="model")

    logger.info("Done!")
