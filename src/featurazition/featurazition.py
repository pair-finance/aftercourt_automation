import logging
import sys
import os

# Add the project root so "intent_recognition" is found as a package
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add intent_recognition/ to sys.path so its internal "from src." imports resolve correctly
_ir_root = os.path.join(_project_root, "intent_recognition")
if _ir_root not in sys.path:
    sys.path.insert(0, _ir_root)

import dill
import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("featurize_ladung")


def featurize_and_save(input_path: str, output_features_path: str, output_labels_path: str, vectorizer, target_col: str):
    logger.info(f"Reading data from {input_path}")
    df = pd.read_csv(input_path)

    X = vectorizer.transform(df["text_w_tags"])
    y = df[target_col].values

    sp.save_npz(output_features_path, X)
    np.save(output_labels_path, y)

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    logger.info(f"Saved features to {output_features_path}")
    logger.info(f"Saved labels to {output_labels_path}")


if __name__ == "__main__":
    all_params = yaml.safe_load(open("params.yaml"))
    prepare_params = all_params["prepare"]
    target = prepare_params["target"]
    target_col = prepare_params[target]["target_col"]

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/featurazition/featurazition.py input-dir vectorizer-path output-dir\n")
        sys.exit(1)

    in_dir = sys.argv[1]
    vectorizer_path = sys.argv[2]
    out_dir = sys.argv[3]

    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading vectorizer from {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = dill.load(f)
    logger.info("Vectorizer loaded")

    featurize_and_save(
        input_path=os.path.join(in_dir, "train.csv"),
        output_features_path=os.path.join(out_dir, "train_features.npz"),
        output_labels_path=os.path.join(out_dir, "train_labels.npy"),
        vectorizer=vectorizer,
        target_col=target_col
    )

    featurize_and_save(
        input_path=os.path.join(in_dir, "test.csv"),
        output_features_path=os.path.join(out_dir, "test_features.npz"),
        output_labels_path=os.path.join(out_dir, "test_labels.npy"),
        vectorizer=vectorizer,
        target_col=target_col
    )
