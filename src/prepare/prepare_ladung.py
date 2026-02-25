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

import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prepare_ladung")

from intent_recognition.src.services.attachment_processing.base_input_processor import AfterCourtAttachmentPreprocessor
from intent_recognition.src.domain.base.blueprints import AfterCourtPreprocessingBlueprint



if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    ladung_params = params["ladung"]

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/prepare/prepare_ladung.py data-file\n")
        sys.exit(1)

    # Initialize preprocessing configuration and processor
    preprocess_params = ladung_params["aftercourt_preprocessing"]
    PREPROCESS_CONFIG = AfterCourtPreprocessingBlueprint.from_dict({
        "clean_text_type": "preprocessed",
        "normalize_whitespace": preprocess_params["normalize_whitespace"],
        "remove_short_lines": preprocess_params["remove_short_lines"],
        "short_line_threshold": preprocess_params["short_line_threshold"],
        "remove_html_tags": preprocess_params["remove_html_tags"],
        "lowercase": preprocess_params["lowercase"]
    })
    aftercourt_attachment_processor = AfterCourtAttachmentPreprocessor("aftercourt_processor_ladung", PREPROCESS_CONFIG)

    # Read raw CSV
    input_path = sys.argv[1]
    df = pd.read_csv(input_path)

    # Apply preprocessing to each row's raw text
    df["cleaned_text"] = df["text"].fillna("").apply(
        aftercourt_attachment_processor._process
    )
    df["text_w_tags"] = df["cleaned_text"].apply(
        aftercourt_attachment_processor._replace_with_tags
    )

    # Train / test split (stratified on is_ladung to preserve class balance)
    train_df, test_df = train_test_split(
        df,
        test_size=params["split"],
        random_state=params["seed"],
        stratify=df[ladung_params["target_col"]]
    )

    # Write outputs
    output_dir = os.path.join("data", "prepared", "ladung")
    output_train = os.path.join(output_dir, "train.csv")
    output_test = os.path.join(output_dir, "test.csv")

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    target_col = ladung_params["target_col"]
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    logger.info(f"Train {target_col} dist: {train_df[target_col].value_counts(normalize=True).to_dict()}")
    logger.info(f"Test {target_col} dist: {test_df[target_col].value_counts(normalize=True).to_dict()}")
    logger.info(f"Train document_type dist: {train_df['document_type'].value_counts().to_dict()}")
    logger.info(f"Test document_type dist: {test_df['document_type'].value_counts().to_dict()}")
    logger.info(f"Saved to {output_train} and {output_test}")