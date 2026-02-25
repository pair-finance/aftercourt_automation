import logging
import yaml
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

import pandas as pd

from intent_recognition.src.services.models.aftercourt_tokenizer import ClassificationSpacyLemmaTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tokenize_ladung")


def generate_and_save_train_tokenized(train_input: str, train_output: str, tokenizer: ClassificationSpacyLemmaTokenizer):
    logger.info(f"Reading train data from {train_input}")
    df_train = pd.read_csv(train_input)
    logger.info(f"Tokenizing {len(df_train)} train rows")
    df_train["tokenized_text"] = df_train["text_w_tags"].apply(tokenizer)
    df_train.to_csv(train_output, index=False)
    logger.info(f"Train features saved to {train_output}")


def generate_and_save_test_tokenized(test_input: str, test_output: str, tokenizer: ClassificationSpacyLemmaTokenizer):
    logger.info(f"Reading test data from {test_input}")
    df_test = pd.read_csv(test_input)
    logger.info(f"Tokenizing {len(df_test)} test rows")
    df_test["tokenized_text"] = df_test["text_w_tags"].apply(tokenizer)
    df_test.to_csv(test_output, index=False)
    logger.info(f"Test features saved to {test_output}")

if __name__ == "__main__":


    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/tokenization/tokenization_ladung.py data-file output-file\n")
        sys.exit(1)
        
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    
    train_input = os.path.join(in_path, "train.csv")
    test_input = os.path.join(in_path, "test.csv")
    train_output = os.path.join(out_path, "train.csv")
    test_output = os.path.join(out_path, "test.csv")
    
    os.makedirs(out_path, exist_ok=True)

    logger.info("Loading tokenizer")
    tokenizer = ClassificationSpacyLemmaTokenizer()
    logger.info("Tokenizer loaded")

    logger.info("Tokenizing train and test data...")
    generate_and_save_train_tokenized(
        train_input=train_input,
        train_output=train_output,
        tokenizer=tokenizer
    )

    generate_and_save_test_tokenized(
        test_input=test_input,
        test_output=test_output,
        tokenizer=tokenizer
    )
    logger.info("Tokenization complete")
