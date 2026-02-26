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
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from intent_recognition.src.services.models.aftercourt_tokenizer import ClassificationSpacyLemmaTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fit_vectorizer_ladung")


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["fit_vectorizer"]
    seed = params["seed"]
    vec_params = params["vectorizer"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython src/featurazition/fit_vectorizer.py train-csv output-path\n")
        sys.exit(1)

    train_input = sys.argv[1]
    output_path = sys.argv[2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Reading training data from {train_input}")
    df_train = pd.read_csv(train_input)

    logger.info("Initializing TF-IDF vectorizer")
    vectorizer = TfidfVectorizer(
        tokenizer=ClassificationSpacyLemmaTokenizer(),
        max_features=vec_params["max_features"],
        ngram_range=(vec_params["ngram_range_min"], vec_params["ngram_range_max"]),
        norm=vec_params["norm"],
        min_df=vec_params["min_df"],
        max_df=vec_params["max_df"],
        stop_words=None,
        lowercase=vec_params["lowercase"]
    )

    logger.info(f"Fitting vectorizer on {len(df_train)} training samples")
    vectorizer.fit(df_train["text_w_tags"])

    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    logger.info(f"Saving vectorizer to {output_path}")
    with open(output_path, "wb") as f:
        dill.dump(vectorizer, f)

    logger.info("Done!")
