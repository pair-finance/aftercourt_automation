import sys
import os
import json
import glob
from typing import Any, Dict, List, Optional, Union

# Add the project root (parent of utils/) so "intent_recognition" is found as a package
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add intent_recognition/ so internal "from src.*" imports resolve correctly
_intent_recog_root = os.path.join(_project_root, "intent_recognition")
if _intent_recog_root not in sys.path:
    sys.path.insert(0, _intent_recog_root)
    

from intent_recognition.src.services.attachment_processing.base_input_processor import AfterCourtAttachmentPreprocessor
from intent_recognition.src.domain.base.blueprints import AfterCourtPreprocessingBlueprint
from intent_recognition.src.services.models.aftercourt_classification_model import ClassificationSpacyLemmaTokenizer

"""
Utility module for intent recognition service, containing shared components like preprocessors and tokenizers.

Model configurations are loaded from
``intent_recognition/configs/models/after_court/*.json``. Use
:func:`list_available_models` to see the available model keys, and pass a
``model`` argument to :func:`apply_text_cleaning` / :func:`get_attachment_processor`
to choose between them (e.g. ``"vermogenverzeichnis"``, ``"ladung"``, ``"pfub"``).
"""

# ---------------------------------------------------------------------------
# Model configuration discovery / loading
# ---------------------------------------------------------------------------

AFTERCOURT_MODEL_CONFIGS_DIR = os.path.join(
    _intent_recog_root, "configs", "models", "after_court"
)

# Short aliases -> config file stem (without .json). Keeps notebook usage terse
# while still allowing callers to pass the full config file name.
_MODEL_ALIASES: Dict[str, str] = {
    "vermogenverzeichnis": "aftercourt_classification_vermogenverzeichnis",
    "vermoegensverzeichnis": "aftercourt_classification_vermogenverzeichnis",
    "ladung": "aftercourt_classification_ladung",
    "pfub": "aftercourt_classification_pfub",
}


def list_available_models() -> List[str]:
    """Return the list of available aftercourt model config stems."""
    paths = sorted(glob.glob(os.path.join(AFTERCOURT_MODEL_CONFIGS_DIR, "*.json")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def _resolve_model_config_path(model: str) -> str:
    """Resolve a model name/alias/path to an absolute JSON config path."""
    if os.path.isabs(model) and os.path.isfile(model):
        return model
    name = _MODEL_ALIASES.get(model, model)
    if not name.endswith(".json"):
        name = f"{name}.json"
    path = os.path.join(AFTERCOURT_MODEL_CONFIGS_DIR, name)
    if not os.path.isfile(path):
        available = list_available_models()
        raise FileNotFoundError(
            f"Unknown aftercourt model config '{model}'. "
            f"Available models: {available} (aliases: {list(_MODEL_ALIASES)})"
        )
    return path


_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


def load_model_config(model: str) -> Dict[str, Any]:
    """Load a model configuration JSON by name, alias, or absolute path."""
    path = _resolve_model_config_path(model)
    cached = _CONFIG_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _CONFIG_CACHE[path] = cfg
    return cfg


def get_preprocess_blueprint(
    model_or_config: Union[str, Dict[str, Any], AfterCourtPreprocessingBlueprint],
) -> AfterCourtPreprocessingBlueprint:
    """Build an :class:`AfterCourtPreprocessingBlueprint` from a model name or dict."""
    if isinstance(model_or_config, AfterCourtPreprocessingBlueprint):
        return model_or_config
    if isinstance(model_or_config, str):
        cfg = load_model_config(model_or_config)
        preproc = cfg.get("preprocessing", {})
    else:
        # Allow passing either a full model config dict or just the preprocessing dict.
        preproc = model_or_config.get("preprocessing", model_or_config)
    return AfterCourtPreprocessingBlueprint.from_dict(preproc)


# ---------------------------------------------------------------------------
# Default preprocessor (kept for backwards compatibility)
# ---------------------------------------------------------------------------

# Initialize preprocessing configuration and processor
PREPROCESS_CONFIG = AfterCourtPreprocessingBlueprint.from_dict({
    "clean_text_type": ["preprocessed", 'original'],
    "normalize_whitespace": True,
    "remove_short_lines": True,
    "short_line_threshold": 3,
    "remove_html_tags": True,
    "lowercase": True
})
aftercourt_attachment_processor = AfterCourtAttachmentPreprocessor("aftercourt_processor", PREPROCESS_CONFIG)


# Cache of (model name) -> AfterCourtAttachmentPreprocessor so repeated calls
# in notebooks don't keep rebuilding the same processor.
_PROCESSOR_CACHE: Dict[str, AfterCourtAttachmentPreprocessor] = {}


def get_attachment_processor(model: Optional[str] = None) -> AfterCourtAttachmentPreprocessor:
    """Return the attachment preprocessor for a given model config.

    If ``model`` is ``None``, returns the default module-level processor.
    """
    if model is None:
        return aftercourt_attachment_processor
    path = _resolve_model_config_path(model)
    cached = _PROCESSOR_CACHE.get(path)
    if cached is not None:
        return cached
    cfg = load_model_config(path)
    blueprint = get_preprocess_blueprint(cfg)
    name = cfg.get("name", os.path.splitext(os.path.basename(path))[0])
    processor = AfterCourtAttachmentPreprocessor(name, blueprint)
    _PROCESSOR_CACHE[path] = processor
    return processor

# Initialize custom tokenizer
class CustomTokenizer(ClassificationSpacyLemmaTokenizer):
    # def __call__(self, text):
    #     doc = self.nlp(text)
    #     tokens_cleared = [
    #         token.lemma_.lower()
    #         for token in doc
    #         if (
    #             (not token.is_stop and not token.is_punct and not token.like_num or token == "m")
    #         ) and (
    #             token.lemma_ != "\n" and token.lemma_ != "\n\n" and 
    #             token.lemma_ != " " and token.lemma_ != "" and 
    #             len(token.lemma_) < 45
    #         ) and
    #         (not token.lemma_.startswith("<") and not token.lemma_.endswith(">"))  # Exclude special tokens
    #     ]
    #     return tokens_cleared
    def __call__(self, text):
        return super().__call__(text)

tokenizer = CustomTokenizer()


def apply_text_cleaning(
    text: str,
    config: Optional[AfterCourtPreprocessingBlueprint] = None,
    model: Optional[str] = None,
) -> str:
    """Apply text cleaning to the extracted text.

    Parameters
    ----------
    text:
        Raw text to clean.
    config:
        Optional explicit :class:`AfterCourtPreprocessingBlueprint`. Takes
        precedence over ``model`` when both are provided.
    model:
        Name/alias of an aftercourt model config under
        ``intent_recognition/configs/models/after_court/`` (e.g.
        ``"vermogenverzeichnis"``, ``"ladung"``, ``"pfub"``). When provided,
        the model's preprocessing settings are used.
    """
    if config is not None:
        processor = AfterCourtAttachmentPreprocessor("custom_processor", config)
        return processor._process(text)
    return get_attachment_processor(model)._process(text)


def apply_replace_with_tags(text: str, model: Optional[str] = None) -> str:
    """Apply replace with tags to the extracted text."""
    return get_attachment_processor(model)._replace_with_tags(text)


def apply_tokenization(text: str) -> List[str]:
    """Apply custom tokenization to the text."""
    return tokenizer(text)