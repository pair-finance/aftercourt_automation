import sys
import os
from typing import List

# Add the project root (parent of utils/) so "intent_recognition" is found as a package
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    

from intent_recognition.src.services.attachment_processing.base_input_processor import AfterCourtAttachmentPreprocessor
from intent_recognition.src.domain.base.blueprints import AfterCourtPreprocessingBlueprint
from intent_recognition.src.services.models.aftercourt_classification_model import ClassificationSpacyLemmaTokenizer

"""
Utility module for intent recognition service, containing shared components like preprocessors and tokenizers.
"""

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


def apply_text_cleaning(text: str) -> str:
    """Apply text cleaning to the extracted text."""
    return aftercourt_attachment_processor._process(text)

def apply_replace_with_tags(text: str) -> str:
    """Apply replace with tags to the extracted text."""
    return aftercourt_attachment_processor._replace_with_tags(text)

def apply_tokenization(text: str) -> List[str]:
    """Apply custom tokenization to the text."""
    return tokenizer(text)