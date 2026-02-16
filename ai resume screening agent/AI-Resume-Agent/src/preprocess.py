from typing import Optional
from pathlib import Path
import re
import logging

import spacy

logger = logging.getLogger(__name__)


def load_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except Exception:
        try:
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as e:
            logger.exception("Failed to load spaCy model: %s", e)
            return None


def preprocess_text(text: str, nlp: Optional[spacy.language.Language] = None) -> str:
    """Preprocess text: lower, remove special chars, stopwords, and lemmatize.

    Args:
        text: Raw text input.
        nlp: Optional spaCy language model instance.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if nlp is None:
        nlp = load_spacy_model()
    if not nlp:
        return text

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.lemma_.strip()]
    return " ".join(tokens)
