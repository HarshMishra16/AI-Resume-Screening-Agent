from typing import List
from pathlib import Path
import logging
import numpy as np
import joblib

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_path: Path = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_path = cache_path

    def embed(self, texts: List[str]):
        """Compute embeddings for a list of texts."""
        if not texts:
            return np.array([])
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        except Exception:
            logger.exception("Failed to compute embeddings")
            return np.array([])

    def save_embeddings(self, key: str, embeddings: np.ndarray):
        if self.cache_path is None:
            return
        self.cache_path.mkdir(parents=True, exist_ok=True)
        path = self.cache_path / f"{key}.joblib"
        joblib.dump(embeddings, path)

    def load_embeddings(self, key: str):
        if self.cache_path is None:
            return None
        path = self.cache_path / f"{key}.joblib"
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                logger.exception("Failed to load embeddings from cache: %s", path)
        return None
