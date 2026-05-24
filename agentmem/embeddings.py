"""
embeddings.py — Local embedding model wrapper.
Uses sentence-transformers so everything runs offline, no API key required.
Falls back to a lightweight model if the primary one isn't cached yet.
"""

import numpy as np
import struct
from typing import Union


_model_cache: dict = {}

DEFAULT_MODEL = "all-MiniLM-L6-v2"   # 80MB, fast, solid quality


def _get_model(model_name: str):
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings.\n"
                "Install it with:  pip install sentence-transformers"
            )
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed(text: Union[str, list[str]], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Embed a string or list of strings into float32 vectors.
    Returns shape (dim,) for a single string, (N, dim) for a list.
    """
    model = _get_model(model_name)
    vectors = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype(np.float32)


# ------------------------------------------------------------------
# Serialisation helpers (store as raw bytes in SQLite BLOB column)
# ------------------------------------------------------------------

def to_bytes(vector: np.ndarray) -> bytes:
    """Serialize a 1-D float32 numpy array to raw bytes."""
    return vector.astype(np.float32).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    """Deserialize raw bytes back to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised vectors.
    Since embed() normalises by default, this is just a dot product.
    """
    return float(np.dot(a, b))
