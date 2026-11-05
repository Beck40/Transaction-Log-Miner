"""Sentence-transformer embeddings with a small sklearn fallback."""
from typing import Any, cast


def encode(texts, model_name="all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name).encode(list(texts), show_progress_bar=False)
    except ImportError:
        from sklearn.feature_extraction.text import TfidfVectorizer
        matrix = TfidfVectorizer(max_features=512).fit_transform(texts)
        return cast(Any, matrix).toarray()
