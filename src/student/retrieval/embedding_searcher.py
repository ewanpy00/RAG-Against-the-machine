from pathlib import Path
from typing import List, cast, Any
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from student.models import Chunk


class EmbeddingSearcher:
    """Search via cosine similarity over dense embeddings."""
    def __init__(
        self,
        embeddings_path: Path = Path("data/processed/embeddings.npy"),
        chunks_path: Path = Path("data/processed/chunks.json"),
        meta_path: Path = Path("data/processed/embeddings_meta.json"),
        use_cache: bool = False,
        cache_dir: Path = Path("data/cache")
    ):
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {embeddings_path}. "
                f"Run 'index --build_embeddings True' first."
            )
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks not found: {chunks_path}. "
                f"Run 'index' command first to create chunks."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {meta_path}. "
                f"Expected metadata with model info."
            )

        meta = json.loads(meta_path.read_text())
        model_name = meta["model_name"]

        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device="cpu")

        print(f"Loading embeddings: {embeddings_path}")
        self.embeddings = np.load(embeddings_path)

        with chunks_path.open() as f:
            chunks_data = json.load(f)
        self.chunks = [Chunk(**data) for data in chunks_data]

        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.chunks)} chunks vs "
                f"{self.embeddings.shape[0]} embeddings. "
                f"Re-run 'make index EMBEDDING=True' to rebuild both indexes."
            )

        self.use_cache = use_cache
        if use_cache:
            model_name_safe = meta["model_name"].replace("/", "_")
            self.cache_dir = cache_dir / model_name_safe
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Query cache enabled: {self.cache_dir}")
        else:
            self.cache_dir = None

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        return query.strip().lower()

    def _query_cache_path(self, query: str) -> Any:
        """Get cache file path for query."""
        import hashlib
        assert self.cache_dir is not None
        normalized = self._normalize_query(query)
        query_hash = hashlib.md5(normalized.encode()).hexdigest()
        return self.cache_dir / f"{query_hash}.npy"

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query with optional disk caching."""

        if self.use_cache:
            cache_path = self._query_cache_path(query)

            if cache_path.exists():
                return cast(np.ndarray, np.load(cache_path))

        embedding = cast(np.ndarray, self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0])

        if self.use_cache:
            cache_path = self._query_cache_path(query)
            np.save(cache_path, embedding)

        return embedding

    def search(self, query: str, k: int = 10) -> List[Chunk]:
        """Retrieve top-K chunks by cosine similarity.

        Since embeddings are normalized, dot product = cosine similarity.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if k < 1:
            raise ValueError("k must be >= 1")

        query_embedding = self._encode_query(query)

        similarities = self.embeddings @ query_embedding

        top_k_indices = np.argpartition(
            -similarities, kth=min(k, len(similarities) - 1)
        )[:k]
        top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]

        return [self.chunks[idx] for idx in top_k_indices]
