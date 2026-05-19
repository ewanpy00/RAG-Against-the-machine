from pathlib import Path
from typing import List
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from student.models import Chunk


class EmbeddingSearcher:
    """Search via cosine similarity over dense embeddings."""
    def __init__(
        self,
        embeddings_path: Path = Path("data/processed/embeddings.npy"),
        chunks_path: Path = Path("data/processed/chunks.json"),
        meta_path: Path = Path("data/processed/embeddings_meta.json"),
        device: str = None,
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

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

        print(f"Loading embeddings: {embeddings_path}")
        self.embeddings = np.load(embeddings_path)

        with chunks_path.open() as f:
            chunks_data = json.load(f)
        self.chunks = [Chunk(**data) for data in chunks_data]

        assert len(self.chunks) == self.embeddings.shape[0], (
            f"Mismatch: {len(self.chunks)} chunks vs {self.embeddings.shape[0]} embeddings"
        )

    def search(self, query: str, k: int = 10) -> List[Chunk]:
        """Retrieve top-K chunks by cosine similarity.
        
        Since embeddings are normalized, dot product = cosine similarity.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if k < 1:
            raise ValueError("k must be >= 1")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        similarities = self.embeddings @ query_embedding

        top_k_indices = np.argpartition(-similarities, kth=min(k, len(similarities) - 1))[:k]
        top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
        
        return [self.chunks[idx] for idx in top_k_indices]
    
    def search_with_scores(self, query: str, k: int = 10) -> List[tuple]:
        """Same as search but returns (chunk, score) tuples."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        
        similarities = self.embeddings @ query_embedding
        top_k_indices = np.argpartition(-similarities, kth=min(k, len(similarities) - 1))[:k]
        top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
        
        return [(self.chunks[idx], float(similarities[idx])) for idx in top_k_indices]
