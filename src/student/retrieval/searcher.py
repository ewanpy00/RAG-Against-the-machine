"""BM25-based search engine for code retrieval."""

import json
from pathlib import Path

import bm25s
import numpy as np

from student.models import Chunk


class Searcher:
    """Loads a BM25 index and retrieves the most relevant chunks for a query."""

    def __init__(
        self,
        index_dir: Path = Path("data/processed/bm25_index"),
        chunks_path: Path = Path("data/processed/chunks.json"),
    ) -> None:
        if not index_dir.exists():
            raise FileNotFoundError(
                f"Index not found: {index_dir}. Run 'index' command first."
            )
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks not found: {chunks_path}. Run 'index' command first."
            )

        self.bm25 = bm25s.BM25.load(str(index_dir), load_corpus=True)

        with chunks_path.open("r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self.chunks: list[Chunk] = [Chunk(**data) for data in chunks_data]

    def search(self, query: str, k: int = 10) -> list[Chunk]:
        """Return top-k chunks most relevant to the query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if k < 1:
            raise ValueError("k must be >= 1")

        query_tokens = bm25s.tokenize(query)
        results, _ = self.bm25.retrieve(
            query_tokens,
            corpus=np.arange(len(self.chunks)),
            k=k,
        )

        return [self.chunks[idx] for idx in results[0]]
