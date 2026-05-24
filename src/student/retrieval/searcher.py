import json
from pathlib import Path

import bm25s
import numpy as np

from student.models import Chunk
from student.retrieval.query_expander import QueryExpander
from student.ingestion.preprocessor import expand_identifiers


class Searcher:
    """Loads a BM25 index and retrieves the most relevant chunks for a query.

    Applies the same expand_identifiers() preprocessing to queries as the
    Indexer applies to documents so that snake_case / CamelCase tokens
    are matched correctly.
    """

    def __init__(
        self,
        index_dir: Path = Path("data/processed/bm25_index"),
        chunks_path: Path = Path("data/processed/chunks.json"),
        expand: bool = True,
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
        self.expander = QueryExpander(expand=expand)

    def _preprocess_query(self, query: str) -> str:
        """Apply synonym expansion + identifier splitting to the query."""
        expanded = self.expander.expand(query)
        return expand_identifiers(expanded)

    def search(self, query: str, k: int = 10) -> list[Chunk]:
        """Return top-k chunks most relevant to the query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if k < 1:
            raise ValueError("k must be >= 1")

        processed = self._preprocess_query(query)
        query_tokens = bm25s.tokenize(processed, show_progress=False)
        results, _ = self.bm25.retrieve(
            query_tokens,
            corpus=np.arange(len(self.chunks)),
            k=k,
        )

        return [self.chunks[idx] for idx in results[0]]
