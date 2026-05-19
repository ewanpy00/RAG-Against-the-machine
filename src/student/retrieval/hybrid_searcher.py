from typing import List

from student.models import Chunk
from student.retrieval.searcher import Searcher
from student.retrieval.embedding_searcher import EmbeddingSearcher


class HybridSearcher:
    def __init__(
        self,
        bm25_searcher: Searcher = None,
        embedding_searcher: EmbeddingSearcher = None,
        rrf_k: int = 60,
    ):
        """Initialize hybrid searcher.

        Args:
            bm25_searcher: Existing BM25 searcher (created if None)
            embedding_searcher: Existing embedding searcher (created if None)
            rrf_k: RRF constant (default 60, see formula below)
        """
        self.bm25 = bm25_searcher or Searcher()
        self.embeddings = embedding_searcher or EmbeddingSearcher()
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 10, candidate_k: int = 50) -> List[Chunk]:
        """Hybrid search via RRF fusion.

        Args:
            query: Search query
            k: Final number of results
            candidate_k: How many candidates from each retriever
                (larger = better fusion, default 50)

        Returns:
            Top-K chunks ranked by RRF score
        """
        bm25_chunks = self.bm25.search(query, k=candidate_k)
        embedding_chunks = self.embeddings.search(query, k=candidate_k)

        rrf_scores = {}

        for rank, chunk in enumerate(bm25_chunks, start=1):
            chunk_id = chunk.chunk_id
            score = 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score

        for rank, chunk in enumerate(embedding_chunks, start=1):
            chunk_id = chunk.chunk_id
            score = 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score

        all_chunks = {c.chunk_id: c for c in bm25_chunks + embedding_chunks}

        ranked_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )
        
        # Return top-K
        return [all_chunks[cid] for cid in ranked_ids[:k]]