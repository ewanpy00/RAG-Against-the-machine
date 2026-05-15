from pathlib import Path

from student.ingestion.chunker import Chunker, ChunkerManager
from student.ingestion.indexer import Indexer
from student.ingestion.reader import Reader
from student.retrieval.searcher import Searcher


class IngestationManager:
    """Orchestrates the full ingestion pipeline: read → chunk → index."""

    def __init__(self) -> None:
        self.repo_path = Path("data/raw/vllm-0.10.1")

    def ingest_data(self) -> None:
        """Run the full ingestion pipeline and verify search works."""
        # Read
        records = Reader(repo_path=self.repo_path).read()

        # Chunk
        chunk_manager = ChunkerManager(
            chunk_dir=Path("data/processed/chunks.json"))
        chunker = Chunker(chunk_size=2000)
        chunks = []
        for record in records:
            chunks.extend(chunker.chunk_record(record))
        chunk_manager.save_chunks(chunks)

        # Index
        Indexer(output_dir=Path("data/processed")).index_chunks(chunks)

        # Smoke-test search
        searcher = Searcher()
        results = searcher.search("def forward")
        print(f"Search results for 'def forward': {len(results)} chunks found")
        for chunk in results[:5]:
            print(f"Chunk ID: {chunk.chunk_id}, ", end="")
            print(f"File: {chunk.file_path}, Type: {chunk.file_type}")
