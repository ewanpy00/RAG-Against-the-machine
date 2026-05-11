from pathlib import Path
from student.ingestion.reader import read_records
from student.ingestion.chunker import Chunker, ChunkerManager
from student.ingestion.indexer import Indexer
from student.retrieval.searcher import Searcher

class IngestationManager:
    def __init__(self):
        self.repo_path = Path("data/raw/vllm-0.10.1")

    def ingest_data(self, data_source):
        #processing records
        records = read_records(self.repo_path)

        chunks = []

        #processing chunks
        chunkManager = ChunkerManager(chunk_dir=Path("data/processed/chunks.json"))
        for record in records:
            chunker = Chunker(chunk_size=2000)
            chunks.extend(chunker.chunk_record(record))
        chunkManager.save_chunks(chunks)

        #indexing chunks
        indexer = Indexer(output_dir=Path("data/processed"))
        indexer.index_chunks(chunks)

        # searching chunks
        searcher = Searcher()
        results = searcher.search("def forward")
        print(f"Search results for 'def forward': {len(results)} chunks found")
        for chunk in results[:5]:
            print(f"Chunk ID: {chunk.chunk_id}, File: {chunk.file_path}, Type: {chunk.file_type}")
