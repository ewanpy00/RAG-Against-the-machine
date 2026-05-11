import bm25s

from pathlib import Path
from student.models import Chunk


class Indexer:
    def __init__(self, output_dir: Path = Path("data/processed")):
        self.output_dir = output_dir

    def index_chunks(self, chunks: list[Chunk]):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not chunks:
            raise ValueError("Cannot index empty chunks list")
    
    
        if not all(isinstance(c, Chunk) for c in chunks):
            raise TypeError("All items must be Chunk instances")

        chunks_text = [chunk.text for chunk in chunks]
        tokens = bm25s.tokenize(chunks_text)
        bm25_index = bm25s.BM25()
        bm25_index.index(tokens)

        bm25_index.save(str(self.output_dir / "bm25_index"))
