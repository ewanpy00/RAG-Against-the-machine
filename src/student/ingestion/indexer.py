import bm25s

from pathlib import Path

from student.models import Chunk
from student.ingestion.preprocessor import expand_identifiers


class Indexer:
    """Builds and saves a BM25 index from a list of chunks.

    Tuned parameters for code search:
    - k1=1.2  : softer term-frequency saturation (default 1.5 is too aggressive)
    - b=0.4   : less length-normalisation (code files vary wildly in size)

    Each chunk is preprocessed with expand_identifiers() so that snake_case
    and CamelCase tokens are split into their component parts before indexing.
    The file path is also prepended to help queries that mention a module name.
    """

    K1: float = 1.2
    B: float = 0.4

    def __init__(self, output_dir: Path = Path("data/processed")) -> None:
        self.output_dir = output_dir

    @staticmethod
    def _prepare_text(chunk: Chunk) -> str:
        """Return text enriched with file-path context and split identifiers."""
        path_hint = chunk.file_path.replace("/", " ").replace("_", " ").replace(".", " ")
        raw = f"{path_hint} {chunk.text}"
        return expand_identifiers(raw)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Tokenize chunks, build BM25 index, and save to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not chunks:
            raise ValueError("Cannot index empty chunks list")

        if not all(isinstance(c, Chunk) for c in chunks):
            raise TypeError("All items must be Chunk instances")

        chunks_text = [self._prepare_text(c) for c in chunks]
        tokens = bm25s.tokenize(chunks_text, show_progress=False)
        bm25_index = bm25s.BM25(k1=self.K1, b=self.B)
        bm25_index.index(tokens)

        bm25_index.save(str(self.output_dir / "bm25_index"))
