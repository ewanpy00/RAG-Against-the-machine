import ast
import json
from pathlib import Path

from student.models import Chunk, FileRecord


class ChunkerManager:
    """Saves and loads chunks to/from a JSON file."""

    def __init__(self, chunk_dir: Path) -> None:
        self.chunk_dir = chunk_dir

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Serialize chunks to a single JSON file."""
        self.chunk_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(self.chunk_dir, "w", encoding="utf-8") as f:
            json.dump(
                [chunk.model_dump() for chunk in chunks],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load_chunks(self) -> list[Chunk]:
        """Deserialize chunks from JSON file."""
        with open(self.chunk_dir, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Chunk(**item) for item in data]


class Chunker:
    """Splits FileRecords into Chunks using AST for .py files."""

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def compute_line_offsets(self, content: str) -> list[int]:
        """Return a list mapping line index to its start character offset."""
        offsets = [0]
        for i, char in enumerate(content):
            if char == "\n":
                offsets.append(i + 1)
        return offsets

    def chunk_py(self, record: FileRecord) -> list[Chunk]:
        """Chunk a Python file by top-level functions and classes."""
        try:
            tree = ast.parse(record.content)
            line_offsets = self.compute_line_offsets(record.content)
            chunks: list[Chunk] = []

            for node in ast.iter_child_nodes(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue

                start_line = node.lineno - 1
                end_line = node.end_lineno
                if end_line is None:
                    continue

                first_char = line_offsets[start_line]
                last_char = (
                    line_offsets[end_line]
                    if end_line < len(line_offsets)
                    else len(record.content)
                )
                node_text = record.content[first_char:last_char]

                if len(node_text) <= self.chunk_size:
                    chunks.append(Chunk(
                        chunk_id=f"{record.filepath}:{first_char}-{last_char}",
                        file_path=record.filepath,
                        first_character_index=first_char,
                        last_character_index=last_char,
                        text=node_text,
                        file_type=record.filetype,
                    ))
                else:
                    for i in range(0, len(node_text), self.chunk_size):
                        abs_first = first_char + i
                        abs_last = min(first_char + i + self.chunk_size, last_char)
                        chunks.append(Chunk(
                            chunk_id=f"{record.filepath}:{abs_first}-{abs_last}",
                            file_path=record.filepath,
                            first_character_index=abs_first,
                            last_character_index=abs_last,
                            text=node_text[i:i + self.chunk_size],
                            file_type=record.filetype,
                        ))

            return chunks
        except Exception as e:
            print(f"Error parsing {record.filepath}: {e}")
            return []

    def chunk_generic(self, record: FileRecord) -> list[Chunk]:
        """Chunk any file by fixed character size."""
        chunks: list[Chunk] = []
        content_length = len(record.content)
        for i in range(0, content_length, self.chunk_size):
            chunk_text = record.content[i:i + self.chunk_size]
            chunks.append(Chunk(
                chunk_id=f"{record.filepath}:{i}-{i + self.chunk_size}",
                file_path=record.filepath,
                first_character_index=i,
                last_character_index=min(i + self.chunk_size, content_length),
                text=chunk_text,
                file_type=record.filetype,
            ))
        return chunks

    def chunk_record(self, record: FileRecord) -> list[Chunk]:
        """Dispatch to the appropriate chunking strategy based on file type."""
        valid_types = [".py", ".txt", ".md"]
        if record.filetype not in valid_types:
            return []
        if record.filetype == ".py":
            try:
                return self.chunk_py(record)
            except Exception as e:
                print(f"Error parsing {record.filepath}: {e}")
                return []
        return self.chunk_generic(record)
