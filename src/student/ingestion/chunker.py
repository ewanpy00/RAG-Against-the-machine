import ast
import json
import re
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

    def _node_bounds(
        self,
        node: ast.AST,
        line_offsets: list[int],
        content_len: int,
    ) -> tuple[int, int] | None:
        """Return (first_char, last_char) for an AST node, or None."""
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return None
        end_line = node.end_lineno
        if end_line is None:
            return None
        first_char = line_offsets[node.lineno - 1]
        last_char = (
            line_offsets[end_line]
            if end_line < len(line_offsets)
            else content_len
        )
        return first_char, last_char

    def _make_chunks(
        self,
        record: FileRecord,
        first_char: int,
        last_char: int,
        prefix: str = "",
    ) -> list[Chunk]:
        """Emit one or more overlapping chunks for a character range.

        `prefix` is prepended to chunk text for BM25 indexing only —
        character indices always point into the original file content.
        """
        text = record.content[first_char:last_char]
        if not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [Chunk(
                chunk_id=f"{record.filepath}:{first_char}-{last_char}",
                file_path=record.filepath,
                first_character_index=first_char,
                last_character_index=last_char,
                text=prefix + text,
                file_type=record.filetype,
            )]
        step = self.chunk_size - 200
        chunks = []
        for i in range(0, len(text), step):
            abs_first = first_char + i
            abs_last = min(first_char + i + self.chunk_size, last_char)
            chunks.append(Chunk(
                chunk_id=f"{record.filepath}:{abs_first}-{abs_last}",
                file_path=record.filepath,
                first_character_index=abs_first,
                last_character_index=abs_last,
                text=prefix + text[i:i + self.chunk_size],
                file_type=record.filetype,
            ))
        return chunks

    def _chunk_class_body(
        self,
        record: FileRecord,
        class_node: ast.ClassDef,
        line_offsets: list[int],
        content_len: int,
        first_char: int,
        last_char: int,
        class_prefix: str,
    ) -> list[Chunk]:
        """Split a large class into per-method chunks, recursing into nested classes."""
        chunks: list[Chunk] = []
        children = list(ast.iter_child_nodes(class_node))
        callables = [
            n for n in children
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        nested_classes = [n for n in children if isinstance(n, ast.ClassDef)]

        if not callables and not nested_classes:
            return self._make_chunks(record, first_char, last_char, prefix=class_prefix)

        # Class header up to first child node
        first_child = min(
            (n for n in children if hasattr(n, "lineno")),
            key=lambda n: n.lineno,
            default=None,
        )
        if first_child:
            fc_bounds = self._node_bounds(first_child, line_offsets, content_len)
            if fc_bounds and fc_bounds[0] > first_char:
                chunks.extend(
                    self._make_chunks(record, first_char, fc_bounds[0], prefix=class_prefix)
                )

        for method in callables:
            mbounds = self._node_bounds(method, line_offsets, content_len)
            if mbounds:
                chunks.extend(self._make_chunks(record, *mbounds, prefix=class_prefix))

        for nested in nested_classes:
            nbounds = self._node_bounds(nested, line_offsets, content_len)
            if nbounds is None:
                continue
            nfc, nlc = nbounds
            nested_prefix = f"{class_prefix}# nested class: {nested.name}\n"
            nested_text = record.content[nfc:nlc]
            if len(nested_text) <= self.chunk_size:
                chunks.extend(self._make_chunks(record, nfc, nlc, prefix=nested_prefix))
            else:
                chunks.extend(self._chunk_class_body(
                    record, nested, line_offsets, content_len,
                    nfc, nlc, nested_prefix,
                ))

        return chunks if chunks else self._make_chunks(
            record, first_char, last_char, prefix=class_prefix
        )

    def chunk_py(self, record: FileRecord) -> list[Chunk]:
        """Chunk a Python file using AST boundaries."""
        try:
            tree = ast.parse(record.content)
        except Exception as e:
            print(f"Error parsing {record.filepath}: {e}")
            return self.chunk_generic(record)

        line_offsets = self.compute_line_offsets(record.content)
        content_len = len(record.content)
        chunks: list[Chunk] = []

        # File path prefix injected into every chunk so BM25 can match
        # e.g. "ModelRunner" in the question to "model_runner.py" in the index.
        file_prefix = f"# file: {record.filepath}\n"

        # 1. Module-level code: imports, constants, assignments, docstrings.
        #    These are skipped by function/class-only iteration but often hold
        #    the exact constants or config values questions ask about.
        module_ranges: list[tuple[int, int]] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            bounds = self._node_bounds(node, line_offsets, content_len)
            if bounds:
                module_ranges.append(bounds)

        if module_ranges:
            module_ranges.sort()
            merged: list[list[int]] = [list(module_ranges[0])]
            for fc, lc in module_ranges[1:]:
                if fc <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], lc)
                else:
                    merged.append([fc, lc])
            for fc, lc in merged:
                chunks.extend(self._make_chunks(record, fc, lc, prefix=file_prefix))

        # 2. Top-level functions → one chunk each (split if large).
        #    Top-level classes → one chunk if small; otherwise split by method
        #    so method boundaries are never cut mid-body.
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bounds = self._node_bounds(node, line_offsets, content_len)
                if bounds:
                    chunks.extend(self._make_chunks(
                        record, *bounds, prefix=file_prefix
                    ))

            elif isinstance(node, ast.ClassDef):
                bounds = self._node_bounds(node, line_offsets, content_len)
                if bounds is None:
                    continue
                first_char, last_char = bounds
                class_text = record.content[first_char:last_char]
                # Class prefix includes class name so method chunks stay
                # discoverable even without the class definition line.
                class_prefix = f"# file: {record.filepath}\n# class: {node.name}\n"

                if len(class_text) <= self.chunk_size:
                    chunks.extend(self._make_chunks(
                        record, first_char, last_char, prefix=class_prefix
                    ))
                else:
                    chunks.extend(self._chunk_class_body(
                        record, node, line_offsets, content_len,
                        first_char, last_char, class_prefix,
                    ))

        return chunks if chunks else self.chunk_generic(record)

    def chunk_generic(self, record: FileRecord, prefix: str = "") -> list[Chunk]:
        """Chunk any file by fixed character size with overlap."""
        chunks = []
        content_length = len(record.content)
        step = self.chunk_size - 200
        for i in range(0, content_length, step):
            chunk_text = record.content[i:i + self.chunk_size]
            chunks.append(Chunk(
                chunk_id=f"{record.filepath}:{i}-{i + self.chunk_size}",
                file_path=record.filepath,
                first_character_index=i,
                last_character_index=min(i + self.chunk_size, content_length),
                text=prefix + chunk_text,
                file_type=record.filetype,
            ))
        return chunks

    def chunk_md(self, record: FileRecord) -> list[Chunk]:
        """Chunk a markdown file by headers with file-path prefix.

        Sections smaller than chunk_size are merged greedily; oversized
        sections are split with overlap via _make_chunks.
        """
        content = record.content
        file_prefix = f"# source: {record.filepath}\n"
        header_re = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
        matches = list(header_re.finditer(content))

        if not matches:
            return self.chunk_generic(record, prefix=file_prefix)

        # Build section boundaries
        boundaries: list[tuple[int, int]] = []
        if matches[0].start() > 0:
            boundaries.append((0, matches[0].start()))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            boundaries.append((m.start(), end))

        # Greedily merge small sections so each chunk is ~chunk_size
        chunks: list[Chunk] = []
        buf_start, buf_end = boundaries[0]
        for start, end in boundaries[1:]:
            if end - buf_start <= self.chunk_size:
                buf_end = end
            else:
                chunks.extend(self._make_chunks(record, buf_start, buf_end, prefix=file_prefix))
                buf_start, buf_end = start, end
        chunks.extend(self._make_chunks(record, buf_start, buf_end, prefix=file_prefix))
        return chunks if chunks else self.chunk_generic(record, prefix=file_prefix)

    def chunk_record(self, record: FileRecord) -> list[Chunk]:
        """Dispatch to the appropriate chunking strategy based on file type."""
        valid_types = [".py", ".txt", ".md", ".rst"]
        if record.filetype not in valid_types:
            return []
        if record.filetype == ".py":
            return self.chunk_py(record)
        if record.filetype == ".md":
            return self.chunk_md(record)
        file_prefix = f"# source: {record.filepath}\n"
        return self.chunk_generic(record, prefix=file_prefix)
