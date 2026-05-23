from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm


from student.evaluation.evaluator import Evaluator
from student.generation.answerer import AnswerGenerator
from student.ingestion.reader import Reader
from student.ingestion.chunker import Chunker, ChunkerManager
from student.ingestion.indexer import Indexer
from student.retrieval.searcher import Searcher
from student.models import (
    Chunk,
    RagDataset,
    QuestionDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalSearchResults,
    MinimalAnswer,
    MinimalSource,
)


def _load_json(path: Path, label: str) -> Any | None:
    """Load and parse a JSON file with clear error messages."""
    if not path.exists():
        print(f"Error: {label} not found: {path}")
        return None
    if path.stat().st_size == 0:
        print(f"Error: {label} is empty: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: {label} contains invalid JSON: {e}")
        return None


def _validate_k(k: int) -> bool:
    """Return False and print an error if k is not a positive integer."""
    if not isinstance(k, int) or isinstance(k, bool):
        print(f"Error: k must be an integer, got {type(k).__name__}.")
        return False
    if k < 1:
        print(f"Error: k must be >= 1, got {k}.")
        return False
    return True


def _validate_query(query: str) -> bool:
    """Return False and print an error if query is empty or whitespace-only."""
    if not isinstance(query, str):
        print(f"Error: query must be a string, got {type(query).__name__}.")
        return False
    if not query.strip():
        print("Error: query must not be empty.")
        return False
    return True


def _validate_directory(path: str, label: str) -> bool:
    """Return False and print an error if directory path is empty."""
    if not isinstance(path, str) or not path.strip():
        print(f"Error: {label} must not be empty.")
        return False
    return True


class CLI:
    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        output_dir: str = "data/processed",
        max_chunk_size: int = 2000,
        build_embeddings: bool = False,
    ) -> None:
        """Index the repository into a searchable BM25 index."""
        if not _validate_directory(output_dir, "output_dir"):
            return
        if (
            not isinstance(max_chunk_size, int)
            or isinstance(max_chunk_size, bool)
        ):
            print(
                f"Error: max_chunk_size must be an integer, "
                f"got {type(max_chunk_size).__name__}."
            )
            return
        if max_chunk_size < 100:
            print(
                f"Error: max_chunk_size must be >= 100, "
                f"got {max_chunk_size}."
            )
            return
        if max_chunk_size > 100_000:
            print(
                f"Error: max_chunk_size must be <= 100000, "
                f"got {max_chunk_size}."
            )
            return

        repo_path_obj = Path(repo_path)
        if not repo_path_obj.exists():
            print(f"Error: Repository path not found: {repo_path}")
            return

        output_dir_obj = Path(output_dir)

        print("Reading files...")
        reader = Reader(repo_path=repo_path_obj)
        records = reader.read()
        print(f"Read {len(records)} files")

        print("Chunking...")
        chunker = Chunker(chunk_size=max_chunk_size)
        chunks: list[Chunk] = []
        for record in tqdm(records, desc="Chunking"):
            chunks.extend(chunker.chunk_record(record))
        print(f"Created {len(chunks)} chunks")

        if build_embeddings:
            print("\nBuilding embedding index...")
            from student.retrieval.embedding_indexer import EmbeddingIndexer
            emb_indexer = EmbeddingIndexer()
            emb_indexer.build_index(chunks, output_dir_obj)

        print("Saving chunks...")
        chunk_manager = ChunkerManager(
            chunk_dir=output_dir_obj / "chunks.json"
        )
        chunk_manager.save_chunks(chunks)

        print("Building BM25 index...")
        indexer = Indexer(output_dir=output_dir_obj)
        indexer.index_chunks(chunks)

        print(f"\nIngestion complete! Indices saved under {output_dir}/")

    def search(self, query: str, k: int = 10) -> None:
        """Search the BM25 index for a single query."""
        if not _validate_query(query):
            return
        if not _validate_k(k):
            return

        try:
            from student.retrieval.hybrid_searcher import HybridSearcher
            searcher = HybridSearcher()
        except FileNotFoundError as e:
            print(
                f"Error: hybrid indexes not found. "
                f"Run 'make index EMBEDDIBG=True' first.\n  {e}"
            )
            return

        results = searcher.search(query, k=k)

        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results\n")

        for i, chunk in enumerate(results, 1):
            print(f"{'=' * 60}")
            print(f"Result #{i}")
            print(f"{'=' * 60}")
            print(f"File: {chunk.file_path}")
            print(
                f"Range: [{chunk.first_character_index}, "
                f"{chunk.last_character_index}]"
            )
            print("\nPreview:")
            print(chunk.text[:300])
            print("...\n")

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 10,
        retriever: str = "bm25",
    ) -> None:
        """Run retrieval over a dataset of questions and save results."""
        if not _validate_k(k):
            return
        if not _validate_directory(save_directory, "save_directory"):
            return

        start_time = time.time()

        searcher: Searcher | EmbeddingSearcher | HybridSearcher
        try:
            if retriever == "bm25":
                searcher = Searcher()
            elif retriever == "embedding":
                from student.retrieval.embedding_searcher import (
                    EmbeddingSearcher,
                )
                searcher = EmbeddingSearcher(use_cache=True)
            elif retriever == "hybrid":
                from student.retrieval.embedding_searcher import (
                    EmbeddingSearcher,
                )
                from student.retrieval.hybrid_searcher import HybridSearcher
                searcher = HybridSearcher(
                    embedding_searcher=EmbeddingSearcher(use_cache=True)
                )
            else:
                print(
                    f"Error: Invalid retriever type '{retriever}'. "
                    f"Choose: bm25, embedding, hybrid"
                )
                return
        except ValueError as e:
            print(f"Error: {e}")
            return
        except FileNotFoundError as e:
            if "embeddings" in str(e).lower():
                print(
                    f"Error: Embedding index not found. "
                    f"Run 'make index build_embeddings=True' first.\n  {e}"
                )
            else:
                print(
                    f"Error: Index not found. "
                    f"Run 'make index' first.\n  {e}"
                )
            return

        dataset_path_obj = Path(dataset_path)
        save_dir = Path(save_directory)

        raw = _load_json(dataset_path_obj, "Dataset")
        if raw is None:
            return

        if isinstance(raw, list):
            print(
                f"Error: Dataset file contains a plain list — "
                f"expected an object with 'rag_questions' key.\n"
                f"  Got {len(raw)} items. "
                f"Check that you're pointing at the right file."
            )
            return

        try:
            questions = QuestionDataset(**raw).rag_questions
        except Exception as e:
            print(f"Error: Failed to parse dataset: {e}")
            return

        if not questions:
            print("Error: Dataset contains no questions.")
            return

        print(f"Loaded {len(questions)} questions")

        search_results = []
        REPO_PREFIX = "data/raw/vllm-0.10.1/"

        for question in tqdm(questions, desc="Searching"):
            chunks = searcher.search(question.question, k=k)

            sources = [
                MinimalSource(
                    file_path=f"{REPO_PREFIX}{chunk.file_path}",
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
                for chunk in chunks
            ]

            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question_str=question.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(search_results=search_results, k=k)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / dataset_path_obj.name

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, indent=2)

        elapsed = time.time() - start_time
        print(f"\nSaved student_search_results to {output_file}")
        print(
            f"Total time: {elapsed:.2f}s "
            f"({elapsed / len(questions):.3f}s per question)"
        )

    def evaluate(
        self,
        student_results_path: str,
        ground_truth_path: str,
    ) -> None:
        """Evaluate recall@k of search results against ground truth."""
        raw = _load_json(Path(student_results_path), "Student results")
        if raw is None:
            return

        if isinstance(raw, list):
            print(
                "Error: Student results file contains a plain list "
                "— expected an object with 'search_results' key.\n"
                "  Run 'make search-dataset' first to generate "
                "the correct format."
            )
            return

        try:
            student_results = StudentSearchResults(**raw)
        except Exception as e:
            print(
                f"Error: Student results file has wrong structure.\n"
                f"  Run 'make search-dataset' to regenerate it.\n  {e}"
            )
            return

        if not student_results.search_results:
            print("Error: Student results contain no entries.")
            return

        raw_gt = _load_json(Path(ground_truth_path), "Ground truth dataset")
        if raw_gt is None:
            return

        if isinstance(raw_gt, list):
            print(
                "Error: Ground truth file contains a plain list — expected an "
                "object with 'rag_questions' key."
            )
            return

        try:
            ground_truth = RagDataset(**raw_gt)
        except Exception as e:
            print(f"Error: Failed to parse ground truth dataset: {e}")
            return

        evaluator = Evaluator()
        recall_at_k = evaluator.evaluate(
            student_results=student_results,
            ground_truth=ground_truth,
            ks=[1, 3, 5, 10],
        )

        print("\nEvaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {len(student_results.search_results)}")
        for k_val in sorted(recall_at_k.keys()):
            print(f"Recall@{k_val}: {recall_at_k[k_val]:.4f}")

    def answer(self, query: str, k: int = 10) -> None:
        """Answer a single question using retrieved context."""
        if not _validate_query(query):
            return
        if not _validate_k(k):
            return

        print("Initializing...")
        try:
            searcher = Searcher()
        except FileNotFoundError as e:
            print(
                f"Error: BM25 index not found. "
                f"Run 'make index' first.\n  {e}"
            )
            return

        generator = AnswerGenerator()

        print(f"\nSearching for: {query}")
        chunks = searcher.search(query, k=k)
        print(f"Found {len(chunks)} chunks")

        print("\nGenerating answer...")
        answer = generator.generate(query, chunks)

        print(f"\nAnswer:\n{answer}")

        print("\nSources:")
        for chunk in chunks[:5]:
            print(f"  - {chunk.file_path}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        k: int = 10,
    ) -> None:
        """Generate answers for existing search results and save enriched
        output."""
        if not _validate_k(k):
            return
        if not _validate_directory(save_directory, "save_directory"):
            return

        raw = _load_json(Path(student_search_results_path), "Search results")
        if raw is None:
            return

        if isinstance(raw, list):
            print(
                "Error: Expected a search results object, not a list. "
                "Run 'make search-dataset' first."
            )
            return

        try:
            search_results = StudentSearchResults(**raw)
        except Exception as e:
            print(f"Error: Failed to parse search results: {e}")
            return

        if not search_results.search_results:
            print("Error: Search results are empty.")
            return

        print(f"Loaded {len(search_results.search_results)} questions")

        try:
            generator = AnswerGenerator()
        except Exception as e:
            print(f"Error: Failed to initialize answer generator: {e}")
            return

        answers = []

        for result in tqdm(search_results.search_results, desc="Answering"):
            chunks: list[Chunk] = []
            for source in result.retrieved_sources[:k]:
                try:
                    with open(source.file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    start = source.first_character_index
                    end = source.last_character_index
                    text = content[start:end]
                    chunks.append(Chunk(
                        chunk_id=(
                            f"{source.file_path}:"
                            f"{source.first_character_index}-"
                            f"{source.last_character_index}"
                        ),
                        file_path=source.file_path,
                        first_character_index=source.first_character_index,
                        last_character_index=source.last_character_index,
                        text=text,
                        file_type=Path(source.file_path).suffix,
                    ))
                except OSError:
                    continue

            answer_text = generator.generate(result.question_str, chunks)
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question_str=result.question_str,
                retrieved_sources=result.retrieved_sources,
                answer=answer_text,
            ))

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.k,
        )
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / Path(student_search_results_path).name

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, indent=2)

        print(f"\nSaved answers to {output_file}")
