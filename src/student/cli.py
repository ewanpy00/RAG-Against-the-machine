import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from student.ingestion.reader import Reader
from student.ingestion.chunker import Chunker, ChunkerManager
from student.ingestion.indexer import Indexer
from student.retrieval.searcher import Searcher
from student.models import (
    Chunk,
    RagDataset,
    StudentSearchResults,
    MinimalSearchResults,
    MinimalSource,
)


class CLI:
    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        output_dir: str = "data/processed",
        max_chunk_size: int = 2000,
    ) -> None:

        repo_path_obj = Path(repo_path)
        output_dir_obj = Path(output_dir)
        
        print("Reading files...")
        reader = Reader(repo_path=repo_path_obj)
        records = reader.read()
        print(f"Read {len(records)} files")
        
        print("Chunking...")
        chunker = Chunker(chunk_size=max_chunk_size)
        chunks: List[Chunk] = []
        for record in tqdm(records, desc="Chunking"):
            chunks.extend(chunker.chunk_record(record))
        print(f"Created {len(chunks)} chunks")
        
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
        searcher = Searcher()
        results = searcher.search(query, k=k)
        
        print(f"\n🔍 Query: {query}")
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
            print(f"\nPreview:")
            print(chunk.text[:300])
            print("...\n")
    
    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 10,
    ) -> None:
        dataset_path_obj = Path(dataset_path)
        save_dir = Path(save_directory)
        
        try:
            with dataset_path_obj.open("r", encoding="utf-8") as f:
                dataset = RagDataset(**json.load(f))
        except FileNotFoundError:
            print(f"Error: Dataset not found: {dataset_path}")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON: {e}")
            return
        
        print(f"Loaded {len(dataset.rag_questions)} questions")
        
        searcher = Searcher()
        
        search_results = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            chunks = searcher.search(question.question, k=k)
            
            sources = [
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
                for chunk in chunks
            ]
            
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(search_results=search_results, k=k)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / dataset_path_obj.name
        
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, indent=2)
        
        print(f"\nSaved student_search_results to {output_file}")