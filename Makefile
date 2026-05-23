DATASET_ANSWERED   = data/datasets/AnsweredQuestions
DATASET_UNANSWERED = data/datasets/UnansweredQuestions
OUTPUT_DIR         = data/output/search_results
K                  = 10
RETRIEVER          = bm25
EMBEDDING		   = False
CHUNK_SIZE 		   = 2000

help:
	@echo "Usage:"
	@echo "  make install                            Install dependencies"
	@echo "  make index                              Read, chunk and build BM25 index"
	@echo "  make search QUERY='...'                 Search the index"
	@echo "  make search-dataset                     Run search on all datasets"
	@echo "  make search-dataset RETRIEVER=embedding Run search with embedding retriever"
	@echo "  make search-dataset RETRIEVER=hybrid    Run search with hybrid retriever"
	@echo "  make evaluate                           Evaluate recall@k on answered datasets"
	@echo "  make answer QUERY='...'                 Answer the given question"
	@echo "  make answer-dataset                     Answer questions on all datasets"
	@echo "  make clean                              Clean up generated files and caches"
	@echo "  make debug                              Run search with pdb for debugging"
	@echo "  make run                                Run example indexing and searching"


run:
	uv run python -m student index
	uv run python -m student search "How does vLLM handle batching?" --k 5

debug:
	uv run python -m pdb -m student search "test query"

clean:
	rm -rf data/processed/
	rm -rf data/output/
	rm -rf data/cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete


lint:
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

install:
	uv pip install -e .

index:
	uv run python -m student index --build_embeddings $(EMBEDDING) --max_chunk_size $(CHUNK_SIZE)

search:
	uv run python -m student search --query "$(QUERY)" --k $(K)

search-dataset:
	uv run python -m student search_dataset \
		--dataset_path $(DATASET_UNANSWERED)/dataset_code_public.json \
		--save_directory $(OUTPUT_DIR) --k $(K) --retriever $(RETRIEVER)
	uv run python -m student search_dataset \
		--dataset_path $(DATASET_UNANSWERED)/dataset_docs_public.json \
		--save_directory $(OUTPUT_DIR) --k $(K) --retriever $(RETRIEVER)

evaluate:
	uv run python -m student evaluate \
		--student_results_path $(OUTPUT_DIR)/dataset_code_public.json \
		--ground_truth_path $(DATASET_ANSWERED)/dataset_code_public.json
	uv run python -m student evaluate \
		--student_results_path $(OUTPUT_DIR)/dataset_docs_public.json \
		--ground_truth_path $(DATASET_ANSWERED)/dataset_docs_public.json

answer:
	uv run python -m student answer "$(QUERY)" --k $(K)

answer-dataset:
	uv run python -m student answer_dataset \
		--student_search_results_path $(OUTPUT_DIR)/dataset_code_public.json \
		--save_directory data/output/search_results_and_answer --k $(K)
	uv run python -m student answer_dataset \
		--student_search_results_path $(OUTPUT_DIR)/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer --k $(K)


.PHONY: install run debug clean lint index search search-dataset evaluate answer answer-dataset help