.PHONY: install index search evaluate help

DATASET_ANSWERED   = data/datasets/AnsweredQuestions
DATASET_UNANSWERED = data/datasets/UnansweredQuestions
OUTPUT_DIR         = data/output/search_results
K                  = 10

help:
	@echo "Usage:"
	@echo "  make install                 Install dependencies"
	@echo "  make index                   Read, chunk and build BM25 index"
	@echo "  make search QUERY='...'      Search the index"
	@echo "  make search-dataset          Run search on all datasets"
	@echo "  make evaluate                Evaluate recall@k on answered datasets"
	@echo "  make answer QUERY='...'      Answer the given question"
	@echo "  make answer-dataset          Answer the given question"


install:
	uv pip install -e .

index:
	uv run python -m student index

search:
	python -m student search --query "$(QUERY)" --k $(K)

search-dataset:
	uv run python -m student search_dataset \
		--dataset_path $(DATASET_UNANSWERED)/dataset_code_public.json \
		--save_directory $(OUTPUT_DIR) --k $(K)
	uv run python -m student search_dataset \
		--dataset_path $(DATASET_UNANSWERED)/dataset_docs_public.json \
		--save_directory $(OUTPUT_DIR) --k $(K)

evaluate:
	uv run python -m student evaluate \
		--student_results_path $(OUTPUT_DIR)/dataset_code_public.json \
		--ground_truth_path $(DATASET_ANSWERED)/dataset_code_public.json \
		--k $(K)
	uv run python -m student evaluate \
		--student_results_path $(OUTPUT_DIR)/dataset_docs_public.json \
		--ground_truth_path $(DATASET_ANSWERED)/dataset_docs_public.json \
		--k $(K)

answer:
	uv run python -m student answer "$(QUERY)" --k $(K)

answer-dataset:
	uv run python -m student answer_dataset \
		--dataset_path $(DATASET_UNANSWERED)/dataset_code_public.json \
		--save_directory $(OUTPUT_DIR) --k $(K)
	uv run python -m student answer_dataset \
		--dataset_path $(DATASET_UNANSWERED)/dataset_docs_public.json \
		--save_directory $(OUTPUT_DIR) --k $(K)

clean: