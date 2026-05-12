*This project has been created as part of the 42 curriculum by ipykhtin.*

# RAG against the machine

A Retrieval-Augmented Generation (RAG) system that answers questions about the vLLM codebase by retrieving relevant code snippets and documentation, then generating answers using a small LLM.

---

## Description

This project implements an end-to-end RAG pipeline that:

1. **Ingests** the vLLM repository (Python code and Markdown documentation)
2. **Indexes** content using BM25 lexical search
3. **Retrieves** the most relevant chunks for a given question
4. **Generates** natural-language answers grounded in retrieved context using Qwen3-0.6B
5. **Evaluates** retrieval quality using Recall@K metrics

The system was built incrementally with a walking-skeleton approach: first a minimal end-to-end pipeline, then optimizations measured against a ground-truth dataset.

---

## Instructions

### Prerequisites
- Python 3.10+
- `uv` package manager
- vLLM repository placed at `data/raw/vllm-0.10.1/`

### Installation
```bash
# Install dependencies
uv venv && uv sync

# Or via make
make install
```

### Running the pipeline
```bash
# 1. Build the index (one-time, ~2 minutes)
uv run python -m student index --max_chunk_size 2000

# 2. Search a single query
uv run python -m student search "How to configure the OpenAI server?" --k 10

# 3. Process a dataset of questions
uv run python -m student search_dataset \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results \
    --k 10

# 4. Evaluate retrieval quality
uv run python -m student evaluate \
    --student_results_path data/output/search_results/dataset_docs_public.json \
    --ground_truth_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10

# 5. Generate answers using LLM
uv run python -m student answer "How does vLLM handle batching?" --k 5
```

### Makefile targets
```bash
make install      # install dependencies
make run          # run main pipeline
make lint         # run flake8 + mypy
make clean        # remove caches and processed data
```

---

## System architecture

The RAG pipeline consists of two distinct phases:

### Offline Phase (Ingestion)
Run once via the `index` command. Produces artifacts that retrieval depends on.
Repository → Reader → Chunker → ChunkManager (save) → Indexer (BM25)
↓
data/processed/chunks.json
data/processed/bm25_index/

**Components:**
- **`Reader`** (`src/student/ingestion/reader.py`) — recursively reads files, filters by extension whitelist (`.py`, `.md`, `.txt`), excludes hidden and build directories.
- **`Chunker`** (`src/student/ingestion/chunker.py`) — splits file contents into fixed-size chunks with strict character-based boundaries.
- **`ChunkManager`** — persists chunks to JSON for later retrieval of metadata and text.
- **`Indexer`** (`src/student/ingestion/indexer.py`) — builds a BM25 index using the `bm25s` library.

### Online Phase (Retrieval + Generation)
Used at query time.
Query → Searcher → top-K Chunks → AnswerGenerator (LLM) → Answer

**Components:**
- **`Searcher`** (`src/student/retrieval/searcher.py`) — loads BM25 index and chunks eagerly, performs tokenization and BM25 retrieval.
- **`AnswerGenerator`** (`src/student/generation/answerer.py`) — wraps Qwen3-0.6B, constructs prompts from retrieved context, generates answers.
- **`Evaluator`** (`src/student/evaluation/evaluator.py`) — computes Recall@K with IoU-based source matching.

---

## Chunking strategy

**Approach:** Naive fixed-size chunking with a configurable maximum size (default 2000 characters).

```python
for i in range(0, len(content), chunk_size):
    chunk = content[i:i + chunk_size]
```

**Rationale:**
- Simple, predictable, and respects the 2000-character limit from Subject (V.4)
- Easy to verify: `chunk.text length == last_char_index - first_char_index`
- Works for both Python code and Markdown documentation in a uniform way

**Trade-off:** Naive chunking can split functions or sections mid-content. Smart chunking strategies (AST-based for Python, header-based for Markdown) were considered but deferred — the baseline already met Subject thresholds. Future work: see "Challenges faced" below.

**Whitelist of indexed extensions:** `.py`, `.md`, `.txt`.

**Rationale for whitelist:**
- `.py` and `.md` are explicitly mentioned in Subject (V.4)
- `.txt` captures READMEs and similar documentation
- Binary files (`.png`, `.jpg`), config files (`.json`, `.yaml`), and compiled artifacts were excluded after experiments showed they added noise without improving recall

---

## Retrieval method

**Algorithm:** BM25 (via the `bm25s` library).

**Why BM25 over embeddings (for baseline):**
- Subject mandates TF-IDF or BM25; embeddings are bonus
- Code search benefits from exact term matching (function names, identifiers)
- No GPU required, fast indexing (~30 seconds), low memory
- Interpretable scoring

**How it works:**
1. Query is tokenized (lowercase, whitespace split via `bm25s.tokenize`)
2. BM25 scores all chunks based on term frequency and inverse document frequency
3. Top-K chunks returned, ranked by score
4. Indices are mapped back to full `Chunk` objects via `self.chunks` lookup

**Key design choice:** The BM25 library stores only token statistics — full text and metadata (file_path, character indices) live in a separate `chunks.json` artifact. The Searcher loads both eagerly at startup so per-query latency stays in the milliseconds range.

---

## Performance analysis

All metrics measured on the public datasets and verified with the official moulinette evaluator.

### Recall@K — docs dataset

| K | Recall |
|---|--------|
| 1 | 0.590 (59.0%) |
| 3 | 0.760 (76.0%) |
| 5 | **0.820 (82.0%)** ← target: 80% ✅ |
| 10 | 0.850 (85.0%) |

### Recall@K — code dataset

| K | Recall |
|---|--------|
| 1 | 0.360 (36.0%) |
| 3 | 0.460 (46.0%) |
| 5 | **0.520 (52.0%)** ← target: 50% ✅ |
| 10 | 0.610 (61.0%) |

### Analysis

**Docs perform significantly better than code:**
- Docs use natural language that matches BM25's strengths
- Code identifiers (function names, type names) are highly specific, but questions often paraphrase them
- Naive chunking splits code arbitrarily, breaking function/class boundaries

**Observation: most relevant results in top-5.** Recall@10 only adds 3% on docs and 9% on code — suggesting BM25 ranks correct sources well, with diminishing returns past top-5.

### Performance characteristics

- **Indexing time:** ~30 seconds for ~14,800 chunks
- **Cold-start latency:** under 5 seconds (Searcher initialization)
- **Per-query latency:** ~10-50 ms (BM25 lookup)

---

## Design decisions

### 1. Walking-skeleton approach
Built the minimal end-to-end pipeline first (Reader → Chunker → BM25 → Searcher → Evaluator), measured baseline metrics, then refined based on data rather than assumption.

### 2. Two-artifact storage (BM25 index + chunks.json)
- BM25 stores only token statistics
- `chunks.json` keeps text, file_path, character indices
- Separation lets us optimize each independently and avoid storing duplicated text

### 3. Eager loading in Searcher
The BM25 index and chunks are loaded in `Searcher.__init__`, not on first query. This trades a one-time ~3-5s startup cost for predictable per-query latency.

### 4. Pydantic for all data models
Type safety, automatic validation, and clear contracts between pipeline stages. Aliases (`question` ↔ `question_str`) handle Subject vs moulinette format differences.

### 5. IoU-based source matching in evaluator
Following Subject (V.6.6), a retrieved source counts as "found" when:
- `file_path` matches the ground truth
- IoU between character ranges ≥ 5%

This balances strictness (no false positives from arbitrary overlaps) with tolerance (chunker boundaries don't have to align perfectly with ground-truth boundaries).

### 6. Path prefix in output
After empirical testing, the moulinette evaluator expects `file_path` values to include the `data/raw/vllm-0.10.1/` prefix. The Subject's example shows paths without the prefix; we follow what moulinette actually validates.

---

## Challenges faced

### 1. `file_path` format mismatch
The Subject example showed relative paths (`docs/server.md`), but the ground-truth dataset and moulinette expected paths with the `data/raw/vllm-0.10.1/` prefix. Discovered when moulinette reported Recall@5 = 0% despite our internal evaluator showing 82%. Fix: prepend the prefix during `search_dataset`.

### 2. `question_str` vs `question` aliasing
Pydantic models for `MinimalSearchResults` use `question_str` as the attribute (matching moulinette) but accept `question` as an alias (matching Subject). This dual support requires `populate_by_name=True` and `by_alias=True` during serialization.

### 3. Oversized chunks from AST-based chunking
An early `chunk_py` implementation used Python AST to split files by function/class. Some classes were 150,000+ characters — far over the 2000-char limit. Removed AST chunking for the baseline; could be reintroduced with size-checking for bonus.

### 4. Building moulinette on macOS
The provided moulinette binary targets Linux (Ubuntu/Fedora). Local development and self-evaluation used a custom Python `evaluate` command (implementing the same Recall@K + IoU metric), with the official moulinette run on Linux for final verification.

### 5. BM25 corpus parameter
`bm25s.retrieve(corpus=...)` requires either a corpus list or an indexable structure when `load_corpus=False`. Passing an integer fails with `'int' object is not subscriptable`. Fix: pass `np.arange(len(chunks))` so retrieval returns indices directly.

---

## Resources

### Documentation and articles
- [BM25 explained](https://en.wikipedia.org/wiki/Okapi_BM25) — Okapi BM25 ranking function
- [bm25s library](https://github.com/xhluca/bm25s) — fast BM25 implementation
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — small LLM used for generation
- [Pydantic v2 docs](https://docs.pydantic.dev/) — data validation and aliasing
- [Python Fire](https://github.com/google/python-fire) — CLI generation library
- vLLM repository — target of the RAG system

### AI assistance
AI tools (Claude) were used in the following ways:
- **Architecture discussions:** clarifying SRP, choosing between Manager-style classes and CLI-as-orchestrator, deciding when to extract Reader vs inline logic
- **Debugging:** diagnosing `endswith` ambiguity in path matching, `question_id` vs `question_str` bug in search, file_path normalization between Subject and moulinette
- **Learning concepts:** understanding IoU for character ranges, BM25 internals, Pydantic aliases, Fire's auto-CLI generation
- **Code review:** identifying SRP violations, hardcoded paths, missing validation, mismatched field names against Subject
- **Documentation structure:** organizing this README following Subject's requirements

Code, design decisions, and bug fixes were ultimately authored by the project owner; AI served as a senior-engineer pair-programmer for explanation and review.

---

## Example usage

### Single query
```bash
$ uv run python -m student search "How does vLLM handle continuous batching?" --k 5

🔍 Query: How does vLLM handle continuous batching?
Found 5 results

============================================================
Result #1
============================================================
File: vllm/engine/llm_engine.py
Range: [12048, 14048]
Preview: class LLMEngine:
    """An LLM engine that receives requests and generates texts...
```

### Batch evaluation
```bash
$ uv run python -m student search_dataset \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results \
    --k 10

Loaded 100 questions
Searching: 100%|████████████| 100/100 [00:05<00:00, 18.34it/s]
Saved student_search_results to data/output/search_results/dataset_docs_public.json
```

### Recall metrics
```bash
$ uv run python -m student evaluate \
    --student_results_path data/output/search_results/dataset_docs_public.json \
    --ground_truth_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10

Evaluation Results
========================================
Questions evaluated: 100
Recall@1: 0.590
Recall@3: 0.760
Recall@5: 0.820
Recall@10: 0.850
```

### Answer generation
```bash
$ uv run python -m student answer "What models does vLLM support?" --k 5

🔍 Searching for: What models does vLLM support?
Found 5 chunks

🤖 Generating answer...

💬 Answer:
vLLM supports a wide range of models including Llama, Mistral, Qwen, GPT-2, 
Falcon, and many others. The full list of supported models is maintained in 
the model registry...

📚 Sources:
  - docs/models/supported_models.md
  - vllm/model_executor/models/__init__.py
  ...
```