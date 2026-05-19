# test_embedding.py
from student.retrieval.embedding_searcher import EmbeddingSearcher

searcher = EmbeddingSearcher()

query = "How to make the system faster?"
results = searcher.search(query, k=5)

for i, chunk in enumerate(results, 1):
    print(f"#{i} {chunk.file_path}")
    print(f"   {chunk.text[:100]}...\n")