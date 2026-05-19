# benchmark_cache.py
import time
from src.student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")# benchmark_cache.py
import time
from student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")# benchmark_cache.py
import time
from student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")# benchmark_cache.py
import time
from student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")# benchmark_cache.py
import time
from student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")# benchmark_cache.py
import time
from student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")# benchmark_cache.py
import time
from student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")# benchmark_cache.py
import time
from student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")# benchmark_cache.py
import time
from student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")# benchmark_cache.py
import time
from student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")# benchmark_cache.py
import time
from student.retrieval.embedding_searcher import EmbeddingSearcher

# Одна и та же query 3 раза
queries = ["How to configure server?"] * 3

# Test 1: cache disabled
print("=" * 60)
print("WITHOUT CACHE")
print("=" * 60)

searcher_nocache = EmbeddingSearcher(use_cache=False)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_nocache.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")

# Test 2: cache enabled
print("\n" + "=" * 60)
print("WITH CACHE")
print("=" * 60)

searcher_cached = EmbeddingSearcher(use_cache=True)

for i, q in enumerate(queries, 1):
    t0 = time.time()
    searcher_cached.search(q, k=5)
    t = time.time() - t0
    print(f"Query {i}: {t*1000:6.1f}ms")