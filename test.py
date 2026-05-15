import time
import torch
from student.retrieval.searcher import Searcher
from student.generation.answerer import AnswerGenerator

print("=" * 60)
print("BENCHMARK")
print("=" * 60)

# 1. Загрузка Searcher
t0 = time.time()
searcher = Searcher()
t_searcher_init = time.time() - t0
print(f"\n1. Searcher init: {t_searcher_init:.2f}s")

# 2. Загрузка LLM
t0 = time.time()
generator = AnswerGenerator()
t_llm_init = time.time() - t0
print(f"2. LLM init: {t_llm_init:.2f}s")

# Какое устройство используется?
print(f"\nDevice: {generator.model.device}")
print(f"Dtype: {generator.model.dtype}")
print(f"PyTorch backend: cuda={torch.cuda.is_available()}, mps={torch.backends.mps.is_available()}")

# 3. Поиск (первый раз)
query = "How does vLLM handle continuous batching?"
t0 = time.time()
chunks = searcher.search(query, k=5)
t_search = time.time() - t0
print(f"\n3. Search: {t_search:.3f}s ({len(chunks)} chunks)")

# 4. Подсчёт токенов в context
context_text = "\n\n".join([c.text for c in chunks])
context_tokens = generator.tokenizer(context_text, return_tensors="pt")["input_ids"].shape[1]
print(f"   Context tokens: {context_tokens}")

# 5. Генерация
t0 = time.time()
answer = generator.generate(query, chunks)
t_generate = time.time() - t0
print(f"\n4. Generate (first): {t_generate:.2f}s")
print(f"   Answer length: {len(answer)} chars")

# 6. Вторая генерация (без model loading)
t0 = time.time()
answer2 = generator.generate(query, chunks)
t_generate2 = time.time() - t0
print(f"\n5. Generate (second): {t_generate2:.2f}s")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total first call: {t_searcher_init + t_llm_init + t_search + t_generate:.2f}s")
print(f"Subsequent calls: ~{t_search + t_generate2:.2f}s")
print(f"Generate fraction: {t_generate2 / (t_search + t_generate2) * 100:.1f}%")