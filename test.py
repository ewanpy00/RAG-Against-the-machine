import json
from collections import Counter

with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}")
print(f"Sample chunk fields: {list(chunks[0].keys())}")

# Проверки
ids = [c["chunk_id"] for c in chunks]
print(f"Unique IDs: {len(set(ids)) == len(ids)}")

sizes = [len(c["text"]) for c in chunks]
print(f"Sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")

oversized = [c for c in chunks if len(c["text"]) > 2000]
print(f"Oversized (>2000): {len(oversized)}")

# Проверка indices vs text
mismatches = [
    c for c in chunks
    if len(c["text"]) != (c["last_character_index"] - c["first_character_index"])
]
print(f"Mismatches (indices ≠ text len): {len(mismatches)}")

types = Counter(c["file_type"] for c in chunks)
print(f"By type: {types}")

# Проверка file_path
sample_paths = [c["file_path"] for c in chunks[:5]]
print(f"Sample paths:")
for p in sample_paths:
    print(f"  {p}")