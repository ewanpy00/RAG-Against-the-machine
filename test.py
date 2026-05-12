from src.student.retrieval.searcher import Searcher
from src.student.generation.answerer import AnswerGenerator

# Initialize (медленно — первый раз скачивает модель)
searcher = Searcher()
generator = AnswerGenerator()

# Test
query = "How to configure OpenAI server in vLLM?"
chunks = searcher.search(query, k=5)
answer = generator.generate(query, chunks)

print(f"Q: {query}")
print(f"A: {answer}")