# debug_strings.py
import json

with open("data/output/search_results/dataset_docs_public.json") as f:
    student = json.load(f)

with open("data/datasets/AnsweredQuestions/dataset_docs_public.json") as f:
    truth = json.load(f)

# Возьми первый вопрос
first_question_id = student["search_results"][0]["question_id"]

# Найди тот же вопрос в truth
truth_question = next(
    q for q in truth["rag_questions"]
    if q["question_id"] == first_question_id
)

print("=" * 60)
print(f"Question: {truth_question['question'][:80]}...")
print("=" * 60)

print("\n📝 Ground truth sources:")
for s in truth_question["sources"]:
    print(f"  file_path: '{s['file_path']}'")
    print(f"  range: [{s['first_character_index']}, {s['last_character_index']}]")
    print()

print("📝 Student retrieved sources (top 5):")
for s in student["search_results"][0]["retrieved_sources"][:5]:
    print(f"  file_path: '{s['file_path']}'")
    print(f"  range: [{s['first_character_index']}, {s['last_character_index']}]")
    print()