from pydantic import BaseModel


class Document(BaseModel):
    id: str
    text: str
    metadata: dict = {}

class FileRecord(BaseModel):
    filepath: str
    content: str
    filetype: str

class Chunk(BaseModel):
    chunk_id: str
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    file_type: str

class Answer(BaseModel):
    question_id: str
    question: str
    answer: str
    sources: list["MinimalSource"] = []

class RagDataset(BaseModel):
    rag_questions: list[Answer]

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]

class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int
