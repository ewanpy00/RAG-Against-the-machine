from pydantic import BaseModel, ConfigDict, Field


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
    model_config = ConfigDict(populate_by_name=True)
    question_id: str
    question_str: str = Field(alias="question")
    answer: str
    sources: list["MinimalSource"] = []

class RagDataset(BaseModel):
    rag_questions: list[Answer]

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class Question(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    question_id: str
    question_str: str = Field(alias="question")

class QuestionDataset(BaseModel):
    rag_questions: list[Question]

class MinimalSearchResults(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    question_id: str
    question_str: str = Field(alias="question")
    retrieved_sources: list[MinimalSource]

class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int
