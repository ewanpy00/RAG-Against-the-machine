from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """Raw document loaded from disk."""

    id: str
    text: str
    metadata: dict = {}


class FileRecord(BaseModel):
    """A single file read from the repository."""

    filepath: str
    content: str
    filetype: str


class Chunk(BaseModel):
    """A text chunk extracted from a FileRecord."""

    chunk_id: str
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    file_type: str


class MinimalSource(BaseModel):
    """Source location referenced by a question or search result."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Answer(BaseModel):
    """A question with its ground-truth answer and sources."""

    model_config = ConfigDict(populate_by_name=True)

    question_id: str
    question_str: str = Field(alias="question")
    answer: str
    sources: list[MinimalSource] = []


class RagDataset(BaseModel):
    """Dataset of answered questions used for evaluation."""

    rag_questions: list[Answer]


class Question(BaseModel):
    """A question without a ground-truth answer."""

    model_config = ConfigDict(populate_by_name=True)

    question_id: str
    question_str: str = Field(alias="question")


class QuestionDataset(BaseModel):
    """Dataset of unanswered questions."""

    rag_questions: list[Question]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for a single question."""

    model_config = ConfigDict(populate_by_name=True)

    question_id: str
    question_str: str = Field(alias="question")
    retrieved_sources: list[MinimalSource]


class StudentSearchResults(BaseModel):
    """Full search output produced by the student pipeline."""

    search_results: list[MinimalSearchResults]
    k: int
