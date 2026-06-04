from pydantic import BaseModel, Field


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


class UnansweredQuestion(BaseModel):
    """A question without a ground-truth answer."""

    question_id: str
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth answer and sources."""

    sources: list[MinimalSource] = []
    answer: str


class RagDataset(BaseModel):
    """Dataset of answered questions used for evaluation."""

    rag_questions: list[AnsweredQuestion]


class QuestionDataset(BaseModel):
    """Dataset of unanswered questions."""

    rag_questions: list[UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for a single question."""

    question_id: str
    question_str: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search result enriched with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Full search output produced by the student pipeline."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search output enriched with generated answers."""

    search_results: list[MinimalAnswer]
    k: int
