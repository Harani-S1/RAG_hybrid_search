from pydantic import BaseModel
from typing import List, Optional, Any


class QuestionRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source: str
    page: Optional[Any] = None
    human_page: Optional[Any] = None
    reranker_score: Optional[float] = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation] = []
    sources: List[str] = []
    confidence: Optional[float] = None
    retrieved_chunk_ids: List[str] = []