"""Pydantic models for FastAPI request bodies."""

from typing import Any

from pydantic import BaseModel


class ManualQaRequest(BaseModel):
    target_dataset: str
    question: str
    expected_answer: str


class BulkQaRequest(BaseModel):
    target_dataset: str
    rows: list[dict[str, Any]]


class GenerateQaRequest(BaseModel):
    source_text: str
    count: int = 5
    language: str = "zh"


class GenerateAnswerRequest(BaseModel):
    question: str
    language: str = "zh"


class RegenerateAnswerRequest(BaseModel):
    question: str
    current_answer: str
    passages: list[dict[str, Any]]
    language: str = "zh"
