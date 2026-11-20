
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The question or search query",
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve",
    )

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question about uploaded documents",
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve",
    )


class AgentRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Message or task for the research agent",
    )
