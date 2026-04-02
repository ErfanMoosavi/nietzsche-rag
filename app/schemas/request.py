from typing import Literal

from pydantic import BaseModel, Field


class RetrieveReq(BaseModel):
    text: str = Field(..., example="Punishment", description="The input text")
    limit: int = Field(
        default=5, gt=0, lt=20, description="Number of results to return"
    )
    book: Literal[
        "all", "thus_spoke_zarathustra", "genealogy_of_morals", "twilight_of_the_idols"
    ] = Field(default="all", description="Restrict retrieval to a specific book.")


class RagReq(BaseModel):
    question: str = Field(
        ..., example="Who is Dionysus?", description="The input question"
    )
    retrieval_limit: int = Field(
        default=3,
        gt=0,
        lt=20,
        description="Number of results to return for RAG pipeline",
    )
    based_on: Literal[
        "all", "thus_spoke_zarathustra", "genealogy_of_morals", "twilight_of_the_idols"
    ] = Field(default="all", description="Restrict retrieval to a specific book.")
