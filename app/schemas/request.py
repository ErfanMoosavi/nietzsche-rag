from typing import Literal

from pydantic import BaseModel, Field

BookName = Literal[
    "thus_spoke_zarathustra",
    "genealogy_of_morals",
    "twilight_of_the_idols",
    "beyond_good_and_evil",
    "gay_science",
    "ecce_homo",
    "birth_of_tragedy",
]


class RetrieveReq(BaseModel):
    text: str = Field(..., example="Punishment", description="The input text")
    limit: int = Field(
        default=5, gt=0, lte=20, description="Number of results to return"
    )
    book: list[BookName] = Field(
        default=[], description="List of books to search. Empty list means all books."
    )


class RagReq(BaseModel):
    question: str = Field(
        ..., example="Who is Dionysus?", description="The input question"
    )
    retrieval_limit: int = Field(
        default=3,
        gt=0,
        lte=20,
        description="Number of results to return for RAG pipeline",
    )
    based_on: list[BookName] = Field(
        default=[], description="List of books to search. Empty list means all books."
    )
