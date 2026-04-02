from typing import Literal

from pydantic import BaseModel, Field


class Point(BaseModel):
    text: str = Field(..., description="The input text")
    book: Literal[
        "thus_spoke_zarathustra", "genealogy_of_morals", "twilight_of_the_idols"
    ] = Field(
        ...,
        description="The book name to search, if None, no book filter will be applied",
    )
    score: float = Field(..., description="The cosine similarity score")


class RetrieveRes(BaseModel):
    points: list[Point] = Field(..., description="The retrieved points")


class RagRes(BaseModel):
    answer: str = Field(..., description="The answer of the question")


class BooksRes(BaseModel):
    books: list[str] = Field(..., description="List of available books")


class BookInfoRes(BaseModel):
    pass
