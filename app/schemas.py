from typing import Literal

from pydantic import BaseModel, Field


class RetrieveReq(BaseModel):
    text: str = Field(..., description="The input text (query)")
    limit: int | None = Field(
        default=5, gt=0, lt=20, description="Number of results to return"
    )
    book_name: (
        Literal[
            "thus_spoke_zarathustra", "genealogy_of_morals", "twilight_of_the_idols"
        ]
        | None
    ) = Field(
        default=None,
        description="The book name to search, if None, no book filter will be applied",
    )


class Point(BaseModel):
    text: str
    book_name: str
    score: float


class RetreiveRes(BaseModel):
    points: list[Point]
