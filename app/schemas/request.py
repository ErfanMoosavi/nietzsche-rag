from typing import Literal

from pydantic import BaseModel, Field


class RetrieveReq(BaseModel):
    text: str = Field(..., example="Punishment", description="The input text")
    limit: int = Field(
        default=5, gt=0, lt=20, description="Number of results to return"
    )
    book: (
        Literal[
            "thus_spoke_zarathustra", "genealogy_of_morals", "twilight_of_the_idols"
        ]
        | None
    ) = Field(
        default=None,
        description="The book name to search, if None, no book filter will be applied",
    )
