from pydantic import BaseModel


class Point(BaseModel):
    text: str
    book: str
    translation: str | None
    score: float


class RetrieveRes(BaseModel):
    points: list[Point]
