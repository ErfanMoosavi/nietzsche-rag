from pydantic import BaseModel


class RetrieveReq(BaseModel):
    text: str
    limit: int | None = None


class RagReq(BaseModel):
    text: str


class Point(BaseModel):
    text: str
    score: float


class RetreiveRes(BaseModel):
    points: list[Point]
