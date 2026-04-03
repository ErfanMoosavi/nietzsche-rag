from pydantic import BaseModel, Field


class Point(BaseModel):
    text: str = Field(..., description="The input text")
    book: str = Field(
        ...,
        description="The book name to search, if None, no book filter will be applied",
    )
    score: float = Field(..., description="The cosine similarity score")


class RetrieveRes(BaseModel):
    points: list[Point] = Field(..., description="The retrieved points")


class RagRes(BaseModel):
    answer: str = Field(..., description="The answer of the question")


class BooksListRes(BaseModel):
    books: list[str] = Field(..., description="List of available books")


class BookInfoRes(BaseModel):
    title: str = Field(..., description="Full title in English")
    original_title: str = Field(..., description="German title")
    year: int = Field(..., description="Publication year")
    chunk_count: int = Field(..., description="Number of chunks in Qdrant")
    summary: str = Field(..., description="One-sentence description")
