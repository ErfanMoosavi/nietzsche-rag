from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.schemas import BooksRes, BookInfoRes

router = APIRouter(prefix="books", tags=["Books"])


@router.get("/", response_model=BooksRes)
def list_books() -> BooksRes:
    try:
        return BooksRes(books=settings.books)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("{book_name}", response_model=BookInfoRes)
def book_info(book_name: str) -> BookInfoRes:
    pass
