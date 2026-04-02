from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client import QdrantClient

from app.config import settings
from app.dependencies import get_qdrant
from app.schemas import BookInfoRes, BooksRes
from app.utils import count_chunks

router = APIRouter(prefix="/books", tags=["Books"])


@router.get("/", response_model=BooksRes)
def list_books() -> BooksRes:
    try:
        return BooksRes(books=settings.books.keys())

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/{book_name}", response_model=BookInfoRes)
def book_info(
    book_name: str, qdrant_client: QdrantClient = Depends(get_qdrant)
) -> BookInfoRes:
    try:
        if book_name not in settings.books:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Book '{book_name}' not found. Available: {list(settings.books.keys())}",
            )

        book_data = settings.books[book_name]

        return BookInfoRes(
            title=book_data["title"],
            original_title=book_data["original_title"],
            year=book_data["year"],
            chunk_count=count_chunks(book_name, qdrant_client),
            summary=book_data["summary"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
