from qdrant_client import QdrantClient, models

from app.config import settings


def count_chunks(book_name: str, qdrant_client: QdrantClient) -> int:
    count_result = qdrant_client.count(
        collection_name=settings.collection_name,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="book", match=models.MatchValue(value=book_name)
                )
            ]
        ),
    )
    return count_result.count
