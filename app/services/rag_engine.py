from qdrant_client import QdrantClient, models

from app.config import settings
from app.utils import embed


class Rag:
    def retrieve(
        self,
        client: QdrantClient,
        query: str,
        limit: int,
        book: str | None,
    ) -> list[models.ScoredPoint]:
        embedding = embed(query)
        query_filter = None

        if book:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="book", match=models.MatchValue(value=book)
                    )
                ]
            )

        result = client.query_points(
            collection_name=settings.collection_name,
            query=embedding,
            limit=limit,
            query_filter=query_filter,
        )
        return result.points
