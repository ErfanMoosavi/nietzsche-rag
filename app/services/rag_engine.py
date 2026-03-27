from qdrant_client import QdrantClient, models

from app.config import settings
from app.schemas import Point
from app.utils import embed


class Rag:
    def retrieve(
        self, qdrant_client: QdrantClient, query: str, limit: int, book: str | None
    ) -> list[Point]:
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

        results = qdrant_client.query_points(
            collection_name=settings.collection_name,
            query=embedding,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        points: list[Point] = []
        for point in results.points:
            points.append(
                Point(
                    text=point.payload["text"],
                    book=point.payload["book"],
                    score=point.score,
                )
            )
        return points

    def generate():
        pass
