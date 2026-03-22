from qdrant_client import QdrantClient, models

from app.config import settings
from app.schemas import Point
from app.utils import embed, translate


class Rag:
    def retrieve(
        self,
        client: QdrantClient,
        query: str,
        limit: int,
        book: str | None,
        language: str | None,
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

        results = client.query_points(
            collection_name=settings.collection_name,
            query=embedding,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        points: list[Point] = []
        for point in results.points:
            translation = None
            if language:
                translation = translate(point.payload["text"], target_lang=language)
            points.append(
                Point(
                    text=point.payload["text"],
                    book=point.payload["book"],
                    translation=translation,
                    score=point.score,
                )
            )
        return points
