from qdrant_client import QdrantClient, models

from app.config import settings
from app.utils import embed


class Rag:
    def retrieve(
        self, client: QdrantClient, query: str, limit: int
    ) -> list[models.ScoredPoint]:
        embedding = embed.embed(query)
        result = client.query_points(
            collection_name=settings.collection_name, query=embedding, limit=limit
        )
        return result.points
