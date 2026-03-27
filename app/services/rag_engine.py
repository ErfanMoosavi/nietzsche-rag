from openai import OpenAI
from qdrant_client import QdrantClient, models

from app.config import settings
from app.schemas import Point
from app.utils import embed, format_chat, format_points


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

    def generate_response(
        self,
        qdrant_client: QdrantClient,
        openai_client: OpenAI,
        model: str,
        message: str,
        limit: int,
        book: str | None,
    ) -> str:
        retrieved_points = self.retrieve(qdrant_client, message, limit, book)
        formatted_points = format_points(retrieved_points)
        main_message = f"""
            You are a Nietzsche specialist.
            Your goal is to answer user's questions based on the provided sources.
            Here are the sources:
            {formatted_points}
            Here is the user's question:
            {message}
            Based on the sources, answer user's question."""
        formatted_main_message = format_chat("user", main_message)

        response = openai_client.chat.completions.create(
            model=model, messages=formatted_main_message
        )
        return response
