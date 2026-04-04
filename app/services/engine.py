from openai import OpenAI
from qdrant_client import QdrantClient, models

from app.config import settings
from app.schemas import Point
from app.utils import embed, format_chat, format_points


class Engine:
    def retrieve(
        self, qdrant_client: QdrantClient, query: str, limit: int, books: list[str]
    ) -> list[Point]:
        embedding = embed(query)
        query_filter = None

        if books:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="book", match=models.MatchValue(value=book)
                    )
                    for book in books
                ]
            )

        results = qdrant_client.query_points(
            collection_name=settings.collection_name,
            using="dense",
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
        openai_client: OpenAI,
        qdrant_client: QdrantClient,
        question: str,
        limit: int,
        book: str | None,
    ) -> str:
        retrieved_points = self.retrieve(qdrant_client, question, limit, book)
        formatted_points = format_points(retrieved_points)
        main_message = f"""
        You are Nietzsche himself, speaking from your works.
        Below are excerpts from your writings relevant to the user's question.
        Use them to answer naturally, as if you're recalling your own ideas.
        Do not mention "sources", "according to the text," or any reference to external information.
        Just answer directly and conversationally.
        If the answer is not directly in the relevant passages, exactly respond (nothing else):
        "Hmm, I cannot remember addressing that in my writings"

        Relevant passages:
        {formatted_points}

        User's question:
        {question}

        Answer as Nietzsche, using only the ideas above.
        """
        formatted_main_message = format_chat("user", main_message)

        response = openai_client.chat.completions.create(
            model=settings.llm_model, messages=formatted_main_message
        )
        return response.choices[0].message.content
