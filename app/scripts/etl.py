import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import logging
from pathlib import Path

from app.config import settings
from app.dependencies import get_qdrant
from app.utils.embed import get_embedding_model
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _create_collection(qdrant_client: QdrantClient) -> None:
    if qdrant_client.collection_exists(collection_name=settings.collection_name):
        return
    else:
        qdrant_client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=get_embedding_model().get_embedding_dimension(),
                    distance=models.Distance.COSINE,
                )
            },
            hnsw_config=models.HnswConfigDiff(
                m=settings.hnsw_m, ef_construct=settings.hnsw_ef_construct
            ),
        )


def _create_payload_index(qdrant_client: QdrantClient) -> None:
    qdrant_client.create_payload_index(
        collection_name=settings.collection_name,
        field_name="book",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _is_collection_populated(qdrant_client: QdrantClient) -> bool:
    try:
        return qdrant_client.count(collection_name=settings.collection_name).count > 0
    except Exception:
        return False


def _read_book(book_path: Path) -> str:
    return book_path.read_text(encoding="utf-8")


def _preprocess(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def _chunk(text: str) -> list[str]:
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return splitter.split_text(text=text)


def _embed_batch(
    chunks: list[str], book_name: str, start_id: int
) -> tuple[list[models.PointStruct], int]:
    logger.info(f"Embedding {len(chunks)} chunks for {book_name}...")
    embeddings = get_embedding_model().encode(chunks, show_progress_bar=True)

    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        points.append(
            models.PointStruct(
                id=start_id + i,
                vector={"dense": emb.tolist()},
                payload={
                    "text": chunk,
                    "index": i,
                    "chunk_size": len(chunk),
                    "book": book_name,
                },
            )
        )
    return points, start_id + len(chunks)


def _upsert_points(
    qdrant_client: QdrantClient, points: list[models.PointStruct]
) -> None:
    qdrant_client.upsert(collection_name=settings.collection_name, points=points)


def setup() -> None:
    # Connect to Qdrant
    qdrant_client = get_qdrant()
    logger.info("Connected to Qdrant")

    # Ensure collection and index exists
    _create_collection(qdrant_client)

    # Create payload index
    _create_payload_index(qdrant_client)

    # Check if data already exists
    if _is_collection_populated(qdrant_client):
        logger.info("Collection already contains points. Skipping indexing.")
    else:
        logger.info("Collection empty. Starting indexing...")

        # Process each book
        next_id = 0
        for book in settings.books.keys():
            logger.info(f"Processing: {book}...")

            book_file = book + ".txt"
            data_path = settings.project_root / "data" / book_file

            text = _read_book(data_path)
            logger.info(f"Read {len(text)} characters")

            preprocessed_text = _preprocess(text)
            chunks = _chunk(preprocessed_text)
            logger.info(f"Created {len(chunks)} chunks")

            points, next_id = _embed_batch(chunks, book, next_id)

            logger.info("Indexing points...")
            _upsert_points(qdrant_client, points)
            logger.info(f"Successfully indexed {len(points)} chunks from {book}!")


if __name__ == "__main__":
    setup()
