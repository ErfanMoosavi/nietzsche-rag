from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.dependencies import get_qdrant
from app.setup.setup_config import setup_config


def _is_model_present() -> bool:
    return (Path(__file__).parent / "all-MiniLM-L6-v2-local").exists()


def _is_collection_populated(qdrant_client: QdrantClient) -> bool:
    try:
        count = qdrant_client.count(collection_name=setup_config.collection_name)
        return count.count > 0
    except Exception:
        return False


def _create_collection(qdrant_client: QdrantClient) -> None:
    """Creates the Qdrant collection if it doesn't exist"""
    if qdrant_client.collection_exists(collection_name=setup_config.collection_name):
        return
    else:
        qdrant_client.create_collection(
            collection_name=setup_config.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=setup_config.vector_size, distance=models.Distance.COSINE
                )
            },
            hnsw_config=models.HnswConfigDiff(
                m=setup_config.hnsw_m, ef_construct=setup_config.hnsw_ef_construct
            ),
        )


def _create_payload_index(qdrant_client: QdrantClient) -> None:
    """Indexed the payload for more efficient search"""
    qdrant_client.create_payload_index(
        collection_name=setup_config.collection_name, field_name="book"
    )


def _read_book(book_path: Path) -> str:
    """Reads a book file and returns the text as a string"""
    with open(book_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def _preprocess(text: str) -> str:
    """Preprocesses the text before embedding"""
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())
    return text


def _chunk(text: str) -> list[str]:
    """Chunk the text using sentence chunking method"""
    splitter = SentenceSplitter(
        chunk_size=setup_config.chunk_size, chunk_overlap=setup_config.chunk_overlap
    )
    return splitter.split_text(text=text)


def _save_model() -> None:
    """Saves the embedding model for offline usage"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.save_pretrained("all-MiniLM-L6-v2-local")


def _embed(
    model: SentenceTransformer, chunks: list[str], book_name: str, start_id: int
) -> tuple[list[models.PointStruct], int]:
    """Embeds the chunks and returns Qdrant PointStructs with book metadata"""
    print("Loading embedding model...")

    points = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        point_id = start_id + i

        if i == 0:
            print(f"Embedding chunk 0/{total_chunks} for {book_name}")
        elif i % 50 == 0 or i == total_chunks - 1:
            print(
                f"Embedding chunk {i}/{total_chunks} for {book_name} ({i / total_chunks * 100:.1f}%)"
            )

        embedding = model.encode(chunk).tolist()

        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": embedding,
                },
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
    """Upserts points to qdrant"""
    qdrant_client.upsert(collection_name=setup_config.collection_name, points=points)


def setup() -> None:
    # Connect to Qdrant (local file mode)
    qdrant_client = get_qdrant()
    print("Connected to Qdrant")

    # Ensure collection and index exist (idempotent)
    _create_collection(qdrant_client)
    _create_payload_index(qdrant_client)

    # Check if data already exists
    if _is_collection_populated(qdrant_client):
        print("Collection already contains points. Skipping indexing.")
    else:
        print("Collection empty. Starting indexing...")

        # Save model only if missing
        if not _is_model_present():
            _save_model()
        else:
            print("Model already saved locally. Skipping download.")

        # Load model
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

        # Process each book
        next_id = 0
        for book in settings.books.keys():
            print(f"Processing: {book}")

            book_file = book + ".txt"
            data_path = Path(__file__).parent / "data" / book_file

            text = _read_book(data_path)
            print(f"Read {len(text)} characters")

            preprocessed_text = _preprocess(text)
            chunks = _chunk(preprocessed_text)
            print(f"Created {len(chunks)} chunks")

            points, next_id = _embed(model, chunks, book, next_id)
            _upsert_points(qdrant_client, points)
            print(f"Successfully indexed {len(points)} chunks from {book}!")
