from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from .config import config


def create_collection(qdrant_client: QdrantClient) -> None:
    """Creates the Qdrant collection if it doesn't exist"""
    if qdrant_client.collection_exists(collection_name=config.COLLECTION_NAME):
        return
    else:
        qdrant_client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=config.VECTOR_SIZE, distance=models.Distance.COSINE
                )
            },
            hnsw_config=models.HnswConfigDiff(
                m=config.HNSW_M, ef_construct=config.HNSW_EF_CONSTRUCT
            ),
        )


def read_book(book_path: Path) -> str:
    """Reads a book file and returns the text as a string"""
    with open(book_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def preprocess(text: str) -> str:
    """Preprocesses the text before embedding"""
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())
    return text


def chunk(text: str) -> list[str]:
    """Chunk the text using sentence chunking method"""
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_text(text=text)


def embed(chunks: list[str], book_name: str) -> list[models.PointStruct]:
    """Embeds the chunks and returns Qdrant PointStructs with book metadata"""
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

    points = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Embedding chunk 0/{total_chunks} for {book_name}")
        elif i % 50 == 0 or i == total_chunks - 1:
            print(
                f"Embedding chunk {i}/{total_chunks} for {book_name} ({i / total_chunks * 100:.1f}%)"
            )

        embedding = model.encode(chunk).tolist()

        points.append(
            models.PointStruct(
                id=i,
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
    return points


def upsert_points(
    qdrant_client: QdrantClient, points: list[models.PointStruct]
) -> None:
    """Upserts points to qdrant"""
    qdrant_client.upsert(collection_name=config.COLLECTION_NAME, points=points)


def main() -> None:
    """Orchestrates the functionalities"""
    qdrant_client = QdrantClient(path="./qdrant_data")
    print("Connected to Qdrant")

    try:
        create_collection(qdrant_client)

        for book in config.BOOKS:
            print(f"\n{'=' * 50}")
            print(f"Processing: {book}")
            print(f"{'=' * 50}")

            book_file = book + ".txt"
            data_path = Path(__file__).parent / "data" / book_file

            text = read_book(data_path)
            text = preprocess(text)
            print(f"Read {len(text)} characters")

            chunks = chunk(text)
            print(f"Created {len(chunks)} chunks")

            points = embed(chunks, book)
            upsert_points(qdrant_client, points)
            print(f"Successfully indexed {len(points)} chunks from {book}!")

    except Exception:
        raise
    finally:
        qdrant_client.close()


if __name__ == "__main__":
    main()
