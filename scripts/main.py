from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "nietzsche_rag"


def create_collection(client: QdrantClient) -> None:
    if client.collection_exists(collection_name=COLLECTION_NAME):
        return
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384, distance=models.Distance.COSINE
            ),
        )


def read_book() -> str:
    data_path = Path(__file__).parent / "data" / "thus_spoke_zarathustra.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def chunk(text: str) -> list[str]:
    chunks = []
    words = text.split()
    chunk_size = 500
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def embed(chunks: list[str]) -> list:
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded")

    points = []
    for i, chunk in enumerate(chunks):
        if i % 50 == 0:
            print(f"  Embedding chunk {i}/{len(chunks)}")

        embedding = model.encode(chunk).tolist()

        points.append(
            models.PointStruct(
                id=i, vector=embedding, payload={"text": chunk, "index": i}
            )
        )
    return points


def upsert(client: QdrantClient, points: list):
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def main() -> None:
    print("Starting indexing process...")

    client = QdrantClient(path="./qdrant_data")
    print("Connected to Qdrant")

    create_collection(client)

    text = read_book()
    print(f"Read book: {len(text)} characters")

    chunks = chunk(text)
    print(f"Created {len(chunks)} chunks")
    print(f"Each chunk has {len(chunks[0])} characters")
    print(chunks[0])

    points = embed(chunks)
    upsert(client, points)

    print(f"Successfully indexed {len(points)} chunks to {COLLECTION_NAME}!")


if __name__ == "__main__":
    main()
