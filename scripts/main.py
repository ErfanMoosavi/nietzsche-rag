from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "nietzsche_rag"


def create_collection(client: QdrantClient) -> None:
    """Creates the Qdrant collection"""
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
    """Reads the book and returns the text as a string"""
    data_path = Path(__file__).parent / "data" / "thus_spoke_zarathustra.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def chunk(text: str) -> list[str]:
    """Chunk the text using sentence chunking method"""
    splitter = SentenceSplitter(chunk_size=128, chunk_overlap=30)
    return splitter.split_text(text=text)


def embed(chunks: list[str]) -> list[models.PointStruct]:
    """Embeds the chunks using sentence-transformers and returns the Qdrant PointStructs"""
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    points = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Embedding chunk 0/{len(chunks)}")
        elif i % 50 == 0 or i == len(chunks) - 1:
            print(f"Embedding chunk {i}/{len(chunks)} ({i / len(chunks) * 100:.1f}%)")

        embedding = model.encode(chunk).tolist()

        points.append(
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={"text": chunk, "index": i, "chunk_size": len(chunk)},
            )
        )
    return points


def upsert(client: QdrantClient, points: list) -> None:
    """Upserts points to the Qdrant collection"""
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def main() -> None:
    """Main function"""
    client = QdrantClient(path="./qdrant_data")
    print("Connected to Qdrant")

    try:
        create_collection(client)

        text = read_book()
        print(f"Read book: {len(text)} characters")

        chunks = chunk(text)
        print(f"Created {len(chunks)} chunks")

        points = embed(chunks)
        upsert(client, points)
        print(f"Successfully indexed {len(points)} chunks to {COLLECTION_NAME}!")

    finally:
        client.close()


if __name__ == "__main__":
    main()
