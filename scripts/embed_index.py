import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from texttools import BatchTheTool

COLLECTION_NAME = "nietzsche_rag"

BOOKS = [
    {
        "filename": "thus_spoke_zarathustra.txt",
        "book_name": "thus_spoke_zarathustra",
    },
    {
        "filename": "genealogy_of_morals.txt",
        "book_name": "genealogy_of_morals",
    },
    {
        "filename": "twilight_of_the_idols.txt",
        "book_name": "twilight_of_the_idols",
    },
]


def create_collection(client: QdrantClient) -> None:
    """Creates the Qdrant collection if it doesn't exist"""
    if client.collection_exists(collection_name=COLLECTION_NAME):
        return
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384, distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(m=32, ef_construct=300),
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
    splitter = SentenceSplitter(chunk_size=180, chunk_overlap=40)
    return splitter.split_text(text=text)


async def extract_keywords(chunks: list[str]) -> list[list[str]]:
    """Extracts keywords using hamtaa-texttools"""
    load_dotenv()
    client = AsyncOpenAI(
        base_url=os.getenv("BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
    )
    tool = BatchTheTool(client=client, model=os.getenv("MODEL"), max_concurrency=10)
    results = await tool.extract_keywords(
        texts=chunks, mode="count", number_of_keywords=5, output_lang="English"
    )
    outputs = []
    for res in results:
        outputs.append(res.result)
    return outputs


def embed(
    chunks: list[str], book_name: str, keywords: list[list[str]]
) -> list[models.PointStruct]:
    """Embeds the chunks and returns Qdrant PointStructs with book metadata"""
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

    points = []
    total_chunks = len(chunks)

    for i, (chunk, keyword_list) in enumerate(zip(chunks, keywords)):
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
                vector=embedding,
                payload={
                    "text": chunk,
                    "index": i,
                    "chunk_size": len(chunk),
                    "book": book_name,
                    "keywords": keyword_list,
                },
            )
        )
    return points


def upsert(client: QdrantClient, points: list) -> None:
    """Upserts points to the Qdrant collection"""
    client.upsert(collection_name=COLLECTION_NAME, points=points)


async def main() -> None:
    """Main function - indexes all books defined in BOOKS list"""
    client = QdrantClient(path="./qdrant_data")
    print("Connected to Qdrant")

    try:
        create_collection(client)

        for book in BOOKS:
            print(f"\n{'=' * 50}")
            print(f"Processing: {book['book_name']}")
            print(f"{'=' * 50}")

            data_path = Path(__file__).parent / "data" / book["filename"]

            text = read_book(data_path)
            text = preprocess(text)
            print(f"Read {len(text)} characters")

            chunks = chunk(text)
            keywords = await extract_keywords(chunks)
            print(f"Created {len(chunks)} chunks")

            points = embed(chunks, book["book_name"], keywords)

            upsert(client, points)
            print(
                f"Successfully indexed {len(points)} chunks from {book['book_name']}!"
            )

    except Exception:
        raise
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
