from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from .config import setup_config


class Setup:
    def __init__(self) -> None:
        self.is_setup = False
        if Path.exists(Path.parent / "all-MiniLM-L6-v2-local") and Path.exists(
            Path.parent / "qdrant_data"
        ):
            self.is_setup = True

    def _create_collection(self, qdrant_client: QdrantClient) -> None:
        """Creates the Qdrant collection if it doesn't exist"""
        if qdrant_client.collection_exists(
            collection_name=setup_config.COLLECTION_NAME
        ):
            return
        else:
            qdrant_client.create_collection(
                collection_name=setup_config.COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(
                        size=setup_config.VECTOR_SIZE, distance=models.Distance.COSINE
                    )
                },
                hnsw_config=models.HnswConfigDiff(
                    m=setup_config.HNSW_M, ef_construct=setup_config.HNSW_EF_CONSTRUCT
                ),
            )

    def _create_payload_index(self, qdrant_client: QdrantClient) -> None:
        """Indexed the payload for more efficient search"""
        qdrant_client.create_payload_index(
            collection_name=setup_config.COLLECTION_NAME, field_name="book"
        )

    def _read_book(self, book_path: Path) -> str:
        """Reads a book file and returns the text as a string"""
        with open(book_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    def _preprocess(self, text: str) -> str:
        """Preprocesses the text before embedding"""
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = " ".join(text.split())
        return text

    def _chunk(self, text: str) -> list[str]:
        """Chunk the text using sentence chunking method"""
        splitter = SentenceSplitter(
            chunk_size=setup_config.CHUNK_SIZE, chunk_overlap=setup_config.CHUNK_OVERLAP
        )
        return splitter.split_text(text=text)

    def _save_model(self) -> None:
        """Saves the embedding model for offline usage"""
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.save_pretrained("all-MiniLM-L6-v2-local")

    def _embed(
        self, model: SentenceTransformer, chunks: list[str], book_name: str
    ) -> list[models.PointStruct]:
        """Embeds the chunks and returns Qdrant PointStructs with book metadata"""
        print("Loading embedding model...")

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

    def _upsert_points(
        self, qdrant_client: QdrantClient, points: list[models.PointStruct]
    ) -> None:
        """Upserts points to qdrant"""
        qdrant_client.upsert(
            collection_name=setup_config.COLLECTION_NAME, points=points
        )

    def setup(self) -> None:
        """Sets up the app"""
        if not self.is_setup:
            # Connect to Qdrant
            qdrant_client = QdrantClient(path="./qdrant_data")
            print("Connected to Qdrant")

            try:
                # Create collection
                self._create_collection(qdrant_client)

                # Create payload index
                self._create_payload_index(qdrant_client)

                # Save the model
                self._save_model()

                # Load the model
                model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

                for book in setup_config.BOOKS:
                    print(f"Processing: {book}")

                    book_file = book + ".txt"
                    data_path = Path(__file__).parent / "data" / book_file

                    # Read the book
                    text = self._read_book(data_path)
                    print(f"Read {len(text)} characters")

                    # Preprocess the book
                    preprocessed_text = self._preprocess(text)

                    # Chunk
                    chunks = self._chunk(preprocessed_text)
                    print(f"Created {len(chunks)} chunks")

                    # Embed
                    points = self._embed(model, chunks, book)

                    # Upsert points
                    self._upsert_points(qdrant_client, points)
                    print(f"Successfully indexed {len(points)} chunks from {book}!")

            except Exception:
                raise
            finally:
                qdrant_client.close()

        else:
            print("Already set up!")
