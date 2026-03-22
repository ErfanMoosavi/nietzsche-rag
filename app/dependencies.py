from qdrant_client import QdrantClient

from app.config import settings
from app.services import Rag

client = QdrantClient(path=settings.qdrant_path)
rag = Rag()


def get_qdrant() -> QdrantClient:
    return client


def get_rag() -> Rag:
    return rag
