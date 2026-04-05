from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.services import Engine

qdrant_client = QdrantClient(path=settings.qdrant_path)
openai_client = OpenAI(base_url=settings.base_url, api_key=settings.openai_api_key)
engine = Engine()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)


def get_qdrant() -> QdrantClient:
    return qdrant_client


def get_openai() -> OpenAI:
    return openai_client


def get_engine() -> Engine:
    return engine


def get_embedding_model() -> SentenceTransformer:
    return embedding_model
