from openai import OpenAI
from qdrant_client import QdrantClient

from app.config import settings
from app.services import Engine

qdrant_client = QdrantClient(path=settings.qdrant_path)
openai_client = OpenAI(base_url=settings.base_url, api_key=settings.openai_api_key)
engine = Engine()


def get_qdrant() -> QdrantClient:
    return qdrant_client


def get_openai() -> OpenAI:
    return openai_client


def get_engine() -> Engine:
    return engine
