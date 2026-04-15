from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model, local_files_only=True)


def embed(text: str) -> list[float]:
    model = get_embedding_model()
    result = model.encode(text, normalize_embeddings=True)
    return result.tolist()
