from sentence_transformers import SentenceTransformer


def embed(text: str) -> list[float]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(text).tolist()
    return embedding
