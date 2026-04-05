from app.dependencies import get_embedding_model


def embed(text: str) -> list[float]:
    model = get_embedding_model()
    return model.encode(text).tolist()
