from sentence_transformers import SentenceTransformer


def save_model() -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.save_pretrained("all-MiniLM-L6-v2-local")


if __name__ == "__main__":
    save_model()
