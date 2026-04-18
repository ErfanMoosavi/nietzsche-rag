from app.config import settings
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def download_model() -> None:
    model_name = settings.embedding_model
    local_model_dir = settings.project_root / "models" / model_name.replace("/", "_")

    if local_model_dir.exists() and any(local_model_dir.iterdir()):
        logger.info(f"Model already exists at {local_model_dir}")
        return

    logger.info(f"Model not found locally. Downloading '{model_name}'...")

    model = SentenceTransformer(model_name)
    local_model_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(local_model_dir))
    logger.info(f"Model saved to {local_model_dir}")


if __name__ == "__main__":
    download_model()
