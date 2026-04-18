from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App settings
    title: str = "Nietzsche-Rag"
    version: str = "0.1.0"
    description: str = "A rag system based on Nietzsche's philosophy!"
    contact: dict[str, str] = {
        "name": "Erfan Moosavi",
        "email": "erfanmoosavi84@gmail.com",
    }
    license_info: dict[str, str] = {"name": "MIT"}

    # Project root
    project_root: Path = Path(__file__).parent.parent

    # OpenAI settings
    openai_api_key: str
    base_url: str
    llm_model: str

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "nietzsche_rag"
    embedding_model: str = "all-MiniLM-L6-v2"
    hnsw_m: int = 64
    hnsw_ef_construct: int = 400
    hnsw_ef: int = 128
    chunk_size: int = 160
    chunk_overlap: int = 35

    # Books
    books: dict[str, dict[str, Any]] = {
        "thus_spoke_zarathustra": {
            "title": "Thus Spoke Zarathustra",
            "original_title": "Also sprach Zarathustra",
            "year": 1885,
            "summary": "A philosophical novel where Zarathustra proclaims the Übermensch, the death of God, and eternal recurrence.",
        },
        "genealogy_of_morals": {
            "title": "On the Genealogy of Morals",
            "original_title": "Zur Genealogie der Moral",
            "year": 1887,
            "summary": "A polemical work tracing the origin of moral concepts like guilt, bad conscience, and ascetic ideals.",
        },
        "twilight_of_the_idols": {
            "title": "Twilight of the Idols",
            "original_title": "Götzen-Dämmerung",
            "year": 1889,
            "summary": "A short, aphoristic attack on German philosophy, Christianity, and Socrates, subtitled 'How to Philosophize with a Hammer'.",
        },
        "beyond_good_and_evil": {
            "title": "Beyond Good and Evil",
            "original_title": "Jenseits von Gut und Böse",
            "year": 1886,
            "summary": "A critique of traditional philosophy and exploration of morality beyond simplistic binaries.",
        },
        "gay_science": {
            "title": "The Gay Science",
            "original_title": "Die fröhliche Wissenschaft",
            "year": 1882,
            "summary": "A joyful affirmation of life, introducing the death of God and eternal recurrence.",
        },
        "ecce_homo": {
            "title": "Ecce Homo",
            "original_title": "Ecce Homo",
            "year": 1888,
            "summary": "Nietzsche's autobiographical self-reflection on his own life and works.",
        },
        "birth_of_tragedy": {
            "title": "The Birth of Tragedy",
            "original_title": "Die Geburt der Tragödie",
            "year": 1872,
            "summary": "An exploration of Greek tragedy through the Apollonian and Dionysian duality.",
        },
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
