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

    # Available books
    books: list[str] = [
        "thus_spoke_zarathustra",
        "genealogy_of_morals",
        "twilight_of_the_idols",
    ]

    # OpenAI settings
    openai_api_key: str
    base_url: str
    llm_model: str

    # Qdrant settings
    qdrant_path: str = "./qdrant_data"
    collection_name: str = "nietzsche_rag"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
