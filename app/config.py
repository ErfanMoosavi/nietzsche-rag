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

    # Qdrant settings
    qdrant_path: str = "./qdrant_data"
    collection_name: str = "nietzsche_rag"

    # OpenAI settings
    # base_url: str
    # api_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
