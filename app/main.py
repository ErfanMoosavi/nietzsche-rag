from fastapi import FastAPI

from app.config import settings
from app.routes import books_router, rag_router

app = FastAPI(
    title=settings.title,
    version=settings.version,
    description=settings.description,
    contact=settings.contact,
    license_info=settings.license_info,
)


@app.get("/health", response_model=dict[str, str])
def health() -> dict[str, str]:
    return {"status": "Nietzsche says hello!"}


@app.get("", response_model=dict[str, str])
def home() -> dict[str, str]:
    return {"message": "Welcome to Nietzsche's world!"}


app.include_router(books_router)
app.include_router(rag_router)
