from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routes import books_router, rag_router
from app.setup import setup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup()

    yield

    # Shutdown


app = FastAPI(
    title=settings.title,
    version=settings.version,
    description=settings.description,
    contact=settings.contact,
    license_info=settings.license_info,
    lifespan=lifespan,
)


@app.get("/", response_model=dict[str, str])
def home() -> dict[str, str]:
    return {"message": "Welcome to Nietzsche-Rag!"}


app.include_router(books_router)
app.include_router(rag_router)
