from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
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

app.include_router(router)
