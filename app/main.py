from fastapi import FastAPI

from .config import settings
from .routes import router

app = FastAPI(
    title=settings.title,
    version=settings.version,
    description=settings.description,
    contact=settings.contact,
    license_info=settings.license_info,
)

app.include_router(router)
