from fastapi import FastAPI

from app.config import settings
from app.routes import router

app = FastAPI(
    title=settings.title,
    version=settings.version,
    description=settings.description,
    contact=settings.contact,
    license_info=settings.license_info,
)

app.include_router(router)
