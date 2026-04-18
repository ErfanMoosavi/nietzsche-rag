import uvicorn
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


@app.get("", response_model=dict[str, str])
def home() -> dict[str, str]:
    return {"message": "Welcome to Nietzsche's world!"}


app.include_router(books_router)
app.include_router(rag_router)

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000)
