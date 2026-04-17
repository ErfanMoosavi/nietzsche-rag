FROM docker.arvancloud.ir/python:3.13-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install .

COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]