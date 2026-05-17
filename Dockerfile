# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd -r app && useradd -r -g app -m -d /app app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[api]"

COPY config/ ./config/

USER app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "dna_sentinel.api:app", "--host", "0.0.0.0", "--port", "8000"]
