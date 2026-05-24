FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/

ENV PYTHONPATH=/app/src
ENV OMP_NUM_THREADS=4

EXPOSE 8000

ENTRYPOINT ["python", "-m", "dna_sentinel.cli"]
CMD ["serve", "--checkpoint", "artifacts/dna_sentinel/cassiopeia_best.pt", "--port", "8000"]
