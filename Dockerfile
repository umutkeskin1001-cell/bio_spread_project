FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app
WORKDIR /app
COPY pyproject.toml src/ config/ artifacts/cassiopeia_prime_v14/ ./artifacts/cassiopeia_prime_v14/
RUN pip install --no-cache-dir ".[api]"
ENV PYTHONPATH=/app/src OMP_NUM_THREADS=4
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; assert urllib.request.urlopen('http://localhost:8000/health').status == 200"
ENTRYPOINT ["python", "-m", "dna_sentinel.cli"]
CMD ["serve", "--checkpoint", "artifacts/cassiopeia_prime_v14/cassiopeia_best.pt", "--port", "8000"]
