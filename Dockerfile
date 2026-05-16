# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for compiling native extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY requirements_inference.txt .
COPY pyproject.toml .

# Install dependencies and freeze exact versions for reproducible builds
RUN pip install --no-cache-dir -r requirements_inference.txt && \
    pip freeze > /pinned-requirements.txt

# -------------------------------------------------------------------
# Runtime stage
# -------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Create non-root user
RUN groupadd -r bio-spread && useradd -r -g bio-spread -m -d /app bio-spread

WORKDIR /app

# Copy pinned dependency list from builder and install exact versions
COPY --from=builder /pinned-requirements.txt .
RUN pip install --no-cache-dir -r pinned-requirements.txt

# Copy package source and install (production: use editable install)
COPY pyproject.toml .
COPY src/ /app/src/
RUN pip install . --no-deps

# Copy runtime config
COPY config/ /app/config/

# Feature data is expected via volume mount at runtime
VOLUME /app/data/features

# Environment variables (can be overridden at runtime)
ENV MODEL_TYPE=bio-spread
ENV CONFIG_PATH=config/default.yaml
ENV FEATURE_DIR=data/features
ENV ARTIFACTS_DIR=artifacts

EXPOSE 8000

# Healthcheck using Python's stdlib (curl not available in slim image)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Drop privileges
USER bio-spread

WORKDIR /app
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
