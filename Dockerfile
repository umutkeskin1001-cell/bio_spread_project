FROM python:3.11-slim

WORKDIR /app

# Install production dependencies
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# Install the package
COPY src/ /app/src/
COPY pyproject.toml .
RUN pip install -e . --no-deps

# Copy config and feature data
COPY config/ /app/config/
COPY data/sovereign_features/ /app/data/sovereign_features/

# Environment variables (can be overridden at runtime)
ENV MODEL_TYPE=sovereign-x-pro
ENV CONFIG_PATH=config/default.yaml
ENV FEATURE_DIR=data/sovereign_features
ENV ARTIFACTS_DIR=artifacts

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
