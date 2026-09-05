# Build context is the repository root (see docker-compose.yml), because
# pyproject.toml lives there and the package lives under api/. Building from
# ./api would leave the dependency manifest outside the context.
FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY api/incidentiq ./api/incidentiq
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /install /usr/local
COPY api/incidentiq ./incidentiq

# Cached in the model_cache volume so the embedding model is downloaded once.
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "incidentiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
