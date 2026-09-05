# Multi-stage so the runtime image does not carry build tooling.
FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY incidentiq ./incidentiq
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /install /usr/local
COPY incidentiq ./incidentiq

# Pre-download the embedding model at build time so the first request is not
# slowed by a model download. Cached in the model_cache volume thereafter.
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "incidentiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
