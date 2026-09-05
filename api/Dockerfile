# Build context is the repository root (see docker-compose.yml): pyproject.toml
# lives there and the package lives under api/.
FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only PyTorch build BEFORE the project, so pip already has a
# satisfying torch when sentence-transformers asks for one. The default wheel
# carries CUDA libraries that are dead weight in a CPU container and took the
# image to 9.7GB on their own.
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml ./
COPY api/incidentiq ./api/incidentiq
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /install /usr/local
COPY api/incidentiq ./incidentiq

# Cached in the model_cache volume so the embedding model downloads once.
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "incidentiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
