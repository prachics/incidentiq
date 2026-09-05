"""Turning text into vectors.

Behind an interface with two implementations, because the choice of embedding
model is one of the few decisions that is genuinely expensive to reverse: the
stored vectors are model-specific, so switching means re-embedding the whole
corpus. Making the swap a config change rather than a rewrite is worth the small
amount of indirection.

The asymmetry that matters
--------------------------
BGE models are trained with an *instruction prefix* on the query side only.
A passage is embedded as-is; a query is embedded as
"Represent this sentence for searching relevant passages: <query>".

This is easy to miss and costs real recall when missed, because it puts queries
and passages in slightly different regions of the space. The interface therefore
has two methods - `embed_documents` and `embed_query` - rather than one
`embed`, so the asymmetry is impossible to forget at a call site.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

from incidentiq.config import Settings, get_settings

log = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Anything that can turn text into fixed-size vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Must match the vector(N) column width in the schema."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for storage. No instruction prefix."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Applies the model's query prefix if it has one."""


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers running in-process. Free, offline, no API key.

    Normalised vectors: with unit-length vectors, cosine distance and inner
    product agree, so pgvector's `<=>` gives a clean 0..2 range where
    similarity = 1 - distance.
    """

    # Query-side instruction prefixes, per model family.
    _QUERY_PREFIXES = {
        "bge": "Represent this sentence for searching relevant passages: ",
        "e5": "query: ",
    }

    def __init__(self, model_name: str, batch_size: int = 64, use_query_prefix: bool = True):
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._batch_size = batch_size
        self._use_query_prefix = use_query_prefix
        log.info("loading embedding model %s", model_name)
        self._model = SentenceTransformer(model_name)
        # Renamed in sentence-transformers 5.x; fall back for older versions.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        self._dimension = get_dim()

        lowered = model_name.lower()
        self._query_prefix = (
            next((p for key, p in self._QUERY_PREFIXES.items() if key in lowered), "")
            if use_query_prefix else ""
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            f"{self._query_prefix}{text}",
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings. Paid, and no query prefix - the model is trained
    symmetrically, so queries and passages are embedded identically."""

    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, model_name: str, api_key: str):
        import httpx

        self._model_name = model_name
        self._client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        self._dimension = self._DIMENSIONS.get(model_name, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _post(self, payload: list[str]) -> list[list[float]]:
        resp = self._client.post(
            "/embeddings", json={"model": self._model_name, "input": payload}
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # The API does not guarantee input order in the response.
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            out.extend(self._post(texts[i:i + 128]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._post([text])[0]


@lru_cache(maxsize=2)
def _build(provider: str, model: str, api_key: str) -> EmbeddingProvider:
    if provider == "local":
        return LocalEmbeddingProvider(model)
    if provider == "openai":
        return OpenAIEmbeddingProvider(model, api_key)
    raise ValueError(f"unknown embedding provider {provider!r}")


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Cached, so the model is loaded into memory once per process."""
    s = settings or get_settings()
    provider = _build(s.embedding_provider, s.embedding_model, s.openai_api_key)
    if provider.dimension != s.embedding_dim:
        raise RuntimeError(
            f"{provider.model_name} produces {provider.dimension}-dim vectors but "
            f"EMBEDDING_DIM is {s.embedding_dim}. The vector(N) column in "
            f"db/migrations/002_corpora.sql must match, and changing it requires "
            f"re-embedding the corpus."
        )
    return provider
