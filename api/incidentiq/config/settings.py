"""Typed configuration, loaded once from the environment.

Every tunable in IncidentIQ lives here rather than being read from os.environ
at the point of use. Two reasons: a typo in a variable name fails at startup
instead of at 2am mid-investigation, and the eval harness can construct a
Settings object directly to run a scenario under a different configuration
(different model, failure injection on) without mutating the process env.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://incidentiq:incidentiq_dev_password@localhost:5432/incidentiq"
    )

    # ── LLM provider ────────────────────────────────────────
    llm_provider: Literal["anthropic", "ollama", "stub"] = "ollama"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"

    # ── Embeddings ──────────────────────────────────────────
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    openai_api_key: str = ""

    # ── Observability ───────────────────────────────────────
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3001"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # ── Agent behaviour ─────────────────────────────────────
    max_iterations: int = 8
    tool_max_retries: int = 3

    # ── Failure injection ───────────────────────────────────
    failure_injection_enabled: bool = False
    failure_injection_rate: float = 0.2
    failure_injection_seed: int = 42

    # ── API ─────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @field_validator("failure_injection_rate")
    @classmethod
    def _rate_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"failure_injection_rate must be in [0, 1], got {v}")
        return v

    @field_validator("max_iterations", "tool_max_retries")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be at least 1, got {v}")
        return v

    def require_llm_credentials(self) -> None:
        """Fail loudly at startup rather than on the first agent call."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty. "
                "Set it in .env, or switch to LLM_PROVIDER=ollama for local inference."
            )
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is empty. "
                "Set it in .env, or switch to EMBEDDING_PROVIDER=local."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Import this, not Settings(), so config is read once."""
    return Settings()
