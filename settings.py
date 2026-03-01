"""
settings.py

Purpose:
Central configuration for the Delaware ESG RAG system.
Loads environment variables, provides safe defaults, and keeps code paths consistent
across ingestion and query runtimes.

Author:
Karan Kadam
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_bool(name: str, default: str = "true") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class Settings:
    """
    Strongly-typed settings used across the project.

    This keeps ingestion and query code deterministic and avoids "magic strings"
    scattered across modules.
    """

    openai_api_key: Optional[str]
    embedding_provider: str
    openai_embedding_model: str
    vectorstore_path: str
    data_dir: str
    chunk_size: int
    chunk_overlap: int

    def validate_for_openai(self) -> None:
        """
        Validate required keys for OpenAI provider.

        Raises:
            RuntimeError: If OpenAI is selected but the API key is missing.
        """
        if self.embedding_provider.lower() == "openai" and not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing, but EMBEDDING_PROVIDER is openai")

def load_settings() -> Settings:
    """
    Load settings from environment.

    Returns:
        Settings object.
    """
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),

        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),

        vectorstore_path=os.getenv(
            "VECTORSTORE_PATH", "vectorstore/faiss_index"
        ),
        data_dir=os.getenv("DATA_DIR", "data"),

        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),

        # Langfuse
        langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=os.getenv(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        ),
    )