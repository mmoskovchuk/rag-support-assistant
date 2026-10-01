"""Centralised, validated configuration loaded from environment variables.

Every tunable value lives here, so the rest of the code never reads
os.environ directly. Pydantic validates types at startup: a typo in .env
fails fast instead of surfacing as a strange error hours later.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Service ---
    log_level: str = "INFO"
    api_key: SecretStr  # shared secret expected in the X-API-Key header

    # --- File lifecycle ---
    data_dir: Path = Path("/data")
    max_file_size_mb: int = 50
    file_stable_seconds: float = 2.0  # size must stay unchanged this long
    file_stable_timeout: float = 120.0  # give up waiting for a slow copy

    # --- Text extraction ---
    # A PDF page with fewer characters in its text layer is treated as a scan and OCR'd.
    pdf_min_text_chars: int = 30

    # --- OCR (Tesseract) ---
    ocr_languages: str = "ukr+eng"
    ocr_dpi: int = 300
    tesseract_config: str = "--oem 1 --psm 3"
    ocr_page_timeout: int = 120  # seconds per page

    # --- Vector store (ChromaDB) ---
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "company_docs"

    # --- Embeddings (Hugging Face) ---
    # Chosen by measurement on the demo set, see docs/experiments.md.
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    # Some models (the e5 family) are trained with role prefixes and work noticeably worse without them.
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 80

    # --- Retrieval ---
    top_k: int = 4
    # Cosine similarity in [0, 1]. Scales differ between models: re-tune it after changing EMBEDDING_MODEL.
    relevance_threshold: float = 0.45

    # --- Topic suggestions when there is no answer ---
    topic_max_chars: int = 4000  # how much of a document the LLM reads to name its topic
    suggestion_pool_k: int = 20  # chunks searched to collect distinct topics
    max_suggested_topics: int = 5

    # --- LLM (OpenAI) ---
    openai_api_key: SecretStr
    openai_model: str = "gpt-4.1-mini"
    openai_temperature: float = 0.0
    openai_timeout: float = 60.0
    openai_max_retries: int = 3

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @property
    def processing_dir(self) -> Path:
        return self.data_dir / "processing"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def failed_dir(self) -> Path:
        return self.data_dir / "failed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
