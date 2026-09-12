"""
Hệ thống cấu hình trung tâm cho AI Worker (FastAPI, OCR, RAG).
Hỗ trợ cả Pydantic Settings và Fallback Python chuẩn (để chạy thử nghiệm ngay lập tức).
"""

from pathlib import Path
import os

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class Settings(BaseSettings):
        APP_NAME: str = "OCR RAG AI Worker"
        APP_VERSION: str = "1.0.0"
        DEBUG: bool = True

        BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
        STORAGE_DIR: Path = BASE_DIR / "storage"
        VECTOR_DB_DIR: Path = STORAGE_DIR / "vector_db"
        RAW_DOCS_DIR: Path = STORAGE_DIR / "raw_documents"
        PROCESSED_IMAGES_DIR: Path = STORAGE_DIR / "processed_images"

        CHROMA_COLLECTION_NAME: str = "ocr_documents_collection"

        DEFAULT_CHUNK_SIZE: int = 500
        DEFAULT_CHUNK_OVERLAP: int = 100
        DEFAULT_TOP_K: int = 4

        EMBEDDING_PROVIDER: str = "lightweight"
        LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-m3"

        LLM_PROVIDER: str = "gemini"
        GEMINI_API_KEY: str = Field(default="", alias="GEMINI_API_KEY")
        GEMINI_MODEL_NAME: str = "gemini-1.5-flash"

        class Config:
            env_file = ".env"
            extra = "allow"

    settings = Settings()

except ImportError:
    class Settings:
        APP_NAME: str = "OCR RAG AI Worker"
        APP_VERSION: str = "1.0.0"
        DEBUG: bool = True

        BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
        STORAGE_DIR: Path = BASE_DIR / "storage"
        VECTOR_DB_DIR: Path = STORAGE_DIR / "vector_db"
        RAW_DOCS_DIR: Path = STORAGE_DIR / "raw_documents"
        PROCESSED_IMAGES_DIR: Path = STORAGE_DIR / "processed_images"

        CHROMA_COLLECTION_NAME: str = "ocr_documents_collection"

        DEFAULT_CHUNK_SIZE: int = 500
        DEFAULT_CHUNK_OVERLAP: int = 100
        DEFAULT_TOP_K: int = 4

        EMBEDDING_PROVIDER: str = "lightweight"
        LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-m3"

        LLM_PROVIDER: str = "gemini"
        GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
        GEMINI_MODEL_NAME: str = "gemini-1.5-flash"

    settings = Settings()

# Đảm bảo các thư mục lưu trữ luôn tồn tại trên đĩa
settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
