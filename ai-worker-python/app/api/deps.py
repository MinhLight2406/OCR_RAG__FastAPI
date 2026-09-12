"""
Dependency Injection (Quản lý các đối tượng Singleton phụ thuộc).

MỤC ĐÍCH HỌC TẬP:
- Áp dụng mẫu thiết kế Dependency Injection trong FastAPI.
- Đảm bảo các mô hình nặng (RAGPipeline, OCREngine, ChromaDB) chỉ khởi tạo MỘT LẦN duy nhất
  khi server khởi động, không khởi tạo lại ở mỗi request gây tốn RAM và chậm chạp.
"""

from functools import lru_cache
from app.services.rag.rag_pipeline import RAGPipeline
from app.services.ocr.ocr_engine import get_ocr_engine, BaseOCREngine
from app.core.config import settings


@lru_cache()
def get_rag_pipeline() -> RAGPipeline:
    """Trả về phiên bản Singleton của RAGPipeline."""
    return RAGPipeline(
        chunk_size=settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
        embedding_provider=settings.EMBEDDING_PROVIDER
    )


@lru_cache()
def get_ocr_service() -> BaseOCREngine:
    """Trả về phiên bản Singleton của OCR Engine."""
    return get_ocr_engine(provider="simulated")
