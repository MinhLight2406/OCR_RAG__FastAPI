"""
API Router cho Module Quản Lý Tài Liệu (Document Management).

Các Endpoints:
- GET /api/v1/documents: Liệt kê danh sách các tệp tài liệu đã tải lên thư mục storage/raw_documents.
"""

from fastapi import APIRouter
from typing import List, Dict, Any
from pathlib import Path
from app.core.config import settings

router = APIRouter(prefix="/documents", tags=["Documents - Quản Lý Tài Liệu"])


@router.get(
    "",
    summary="Liệt kê toàn bộ các tệp tài liệu trong hệ thống"
)
async def list_stored_documents() -> List[Dict[str, Any]]:
    """Trả về danh sách các tệp ảnh và PDF hiện có trong kho lưu trữ storage/raw_documents."""
    documents = []
    if settings.RAW_DOCS_DIR.exists():
        for item in settings.RAW_DOCS_DIR.iterdir():
            if item.is_file() and not item.name.startswith("."):
                documents.append({
                    "filename": item.name,
                    "size_bytes": item.stat().st_size,
                    "file_extension": item.suffix.lower(),
                    "path": str(item)
                })
    return documents
