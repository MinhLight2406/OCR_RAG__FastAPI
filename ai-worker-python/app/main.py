"""
Điểm Khởi Động Chính Của Ứng Dụng FastAPI (FastAPI Application Entrypoint).

MỤC ĐÍCH HỌC TẬP:
- Đóng gói toàn bộ các routers (OCR, RAG, Documents) thành một ứng dụng web thống nhất.
- Cấu hình Middleware (CORS) cho phép giao diện Frontend kết nối an toàn.
- Cung cấp trang tài liệu tương tác tự động Swagger UI tại: http://localhost:8000/docs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from app.core.config import settings
from app.api.v1.ocr import router as ocr_router
from app.api.v1.rag import router as rag_router
from app.api.v1.documents import router as documents_router

# Khởi tạo ứng dụng FastAPI với đầy đủ Metadata hiển thị trên Swagger UI
app = FastAPI(
    title="OCR & RAG Intelligent Document Processing API",
    description=(
        "## Hệ Thống Xử Lý Tài Liệu Thông Minh Kết Hợp OCR và RAG\n\n"
        "### 🎯 Tính Năng Chính:\n"
        "- **OCR Module:** Trích xuất văn bản từ ảnh chụp/PDF kèm tọa độ Bounding Box (pixel & chuẩn hóa).\n"
        "- **RAG Pipeline:** Cắt đoạn đệ quy, lưu trữ Vector ChromaDB, tìm kiếm kết hợp Hybrid Search (Dense + BM25).\n"
        "- **Hỏi Đáp Thông Minh:** Tổng hợp câu trả lời từ tài liệu gốc, trích dẫn nguồn trang và chống ảo giác.\n"
        "- **Streaming SSE:** Truyền phát câu trả lời thời gian thực từng từ.\n\n"
        "👉 **Tài liệu học tập chi tiết:** Nằm trong thư mục `/learning` của dự án."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# 1. Cấu hình CORS Middleware (Cho phép Frontend kết nối từ bất kỳ port nào)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Đăng ký các Router vào tiền tố /api/v1
app.include_router(ocr_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")


@app.get("/", summary="Trang chủ API & Bản đồ các Endpoints")
async def root():
    """Trả về thông tin tổng quan và đường dẫn truy cập Swagger UI."""
    return JSONResponse(
        content={
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "online",
            "swagger_ui": "/docs",
            "redoc_ui": "/redoc",
            "available_endpoints": {
                "ocr_extract": "POST /api/v1/ocr/extract",
                "ocr_and_ingest": "POST /api/v1/ocr/extract-and-ingest",
                "rag_ingest": "POST /api/v1/rag/ingest",
                "rag_query": "POST /api/v1/rag/query",
                "rag_stream": "GET /api/v1/rag/chat-stream",
                "rag_stats": "GET /api/v1/rag/stats",
                "documents_list": "GET /api/v1/documents"
            }
        }
    )


@app.get("/health", summary="Kiểm tra trạng thái hệ thống (Health Check)")
async def health_check():
    """Endpoint giám sát tình trạng hoạt động của server."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "storage_ready": settings.STORAGE_DIR.exists()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
