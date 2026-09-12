"""
API Router cho Module OCR (Nhận Diện Chữ & Bounding Box).

Các Endpoints:
- POST /api/v1/ocr/extract: Tải ảnh lên và nhận về JSON chứa text + Bounding Box (pixel + chuẩn hóa).
- POST /api/v1/ocr/extract-and-ingest: Tải ảnh lên, OCR và nạp ngay lập tức vào Vector Database trong một bước.
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from fastapi.concurrency import run_in_threadpool
import time
import shutil
from pathlib import Path

from app.models.ocr_models import OCRExtractResponse, OCRPageResult
from app.models.rag_models import DocumentIngestRequest, DocumentIngestResponse
from app.services.ocr.ocr_engine import BaseOCREngine
from app.services.rag.rag_pipeline import RAGPipeline
from app.api.deps import get_ocr_service, get_rag_pipeline
from app.core.config import settings

router = APIRouter(prefix="/ocr", tags=["OCR - Nhận Diện Chữ"])


@router.post(
    "/extract",
    response_model=OCRExtractResponse,
    summary="Trích xuất chữ và tọa độ Bounding Box từ tệp ảnh"
)
async def extract_text_from_image(
    file: UploadFile = File(..., description="Tệp ảnh tài liệu (PNG, JPG, JPEG, TIFF)"),
    ocr_service: BaseOCREngine = Depends(get_ocr_service)
):
    """
    Nhận tệp ảnh, thực thi quy trình:
    1. Lưu tệp tạm thời vào thư mục storage/raw_documents
    2. Chạy tiền xử lý và nhận diện chữ (đẩy vào run_in_threadpool để không block Event Loop)
    3. Trả về cấu trúc JSON gồm: các dòng chữ, độ tin cậy và tọa độ Bounding Box (tuyệt đối + chuẩn hóa [0.0, 1.0]).
    """
    allowed_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng tệp không được hỗ trợ: {file_ext}. Vui lòng tải file: {', '.join(allowed_extensions)}"
        )

    start_time = time.time()
    save_path = settings.RAW_DOCS_DIR / file.filename

    # Ghi tệp an toàn
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Chạy tác vụ OCR nặng trong Threadpool
    page_result: OCRPageResult = await run_in_threadpool(
        ocr_service.extract, str(save_path), 1
    )

    elapsed_ms = (time.time() - start_time) * 1000.0

    return OCRExtractResponse(
        filename=file.filename,
        total_pages=1,
        pages=[page_result],
        processing_time_ms=round(elapsed_ms, 2),
        status="success"
    )


@router.post(
    "/extract-and-ingest",
    response_model=DocumentIngestResponse,
    summary="OCR tệp ảnh và nạp tự động vào Vector Database (End-to-End)"
)
async def extract_and_ingest_document(
    file: UploadFile = File(..., description="Tệp ảnh cần OCR và nạp vào RAG"),
    document_id: str = Form(..., description="Mã định danh cho tài liệu (ví dụ: DOC_HD_01)"),
    ocr_service: BaseOCREngine = Depends(get_ocr_service),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Quy trình tích hợp 2-trong-1:
    1. Nhận ảnh và chạy OCR trích xuất văn bản + Bounding Box.
    2. Tự động cắt đoạn (Chunking), tạo Vector Embedding và lưu vào ChromaDB.
    """
    save_path = settings.RAW_DOCS_DIR / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Chạy OCR
    page_result: OCRPageResult = await run_in_threadpool(
        ocr_service.extract, str(save_path), 1
    )

    # 2. Nạp vào RAG Pipeline
    ingest_req = DocumentIngestRequest(
        document_id=document_id,
        filename=file.filename,
        content_text=page_result.full_text,
        page_number=1,
        metadata={"width": page_result.width, "height": page_result.height}
    )

    ingest_res = await run_in_threadpool(rag_pipeline.ingest_document, ingest_req)
    return ingest_res
