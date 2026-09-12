"""
API Router cho Module RAG (Truy Vấn & Hỏi Đáp Tài Liệu).

Các Endpoints:
- POST /api/v1/rag/ingest: Nạp trực tiếp văn bản vào Vector DB.
- POST /api/v1/rag/query: Gửi câu hỏi và nhận câu trả lời tổng hợp kèm trích dẫn nguồn.
- GET /api/v1/rag/chat-stream: Truy vấn theo dạng Server-Sent Events (SSE) chạy từng từ.
- GET /api/v1/rag/stats: Thống kê số lượng chunk trong Vector Database.
"""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
import asyncio
import json

from app.models.rag_models import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    RagQueryRequest,
    RagQueryResponse
)
from app.services.rag.rag_pipeline import RAGPipeline
from app.api.deps import get_rag_pipeline

router = APIRouter(prefix="/rag", tags=["RAG - Hỏi Đáp Tài Liệu"])


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    summary="Nạp văn bản vào cơ sở dữ liệu vector ChromaDB"
)
async def ingest_document(
    request: DocumentIngestRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Nạp tài liệu dạng text:
    - Cắt đoạn (Recursive Chunking)
    - Nhúng vector (Embedding)
    - Lưu vào ChromaDB
    """
    return await run_in_threadpool(pipeline.ingest_document, request)


@router.post(
    "/query",
    response_model=RagQueryResponse,
    summary="Đặt câu hỏi và nhận câu trả lời từ tài liệu"
)
async def query_rag(
    request: RagQueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Quy trình xử lý câu hỏi:
    1. Tìm kiếm các đoạn tài liệu tương đồng nhất bằng Hybrid Search (Dense + BM25).
    2. Tổng hợp Context và gửi vào LLM sinh câu trả lời kèm trích dẫn nguồn.
    """
    return await run_in_threadpool(pipeline.query, request)


@router.get(
    "/chat-stream",
    summary="Hỏi đáp tài liệu dạng Streaming từng từ (Server-Sent Events - SSE)"
)
async def chat_stream(
    query: str = Query(..., description="Câu hỏi của bạn"),
    top_k: int = Query(default=3, description="Số lượng đoạn tài liệu liên quan"),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Truyền phát câu trả lời thời gian thực (Real-time Streaming) giống trải nghiệm ChatGPT.
    Frontend có thể dùng EventSource hoặc fetch() để lắng nghe luồng dữ liệu này.
    """
    async def event_generator():
        # Lấy câu trả lời hoàn chỉnh từ pipeline
        query_req = RagQueryRequest(query=query, top_k=top_k)
        result: RagQueryResponse = await run_in_threadpool(pipeline.query, query_req)

        # Bắn từng từ ra luồng SSE
        words = result.answer.split(" ")
        for word in words:
            data = json.dumps({"token": word + " "}, ensure_ascii=False)
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.03)  # Giả lập hiệu ứng gõ máy tính 30ms

        # Bắn thông tin trích dẫn nguồn cuối cùng
        sources_payload = [s.__dict__ if hasattr(s, "__dict__") else s for s in result.sources]
        done_data = json.dumps({"done": True, "sources": sources_payload}, ensure_ascii=False)
        yield f"data: {done_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.get(
    "/stats",
    summary="Thống kê cơ sở dữ liệu Vector Store"
)
async def get_rag_stats(
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """Xem số lượng Chunks hiện tại đang được lưu trong Vector DB."""
    total_count = pipeline.vector_store.count()
    return {
        "status": "online",
        "total_chunks_indexed": total_count,
        "embedding_provider": pipeline.embedding_engine.__class__.__name__,
        "vector_dimension": pipeline.embedding_engine.dimension
    }
