"""
Bộ Điều Phối Quy Trình RAG Hoàn Chỉnh (RAG Pipeline Orchestrator).

MỤC ĐÍCH HỌC TẬP:
- Đóng gói toàn bộ chuỗi mắt xích:
  Ingest Text -> Chunking -> Embedding -> Vector DB -> Hybrid Retrieval -> LLM Generation.
- Đo lường thời gian thực thi (Latency / Processing Time) cho từng yêu cầu.
"""

import time
from typing import Optional
from app.models.rag_models import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    RagQueryRequest,
    RagQueryResponse
)
from app.core.config import settings
from app.services.rag.chunking import RecursiveCharacterChunker
from app.services.rag.embedding import (
    BaseEmbedding,
    LightweightEmbedding,
    LocalSentenceTransformerEmbedding
)
from app.db.chroma_client import ChromaVectorStore
from app.services.rag.retriever import HybridRetriever
from app.services.rag.generator import LLMGenerator


class RAGPipeline:
    """Pipeline RAG trung tâm kết nối toàn bộ các thành phần."""

    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
        embedding_provider: str = settings.EMBEDDING_PROVIDER
    ):
        # 1. Khởi tạo bộ cắt đoạn
        self.chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 2. Khởi tạo mô hình Embedding
        if embedding_provider == "local":
            self.embedding_engine: BaseEmbedding = LocalSentenceTransformerEmbedding(
                model_name=settings.LOCAL_EMBEDDING_MODEL
            )
        else:
            self.embedding_engine: BaseEmbedding = LightweightEmbedding()

        # 3. Khởi tạo Vector Store
        self.vector_store = ChromaVectorStore()

        # 4. Khởi tạo bộ truy xuất Hybrid
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_engine=self.embedding_engine
        )

        # 5. Khởi tạo bộ sinh câu trả lời LLM
        self.generator = LLMGenerator()

    def ingest_document(self, request: DocumentIngestRequest) -> DocumentIngestResponse:
        """
        GIAI ĐOẠN 1: Nạp tài liệu vào hệ thống.
        1. Cắt nhỏ văn bản thành chunks
        2. Nhúng từng chunk thành vector
        3. Lưu vào Vector Database
        """
        # Bước 1: Chunking
        chunks = self.chunker.split_text(
            text=request.content_text,
            document_id=request.document_id,
            source_file=request.filename,
            page_number=request.page_number
        )

        if not chunks:
            return DocumentIngestResponse(
                document_id=request.document_id,
                filename=request.filename,
                total_chunks_created=0,
                status="warning",
                message="Tài liệu rỗng hoặc không có nội dung hợp lệ sau khi làm sạch."
            )

        # Bước 2: Embedding
        texts_to_embed = [c.text for c in chunks]
        embeddings = self.embedding_engine.embed_batch(texts_to_embed)

        # Bước 3: Lưu vào Vector DB
        self.vector_store.add_chunks(chunks, embeddings)

        return DocumentIngestResponse(
            document_id=request.document_id,
            filename=request.filename,
            total_chunks_created=len(chunks),
            status="success",
            message=f"Đã nạp thành công {len(chunks)} đoạn vào Vector Database."
        )

    def query(self, request: RagQueryRequest) -> RagQueryResponse:
        """
        GIAI ĐOẠN 2: Truy vấn & Hỏi đáp.
        1. Tìm kiếm các đoạn tài liệu phù hợp nhất
        2. Gửi context vào LLM để sinh câu trả lời
        """
        start_time = time.time()

        # Bước 1: Truy xuất Top-K Chunks
        citations = self.retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filter_document_id=request.filter_document_id,
            use_hybrid=request.use_hybrid_search
        )

        # Bước 2: Sinh câu trả lời từ LLM
        answer = self.generator.generate_answer(
            query=request.query,
            citations=citations
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        return RagQueryResponse(
            query=request.query,
            answer=answer,
            sources=citations,
            processing_time_ms=round(elapsed_ms, 2)
        )
