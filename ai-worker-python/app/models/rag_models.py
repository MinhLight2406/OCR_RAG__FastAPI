"""
Pydantic Models định nghĩa cấu trúc dữ liệu cho Module RAG:
- Quản lý Chunk tài liệu
- Request / Response cho API tìm kiếm & hỏi đáp
- Trích dẫn nguồn (Citations)
Hỗ trợ cả Pydantic và DataClass Fallback để chạy demo ngay lập tức mà không cần cài thêm thư viện.
"""

from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field

    class DocumentChunk(BaseModel):
        chunk_id: str = Field(..., description="ID duy nhất của đoạn")
        text: str = Field(..., description="Nội dung văn bản của đoạn")
        document_id: str = Field(..., description="ID của tài liệu chứa đoạn này")
        source_file: str = Field(..., description="Tên tệp tài liệu gốc")
        page_number: int = Field(default=1, description="Số trang chứa đoạn văn này")
        chunk_index: int = Field(default=0, description="Vị trí thứ tự của đoạn")
        bbox: Optional[List[float]] = Field(default=None, description="Tọa độ vùng chữ [x1, y1, x2, y2]")
        extra_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class SourceCitation(BaseModel):
        document_id: str
        source_file: str
        page_number: int
        chunk_index: int
        text_snippet: str
        relevance_score: float
        bbox: Optional[List[float]] = None

    class DocumentIngestRequest(BaseModel):
        document_id: str
        filename: str
        content_text: str
        page_number: int = 1
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class DocumentIngestResponse(BaseModel):
        document_id: str
        filename: str
        total_chunks_created: int
        status: str = "success"
        message: str

    class RagQueryRequest(BaseModel):
        query: str
        top_k: int = 4
        filter_document_id: Optional[str] = None
        use_hybrid_search: bool = True

    class RagQueryResponse(BaseModel):
        query: str
        answer: str
        sources: List[SourceCitation] = Field(default_factory=list)
        processing_time_ms: float

except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class DocumentChunk:
        chunk_id: str
        text: str
        document_id: str
        source_file: str
        page_number: int = 1
        chunk_index: int = 0
        bbox: Optional[List[float]] = None
        extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class SourceCitation:
        document_id: str
        source_file: str
        page_number: int
        chunk_index: int
        text_snippet: str
        relevance_score: float
        bbox: Optional[List[float]] = None

    @dataclass
    class DocumentIngestRequest:
        document_id: str
        filename: str
        content_text: str
        page_number: int = 1
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class DocumentIngestResponse:
        document_id: str
        filename: str
        total_chunks_created: int
        status: str = "success"
        message: str = ""

    @dataclass
    class RagQueryRequest:
        query: str
        top_k: int = 4
        filter_document_id: Optional[str] = None
        use_hybrid_search: bool = True

    @dataclass
    class RagQueryResponse:
        query: str
        answer: str
        sources: List[SourceCitation] = field(default_factory=list)
        processing_time_ms: float = 0.0
