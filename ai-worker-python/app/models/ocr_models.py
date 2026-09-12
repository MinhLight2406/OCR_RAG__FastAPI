"""
Pydantic / DataClass Models định nghĩa cấu trúc dữ liệu cho Module OCR:
- Tọa độ Bounding Box (Tuyệt đối và Chuẩn hóa)
- Cấp độ Từ (Word), Dòng (Line), Trang (Page)
- Request / Response cho API nhận diện chữ
"""

from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field

    class BoundingBox(BaseModel):
        """Hộp bao quanh chữ (Bounding Box)."""
        x_min: float = Field(..., description="Tọa độ góc trái x (pixel)")
        y_min: float = Field(..., description="Tọa độ góc trên y (pixel)")
        x_max: float = Field(..., description="Tọa độ góc phải x (pixel)")
        y_max: float = Field(..., description="Tọa độ góc dưới y (pixel)")
        polygon: Optional[List[List[float]]] = Field(
            default=None,
            description="Tọa độ đa giác 4 đỉnh [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]"
        )
        normalized: Optional[List[float]] = Field(
            default=None,
            description="Tọa độ chuẩn hóa trong dải [0.0, 1.0]: [norm_xmin, norm_ymin, norm_xmax, norm_ymax]"
        )

    class OCRWord(BaseModel):
        """Đơn vị từ đơn lẻ."""
        text: str
        bbox: BoundingBox
        confidence: float = Field(default=1.0, description="Độ tin cậy từ 0.0 đến 1.0")

    class OCRLine(BaseModel):
        """Một dòng văn bản gồm nhiều từ ghép lại."""
        line_number: int
        text: str
        bbox: BoundingBox
        words: List[OCRWord] = Field(default_factory=list)
        confidence: float = 1.0

    class OCRPageResult(BaseModel):
        """Kết quả OCR hoàn chỉnh cho một trang tài liệu."""
        page_number: int = 1
        width: int = Field(..., description="Chiều rộng ảnh (pixel)")
        height: int = Field(..., description="Chiều cao ảnh (pixel)")
        lines: List[OCRLine] = Field(default_factory=list)
        full_text: str = Field(..., description="Toàn bộ văn bản đã sắp xếp theo thứ tự đọc tự nhiên")
        avg_confidence: float = 1.0

    class OCRExtractResponse(BaseModel):
        filename: str
        total_pages: int = 1
        pages: List[OCRPageResult]
        processing_time_ms: float
        status: str = "success"

except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class BoundingBox:
        x_min: float
        y_min: float
        x_max: float
        y_max: float
        polygon: Optional[List[List[float]]] = None
        normalized: Optional[List[float]] = None

    @dataclass
    class OCRWord:
        text: str
        bbox: BoundingBox
        confidence: float = 1.0

    @dataclass
    class OCRLine:
        line_number: int
        text: str
        bbox: BoundingBox
        words: List[OCRWord] = field(default_factory=list)
        confidence: float = 1.0

    @dataclass
    class OCRPageResult:
        page_number: int
        width: int
        height: int
        lines: List[OCRLine] = field(default_factory=list)
        full_text: str = ""
        avg_confidence: float = 1.0

    @dataclass
    class OCRExtractResponse:
        filename: str
        total_pages: int = 1
        pages: List[OCRPageResult] = field(default_factory=list)
        processing_time_ms: float = 0.0
        status: str = "success"
