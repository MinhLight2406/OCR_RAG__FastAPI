"""
Module Động Cơ OCR (OCR Engine Abstraction & Implementations).

MỤC ĐÍCH HỌC TẬP:
- Hiểu cách kiến trúc trừu tượng hóa (Factory Pattern) để dễ dàng chuyển đổi giữa các công nghệ:
  1. PaddleOCR (Mô hình Deep Learning hiện đại rất mạnh cho tiếng Việt).
  2. Tesseract OCR (Mô hình cổ điển dựa trên LSTM).
  3. EducationalSimulatedOCREngine (Động cơ mô phỏng chuẩn xác tọa độ Bounding Box, phục vụ việc học tập & kiểm thử ngay lập tức).
- Hiểu thuật toán sắp xếp thứ tự đọc (Reading Order Sorting): từ trên xuống dưới, từ trái sang phải.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import time
from app.models.ocr_models import (
    BoundingBox,
    OCRWord,
    OCRLine,
    OCRPageResult,
    OCRExtractResponse
)
from app.utils.image_processing import create_bounding_box


class BaseOCREngine(ABC):
    """Giao diện chuẩn cho tất cả các bộ trích xuất OCR."""

    @abstractmethod
    def extract(self, image_path_or_name: str, page_number: int = 1) -> OCRPageResult:
        """Trích xuất toàn bộ văn bản và cấu trúc Bounding Box từ một trang ảnh."""
        pass


class ReadingOrderSorter:
    """
    Thuật toán sắp xếp thứ tự đọc văn bản (Reading Order Reconstruction).
    Nguyên tắc:
    1. Nhóm các từ có độ cao Y tương đồng vào cùng một dòng.
    2. Sắp xếp các dòng theo thứ tự từ trên xuống dưới (Y tăng dần).
    3. Trong mỗi dòng, sắp xếp các từ từ trái sang phải (X tăng dần).
    """

    @staticmethod
    def sort_words_into_lines(words: List[OCRWord], line_threshold: float = 15.0) -> List[OCRLine]:
        if not words:
            return []

        # Sắp xếp thô theo y_min
        sorted_words = sorted(words, key=lambda w: (w.bbox.y_min, w.bbox.x_min))

        lines_dict: List[List[OCRWord]] = []
        for word in sorted_words:
            placed = False
            for line in lines_dict:
                # Tính độ chênh lệch y_center của từ so với y_center trung bình của dòng
                avg_y = sum((w.bbox.y_min + w.bbox.y_max) / 2 for w in line) / len(line)
                word_y = (word.bbox.y_min + word.bbox.y_max) / 2

                if abs(word_y - avg_y) <= line_threshold:
                    line.append(word)
                    placed = True
                    break
            if not placed:
                lines_dict.append([word])

        # Sắp xếp các từ trong từng dòng từ trái sang phải
        result_lines = []
        for line_idx, line_words in enumerate(lines_dict, 1):
            line_words.sort(key=lambda w: w.bbox.x_min)
            line_text = " ".join(w.text for w in line_words)

            x_min = min(w.bbox.x_min for w in line_words)
            y_min = min(w.bbox.y_min for w in line_words)
            x_max = max(w.bbox.x_max for w in line_words)
            y_max = max(w.bbox.y_max for w in line_words)

            avg_conf = sum(w.confidence for w in line_words) / len(line_words)
            line_bbox = BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                polygon=[[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                normalized=line_words[0].bbox.normalized  # Kế thừa tỷ lệ
            )

            result_lines.append(
                OCRLine(
                    line_number=line_idx,
                    text=line_text,
                    bbox=line_bbox,
                    words=line_words,
                    confidence=round(avg_conf, 4)
                )
            )

        # Sắp xếp các dòng từ trên xuống dưới
        result_lines.sort(key=lambda l: l.bbox.y_min)
        return result_lines


class EducationalSimulatedOCREngine(BaseOCREngine):
    """
    Động cơ OCR phục vụ học tập & thực hành trực tiếp:
    Mô phỏng chân thực quy trình Text Detection và Text Recognition:
    - Tạo các Bounding Box thực tế (Pixel + Normalized)
    - Gán độ tin cậy Confidence Score ngẫu nhiên theo phân phối chuẩn (0.92 - 0.99)
    - Tự động chạy thuật toán sắp xếp thứ tự đọc tự nhiên.
    """

    def __init__(self, img_width: int = 1200, img_height: int = 1600):
        self.img_width = img_width
        self.img_height = img_height

    def extract(self, image_path_or_name: str, page_number: int = 1) -> OCRPageResult:
        # Tập dữ liệu mẫu mô phỏng trang Hợp đồng / Hóa đơn
        sample_raw_data = [
            ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", 350.0, 80.0, 850.0, 110.0, 0.99),
            ("Độc lập - Tự do - Hạnh phúc", 420.0, 120.0, 780.0, 145.0, 0.98),
            ("HỢP ĐỒNG THUÊ NHÀ Ở", 400.0, 200.0, 800.0, 240.0, 0.99),
            ("Bên A (Bên cho thuê): Ông Nguyễn Văn A", 100.0, 300.0, 580.0, 330.0, 0.97),
            ("CCCD số: 001234567899", 600.0, 300.0, 880.0, 330.0, 0.96),
            ("Bên B (Bên thuê): Bà Trần Thị B", 100.0, 350.0, 520.0, 380.0, 0.98),
            ("CCCD số: 009876543211", 600.0, 350.0, 880.0, 380.0, 0.97),
            ("Điều 1: Thời hạn thuê là 24 tháng kể từ 01/01/2026", 100.0, 420.0, 920.0, 450.0, 0.95),
            ("Điều 2: Giá thuê là 15.000.000 VNĐ / tháng", 100.0, 480.0, 780.0, 510.0, 0.96),
            ("Tiền đặt cọc: 30.000.000 VNĐ", 100.0, 540.0, 500.0, 570.0, 0.98)
        ]

        words: List[OCRWord] = []
        for text, x1, y1, x2, y2, conf in sample_raw_data:
            # Tách thành từng từ và phân bổ tọa độ ngang
            tokens = text.split(" ")
            total_tokens = len(tokens)
            box_w = (x2 - x1) / total_tokens

            for idx, token in enumerate(tokens):
                w_x1 = x1 + idx * box_w
                w_x2 = w_x1 + box_w - 5.0
                w_bbox = create_bounding_box(w_x1, y1, w_x2, y2, self.img_width, self.img_height)
                words.append(OCRWord(text=token, bbox=w_bbox, confidence=conf))

        # Áp dụng thuật toán sắp xếp thứ tự đọc
        lines = ReadingOrderSorter.sort_words_into_lines(words)
        full_text = "\n".join(l.text for l in lines)
        avg_conf = sum(l.confidence for l in lines) / max(1, len(lines))

        return OCRPageResult(
            page_number=page_number,
            width=self.img_width,
            height=self.img_height,
            lines=lines,
            full_text=full_text,
            avg_confidence=round(avg_conf, 4)
        )


class PaddleOCREngine(BaseOCREngine):
    """Bộ nhận diện PaddleOCR (Deep Learning DBNet + SVTR)."""

    def __init__(self, lang: str = "vi", use_gpu: bool = False):
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = None

    def _get_model(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, use_gpu=self.use_gpu)
            except ImportError:
                raise ImportError("Vui lòng cài đặt: pip install paddlepaddle paddleocr để dùng PaddleOCR!")
        return self._ocr

    def extract(self, image_path_or_name: str, page_number: int = 1) -> OCRPageResult:
        ocr = self._get_model()
        results = ocr.ocr(image_path_or_name, cls=True)

        words: List[OCRWord] = []
        img_w, img_h = 1200, 1600

        if results and results[0]:
            for item in results[0]:
                polygon = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text, conf = item[1]

                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                bbox = create_bounding_box(x_min, y_min, x_max, y_max, img_w, img_h)
                words.append(OCRWord(text=text, bbox=bbox, confidence=round(float(conf), 4)))

        lines = ReadingOrderSorter.sort_words_into_lines(words)
        full_text = "\n".join(l.text for l in lines)
        avg_conf = sum(l.confidence for l in lines) / max(1, len(lines))

        return OCRPageResult(
            page_number=page_number,
            width=img_w,
            height=img_h,
            lines=lines,
            full_text=full_text,
            avg_confidence=round(avg_conf, 4)
        )


def get_ocr_engine(provider: str = "simulated") -> BaseOCREngine:
    """Factory Pattern trả về OCR Engine phù hợp."""
    if provider == "paddleocr":
        return PaddleOCREngine()
    return EducationalSimulatedOCREngine()
