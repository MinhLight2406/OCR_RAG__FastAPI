"""
Unit Tests cho Module OCR & Tiền Xử Lý Ảnh.

Kiểm tra:
1. Chuẩn hóa Bounding Box sang dải [0.0, 1.0]
2. Thuật toán tính góc nghiêng Deskew
3. Thuật toán sắp xếp thứ tự đọc (Reading Order Sorter)
4. OCR Page Extraction & Line grouping
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.image_processing import normalize_bbox, create_bounding_box, ImagePreprocessor
from app.services.ocr.ocr_engine import get_ocr_engine, ReadingOrderSorter
from app.models.ocr_models import OCRWord


def test_normalize_bbox():
    norm = normalize_bbox(x_min=120, y_min=160, x_max=600, y_max=800, width=1200, height=1600)
    assert norm == [0.1, 0.1, 0.5, 0.5]
    print("[PASS] test_normalize_bbox passed.")


def test_create_bounding_box():
    bbox = create_bounding_box(100, 200, 300, 400, 1000, 1000)
    assert bbox.x_min == 100.0
    assert bbox.y_max == 400.0
    assert bbox.normalized == [0.1, 0.2, 0.3, 0.4]
    assert len(bbox.polygon) == 4
    print("[PASS] test_create_bounding_box passed.")


def test_deskew_angle_calculation():
    points = [(0, 0), (10, 1), (20, 2)]  # Đường dốc 1/10 ~ 5.7 độ
    angle = ImagePreprocessor.calculate_deskew_angle_from_points(points)
    assert 5.0 < angle < 6.0
    print("[PASS] test_deskew_angle_calculation passed.")


def test_reading_order_sorter():
    # Tạo 2 dòng bị xáo trộn thứ tự
    b1 = create_bounding_box(10, 10, 50, 30, 1000, 1000)
    w1 = OCRWord(text="Xin", bbox=b1)

    b2 = create_bounding_box(60, 10, 100, 30, 1000, 1000)
    w2 = OCRWord(text="chào", bbox=b2)

    b3 = create_bounding_box(10, 50, 80, 70, 1000, 1000)
    w3 = OCRWord(text="Việt Nam", bbox=b3)

    # Đưa vào theo thứ tự lộn xộn: dòng 2 trước, rồi mới tới dòng 1
    unordered = [w3, w2, w1]
    lines = ReadingOrderSorter.sort_words_into_lines(unordered)

    assert len(lines) == 2
    assert lines[0].text == "Xin chào"
    assert lines[1].text == "Việt Nam"
    print("[PASS] test_reading_order_sorter passed.")


def test_ocr_engine_extraction():
    engine = get_ocr_engine("simulated")
    page = engine.extract("test_image.png", page_number=1)
    assert page.page_number == 1
    assert len(page.lines) > 0
    assert "HỢP ĐỒNG" in page.full_text
    assert page.avg_confidence > 0.9
    print("[PASS] test_ocr_engine_extraction passed.")


if __name__ == "__main__":
    test_normalize_bbox()
    test_create_bounding_box()
    test_deskew_angle_calculation()
    test_reading_order_sorter()
    test_ocr_engine_extraction()
    print("\n[SUCCESS] ALL OCR TESTS PASSED!")
