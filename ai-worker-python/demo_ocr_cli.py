"""
Script Thực Hành Tương Tác: Khám Phá Toàn Bộ Quy Trình OCR & Tiền Xử Lý Ảnh.

Cách chạy:
    cd ai-worker-python
    python demo_ocr_cli.py

Mục tiêu học tập:
1. Quan sát cách tiền xử lý ảnh: Grayscale -> Denoise -> Deskew -> Adaptive Threshold.
2. Trích xuất Bounding Box ở cả 2 định dạng: Tuyệt đối (Pixel) và Chuẩn hóa ([0.0, 1.0]).
3. Quan sát thuật toán sắp xếp thứ tự đọc (Reading Order).
4. KẾT HỢP VỚI RAG: Đưa kết quả OCR có Bounding Box thẳng vào RAG Pipeline để hỏi đáp!
"""

import sys
import os

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.image_processing import ImagePreprocessor
from app.services.ocr.ocr_engine import get_ocr_engine
from app.services.rag.rag_pipeline import RAGPipeline
from app.models.rag_models import DocumentIngestRequest, RagQueryRequest


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  [+] {title}")
    print("=" * 70)


def main():
    print_banner("CHUONG TRINH THUC HANH: OCR & TIEN XU LY ANH CO BOUNDING BOX")

    # 1. Khởi tạo Engine
    print("\n[1] Đang khởi tạo Module Tiền Xử Lý & Động cơ OCR Engine...")
    ocr_engine = get_ocr_engine(provider="simulated")
    print("    -> Khởi tạo OCR Engine thành công!")

    # 2. Tiền xử lý ảnh mẫu
    print_banner("BUOC 1: TIEN XU LY ANH (PREPROCESSING)")
    simulated_points = [(100, 200), (250, 210), (400, 220), (550, 230)]
    detected_angle = ImagePreprocessor.calculate_deskew_angle_from_points(simulated_points)
    print(f"  [*] Phat hien goc nghieng tai lieu (Deskew Angle): {detected_angle}°")
    print("  [*] Thuc hien phep xoay Affine Matrix de can thang tai lieu.")
    print("  [*] Ap dung Adaptive Gaussian Thresholding de khu bong do.")

    # 3. Chạy OCR Text Detection & Recognition
    print_banner("BUOC 2: TRICH XUAT TEXT & TOA DO BOUNDING BOX")
    page_result = ocr_engine.extract("sample_contract.png", page_number=1)

    print(f"  [*] Kich thuoc trang anh: {page_result.width} x {page_result.height} pixels")
    print(f"  [*] So dong van ban phat hien: {len(page_result.lines)} dong")
    print(f"  [*] Do tin cay trung binh (Confidence): {page_result.avg_confidence * 100:.2f}%\n")

    print(f"{'DÒNG':<6} | {'TỌA ĐỘ PIXEL [X1, Y1, X2, Y2]':<32} | {'TỌA ĐỘ CHUẨN HÓA [0.0-1.0]':<26} | NỘI DUNG")
    print("-" * 100)

    for line in page_result.lines:
        b = line.bbox
        pixel_box = f"[{b.x_min:.0f}, {b.y_min:.0f}, {b.x_max:.0f}, {b.y_max:.0f}]"
        norm_box = f"[{b.normalized[0]:.2f}, {b.normalized[1]:.2f}, {b.normalized[2]:.2f}, {b.normalized[3]:.2f}]"
        print(f"Dong {line.line_number:<2} | {pixel_box:<32} | {norm_box:<26} | {line.text}")

    # 4. Tích hợp trực tiếp với RAG Pipeline
    print_banner("BUOC 3: KET NOI OCR VOI HE THONG RAG (END-TO-END PIPELINE)")
    print("  [*] Nap van ban vua OCR duoc kem Metadata toa do vao Vector Database...")

    # Cắt chunk kích thước 150 ký tự để mỗi điều khoản là 1 chunk riêng biệt
    rag_pipeline = RAGPipeline(chunk_size=150, chunk_overlap=30)
    ingest_req = DocumentIngestRequest(
        document_id="DOC_OCR_HOP_DONG",
        filename="Hop_Dong_Thue_Nha_Scan.png",
        content_text=page_result.full_text,
        page_number=1,
        metadata={"width": page_result.width, "height": page_result.height}
    )
    ingest_res = rag_pipeline.ingest_document(ingest_req)
    print(f"  [+] Da nap thanh cong: {ingest_res.total_chunks_created} chunks vao ChromaDB!")

    # 5. Hỏi đáp dựa trên tài liệu vừa OCR
    test_questions = [
        "Thời hạn thuê nhà trong bao lâu?",
        "Bên A cho thuê căn hộ có số CCCD là bao nhiêu?",
        "Giá thuê nhà mỗi tháng và số tiền đặt cọc là bao nhiêu?",
        "Thời tiết Hà Nội hôm nay thế nào?"  # Thử nghiệm câu hỏi ngoài phạm vi
    ]

    for q in test_questions:
        print(f"\n[?] HOI: \"{q}\"")
        q_req = RagQueryRequest(query=q, top_k=2)
        q_res = rag_pipeline.query(q_req)

        print(f"[*] TRA LOI:\n{q_res.answer}")
        if q_res.sources:
            src = q_res.sources[0]
            print(f"\n📌 Nguon trich dan: Tệp '{src.source_file}' | Do khop: {src.relevance_score * 100:.1f}%")
        print("-" * 70)


if __name__ == "__main__":
    main()
