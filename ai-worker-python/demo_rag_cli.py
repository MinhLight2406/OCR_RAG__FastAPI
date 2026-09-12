"""
Script Thực Hành Tương Tác: Khám Phá Toàn Bộ Vòng Đời Của Hệ Thống RAG.

Cách chạy:
    cd ai-worker-python
    python demo_rag_cli.py

Mục tiêu học tập:
1. Quan sát cách văn bản được cắt đoạn (Chunking & Overlap).
2. Quan sát vector embedding nhiều chiều được sinh ra.
3. Thử nghiệm tìm kiếm ngữ nghĩa và xem điểm tương đồng Cosine.
4. Nhận câu trả lời có trích dẫn nguồn chuẩn xác.
"""

import sys
import os

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.rag.rag_pipeline import RAGPipeline
from app.models.rag_models import DocumentIngestRequest, RagQueryRequest


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  [+] {title}")
    print("=" * 70)


def main():
    print_banner("BAT DAU CHUONG TRINH THUC HANH HE THONG RAG")

    # 1. Khởi tạo Pipeline
    print("\n[1] Đang khởi tạo RAG Pipeline (Chunker, Embedding, Vector Store, LLM)...")
    pipeline = RAGPipeline(chunk_size=300, chunk_overlap=60)
    print("    -> Khởi tạo thành công!")

    # 2. Dữ liệu mẫu giả lập kết quả sau khi OCR tài liệu
    sample_doc_1 = (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "Độc lập - Tự do - Hạnh phúc\n\n"
        "HỢP ĐỒNG THUÊ NHÀ Ở NĂM 2026\n\n"
        "Bên A (Bên cho thuê): Ông Nguyễn Văn A, CCCD số 001234567899, cư trú tại Quận Cầu Giấy, Hà Nội.\n"
        "Bên B (Bên thuê): Bà Trần Thị B, CCCD số 009876543211, cư trú tại Quận 1, TP. Hồ Chí Minh.\n\n"
        "ĐIỀU 1: ĐỐI TƯỢNG VÀ THỜI HẠN THUÊ\n"
        "Bên A đồng ý cho Bên B thuê căn hộ số 1204 tại Tòa nhà Sunshine, Hà Nội.\n"
        "Thời hạn thuê là 02 năm (24 tháng), bắt đầu từ ngày 01/01/2026 đến hết ngày 31/12/2027.\n\n"
        "ĐIỀU 2: GIÁ THUÊ VÀ PHƯƠNG THỨC THANH TOÁN\n"
        "Giá thuê nhà cố định là 15.000.000 VNĐ / tháng (Mười lăm triệu đồng một tháng).\n"
        "Bên B thanh toán tiền thuê nhà định kỳ 03 tháng một lần vào ngày mùng 05 của tháng đầu kỳ.\n"
        "Tiền đặt cọc thế chân là 30.000.000 VNĐ (Ba mươi triệu đồng) và sẽ được hoàn trả khi kết thúc hợp đồng."
    )

    sample_doc_2 = (
        "HÓA ĐƠN GIÁ TRỊ GIA TĂNG (VAT INVOICE)\n"
        "Mẫu số: 01GTKT0/001 - Ký hiệu: AA/26E - Số: 009982\n"
        "Đơn vị bán hàng: Công ty Cổ phần Công nghệ AI Việt Nam\n"
        "Mã số thuế: 0109887766\n"
        "Khách hàng: Công ty TNHH Giải pháp Phần mềm ABC\n\n"
        "DANH MỤC HÀNG HÓA DỊCH VỤ:\n"
        "1. Dịch vụ Máy chủ GPU H100 80GB (Thời gian 1 tháng): Đơn giá 45.000.000 VNĐ. Số lượng: 02. Thành tiền: 90.000.000 VNĐ.\n"
        "2. Bản quyền Mô hình OCR Tiếng Việt Enterprise: Đơn giá 25.000.000 VNĐ. Số lượng: 01. Thành tiền: 25.000.000 VNĐ.\n\n"
        "Cộng tiền hàng: 115.000.000 VNĐ.\n"
        "Thuế suất GTGT (10%): 11.500.000 VNĐ.\n"
        "Tổng cộng tiền thanh toán: 126.500.000 VNĐ (Một trăm hai mươi sáu triệu năm trăm nghìn đồng)."
    )

    # 3. Nạp tài liệu vào Vector DB
    print_banner("GIAI ĐOẠN 1: NẠP TÀI LIỆU VÀ TẠO CHUNK & VECTOR EMBEDDING")

    req1 = DocumentIngestRequest(
        document_id="DOC_HOP_DONG_01",
        filename="Hop_Dong_Thue_Nha_2026.pdf",
        content_text=sample_doc_1,
        page_number=1
    )
    res1 = pipeline.ingest_document(req1)
    print(f"  [+] Đã nạp: {res1.filename} -> Tạo ra {res1.total_chunks_created} Chunks.")

    req2 = DocumentIngestRequest(
        document_id="DOC_HOA_DON_02",
        filename="Hoa_Don_VAT_AI_Service.pdf",
        content_text=sample_doc_2,
        page_number=1
    )
    res2 = pipeline.ingest_document(req2)
    print(f"  [+] Đã nạp: {res2.filename} -> Tạo ra {res2.total_chunks_created} Chunks.")
    print(f"\n  👉 Tổng số Chunks hiện có trong Vector DB: {pipeline.vector_store.count()} Chunks.")

    # 4. Thử nghiệm Truy vấn & Hỏi đáp (RAG Query)
    print_banner("GIAI ĐOẠN 2: TRUY VẤN VÀ HỎI ĐÁP VỚI HỆ THỐNG RAG")

    test_queries = [
        "Giá thuê nhà mỗi tháng là bao nhiêu và thời hạn thuê trong bao lâu?",
        "Tổng tiền thanh toán trên hóa đơn VAT là bao nhiêu tiền?",
        "Tiền đặt cọc thuê nhà là bao nhiêu và khi nào được trả lại?",
        "Thời tiết Hà Nội hôm nay thế nào?"  # Câu hỏi ngoài phạm vi tài liệu để test chống ảo giác
    ]

    for q in test_queries:
        print(f"\n[?] CÂU HỎI: \"{q}\"")
        query_req = RagQueryRequest(query=q, top_k=2, use_hybrid_search=True)
        resp = pipeline.query(query_req)

        print(f"[*] Thời gian xử lý: {resp.processing_time_ms} ms")
        print(f"[*] CÂU TRẢ LỜI:\n{resp.answer}")
        print("\n[*] TRÍCH DẪN NGUỒN (SOURCES):")
        for s in resp.sources:
            print(f"   - Tệp: {s.source_file} (Trang {s.page_number}) | Độ tương đồng: {s.relevance_score * 100:.1f}%")
            print(f"     Nội dung: \"{s.text_snippet[:90]}...\"")
        print("-" * 70)


if __name__ == "__main__":
    main()
