"""
Automated Test Suite cho FastAPI Endpoints (TestClient).

Kiểm tra:
1. Root endpoint / & Health check /health
2. OCR extraction endpoint /api/v1/ocr/extract
3. RAG ingest & query endpoints (/api/v1/rag/ingest, /api/v1/rag/query, /api/v1/rag/stats)
4. Documents listing endpoint /api/v1/documents
"""

import sys
import os
import io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_and_health():
    # 1. Test Root
    res_root = client.get("/")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert data_root["status"] == "online"
    assert "/docs" in data_root["swagger_ui"]
    print("[PASS] test_root_and_health: GET / passed.")

    # 2. Test Health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"
    print("[PASS] test_root_and_health: GET /health passed.")


def test_rag_endpoints():
    # 1. Test Ingest
    ingest_payload = {
        "document_id": "api_test_doc_01",
        "filename": "chinh_sach_bao_hanh.pdf",
        "content_text": "Thời hạn bảo hành thiết bị là 12 tháng kể từ ngày giao hàng. Đổi mới trong 30 ngày nếu có lỗi của nhà sản xuất.",
        "page_number": 1,
        "metadata": {"type": "warranty"}
    }
    res_ingest = client.post("/api/v1/rag/ingest", json=ingest_payload)
    assert res_ingest.status_code == 200
    data_ingest = res_ingest.json()
    assert data_ingest["status"] == "success"
    assert data_ingest["total_chunks_created"] >= 1
    print("[PASS] test_rag_endpoints: POST /api/v1/rag/ingest passed.")

    # 2. Test Stats
    res_stats = client.get("/api/v1/rag/stats")
    assert res_stats.status_code == 200
    data_stats = res_stats.json()
    assert data_stats["total_chunks_indexed"] >= 1
    print("[PASS] test_rag_endpoints: GET /api/v1/rag/stats passed.")

    # 3. Test Query
    query_payload = {
        "query": "Thời hạn bảo hành thiết bị là bao lâu?",
        "top_k": 2,
        "use_hybrid_search": True
    }
    res_query = client.post("/api/v1/rag/query", json=query_payload)
    assert res_query.status_code == 200
    data_query = res_query.json()
    assert len(data_query["sources"]) > 0
    assert "chinh_sach_bao_hanh.pdf" in data_query["sources"][0]["source_file"]
    print("[PASS] test_rag_endpoints: POST /api/v1/rag/query passed.")


def test_ocr_extract_endpoint():
    # Tạo một file ảnh giả lập dạng byte stream
    fake_image_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100
    files = {
        "file": ("hop_dong_scan.png", io.BytesIO(fake_image_content), "image/png")
    }

    res_ocr = client.post("/api/v1/ocr/extract", files=files)
    assert res_ocr.status_code == 200
    data_ocr = res_ocr.json()
    assert data_ocr["status"] == "success"
    assert len(data_ocr["pages"]) == 1
    assert len(data_ocr["pages"][0]["lines"]) > 0

    # Kiểm tra cấu trúc Bounding Box chuẩn hóa
    first_line = data_ocr["pages"][0]["lines"][0]
    assert "bbox" in first_line
    assert first_line["bbox"]["normalized"] is not None
    assert len(first_line["bbox"]["normalized"]) == 4
    print("[PASS] test_ocr_extract_endpoint: POST /api/v1/ocr/extract passed.")


def test_documents_endpoint():
    res_docs = client.get("/api/v1/documents")
    assert res_docs.status_code == 200
    assert isinstance(res_docs.json(), list)
    print("[PASS] test_documents_endpoint: GET /api/v1/documents passed.")


if __name__ == "__main__":
    test_root_and_health()
    test_rag_endpoints()
    test_ocr_extract_endpoint()
    test_documents_endpoint()
    print("\n[SUCCESS] ALL FASTAPI API TESTS PASSED!")
