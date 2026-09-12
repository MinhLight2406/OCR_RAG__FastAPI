"""
Unit Tests cho Hệ Thống RAG.

Kiểm tra:
1. Thuật toán Recursive Character Chunking
2. Thuật toán Embedding & Chuẩn hóa L2
3. Vector Store Ingest & Query
4. Hybrid Retrieval BM25 + Dense Search
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag.chunking import RecursiveCharacterChunker
from app.services.rag.embedding import LightweightEmbedding
from app.db.chroma_client import InMemoryVectorDB
from app.services.rag.retriever import SimpleBM25
from app.models.rag_models import DocumentIngestRequest, RagQueryRequest
from app.services.rag.rag_pipeline import RAGPipeline


def test_chunking_overlap():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    sample_text = "Câu thứ nhất rất dài để kiểm tra. " * 10
    chunks = chunker.split_text(sample_text, "doc_test", "test.txt", 1)
    assert len(chunks) > 1
    assert chunks[0].document_id == "doc_test"
    print("[PASS] test_chunking_overlap passed.")


def test_embedding_l2_norm():
    embedder = LightweightEmbedding(dimension=64)
    vec = embedder.embed_text("Hợp đồng lao động năm 2026")
    assert len(vec) == 64
    # Kiểm tra chuẩn L2 xấp xỉ 1.0
    l2_norm = sum(x * x for x in vec) ** 0.5
    assert abs(l2_norm - 1.0) < 1e-4
    print("[PASS] test_embedding_l2_norm passed.")


def test_bm25_scoring():
    corpus = [
        "Công ty Cổ phần Trí Tuệ Nhân Tạo Việt Nam",
        "Hợp đồng kinh tế mua bán thiết bị điện tử",
        "Hóa đơn dịch vụ đám mây AWS và GCP"
    ]
    bm25 = SimpleBM25()
    bm25.fit(corpus)
    scores = bm25.score("Hóa đơn đám mây")
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]
    print("[PASS] test_bm25_scoring passed.")


def test_rag_pipeline_end_to_end():
    pipeline = RAGPipeline()
    req = DocumentIngestRequest(
        document_id="doc_unit_test",
        filename="chinh_sach_cong_ty.txt",
        content_text="Nhân viên được nghỉ phép năm 12 ngày hưởng nguyên lương.",
        page_number=1
    )
    res = pipeline.ingest_document(req)
    assert res.total_chunks_created >= 1

    q_req = RagQueryRequest(query="Nhân viên được nghỉ phép bao nhiêu ngày?")
    q_res = pipeline.query(q_req)
    assert len(q_res.sources) > 0
    assert "12 ngày" in q_res.answer or "12 ngày" in q_res.sources[0].text_snippet
    print("[PASS] test_rag_pipeline_end_to_end passed.")


if __name__ == "__main__":
    test_chunking_overlap()
    test_embedding_l2_norm()
    test_bm25_scoring()
    test_rag_pipeline_end_to_end()
    print("\n[SUCCESS] ALL TESTS PASSED!")
