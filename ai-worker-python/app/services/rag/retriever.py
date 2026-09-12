"""
Module Truy Xuất Nâng Cao (Advanced Retriever & Hybrid Search).

MỤC ĐÍCH HỌC TẬP:
- Hiểu giải thuật Hybrid Search kết hợp:
  1. Dense Vector Search: Bắt ý nghĩa tương đồng bằng Embedding.
  2. Sparse Keyword Search (BM25): Bắt chính xác từ khóa, số liệu, mã định danh.
- Hiểu giải thuật Reciprocal Rank Fusion (RRF) để gộp kết quả từ 2 danh sách xếp hạng khác nhau.
"""

from typing import List, Dict, Optional
import math
import re
from app.models.rag_models import SourceCitation
from app.services.rag.embedding import BaseEmbedding
from app.db.chroma_client import ChromaVectorStore


class SimpleBM25:
    """
    Thuật toán BM25 (Best Matching 25) chấm điểm khớp từ khóa chính xác.
    Dùng để tìm kiếm chính xác các mã hóa đơn, số tiền, tên riêng mà Vector Search có thể bỏ sót.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.corpus_size: int = 0

    def fit(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return

        self.doc_len = []
        self.doc_freqs = []
        df = {}

        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))
            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freqs.append(freq)

            for t in freq:
                df[t] = df.get(t, 0) + 1

        self.avg_doc_len = sum(self.doc_len) / max(1, self.corpus_size)

        for token, freq in df.items():
            # Công thức chuẩn IDF
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> List[float]:
        tokens = self._tokenize(query)
        scores = []

        for i in range(self.corpus_size):
            score = 0.0
            doc_freq = self.doc_freqs[i]
            d_len = self.doc_len[i]

            for t in tokens:
                if t in doc_freq:
                    tf = doc_freq[t]
                    idf = self.idf.get(t, 0.0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (d_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)
            scores.append(score)

        return scores

    def _tokenize(self, text: str) -> List[str]:
        return [w for w in re.split(r"[^\w\d_]+", text.lower()) if w]


class HybridRetriever:
    """
    Bộ truy xuất kết hợp (Hybrid Search) với thuật toán Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_engine: BaseEmbedding,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        filter_document_id: Optional[str] = None,
        use_hybrid: bool = True
    ) -> List[SourceCitation]:
        """Truy xuất tài liệu theo cơ chế kết hợp Dense Search + BM25 hoặc thuần Dense."""
        # 1. Bước 1: Dense Vector Search
        query_vector = self.embedding_engine.embed_text(query)
        dense_results = self.vector_store.query(
            query_vector=query_vector,
            top_k=top_k * 2,  # Lấy dư ra để tái xếp hạng
            filter_document_id=filter_document_id
        )

        if not use_hybrid or not dense_results:
            return dense_results[:top_k]

        # 2. Bước 2: Sparse BM25 Search trên tập ứng viên lấy về
        corpus = [res.text_snippet for res in dense_results]
        bm25 = SimpleBM25()
        bm25.fit(corpus)
        bm25_scores = bm25.score(query)

        # Tạo bảng xếp hạng BM25
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        # 3. Bước 3: Hợp nhất bằng Reciprocal Rank Fusion (RRF)
        # RRF_Score = 1 / (k + rank_dense) + 1 / (k + rank_bm25)
        fused_scores = {}
        for rank_dense, res in enumerate(dense_results):
            fused_scores[res.chunk_index] = 1.0 / (self.rrf_k + rank_dense + 1)

        for rank_bm25, idx in enumerate(bm25_ranked_indices):
            chunk_idx = dense_results[idx].chunk_index
            fused_scores[chunk_idx] = fused_scores.get(chunk_idx, 0.0) + (1.0 / (self.rrf_k + rank_bm25 + 1))

        # Sắp xếp lại theo điểm RRF tổng hợp
        dense_results.sort(key=lambda x: fused_scores.get(x.chunk_index, 0.0), reverse=True)

        return dense_results[:top_k]
