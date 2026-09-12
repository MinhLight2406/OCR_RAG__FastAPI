"""
Module Nhúng Văn Bản Thành Vector (Embedding Engine).

MỤC ĐÍCH HỌC TẬP:
- Hiểu cách ánh xạ văn bản thành vector số học nhiều chiều (Dense Vector).
- Cung cấp nhiều lựa chọn:
  1. LightweightEmbedding: Chạy thuần thuật toán toán học (TF-IDF + Hashing + L2 Normalization), không cần tải model nặng hàng GB, cực tốt để học tập & chạy thử ngay lập tức.
  2. LocalSentenceTransformerEmbedding: Dùng mô hình Deep Learning chuẩn (BAAI/bge-m3, all-MiniLM-L6-v2).
  3. GeminiAPIEmbedding: Gọi Google Gemini Embedding API.
"""

from abc import ABC, abstractmethod
from typing import List
import math
import hashlib
import re


class BaseEmbedding(ABC):
    """Giao diện chuẩn cho tất cả các mô hình Embedding."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Biến đổi 1 đoạn văn bản thành 1 vector số học."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Biến đổi hàng loạt đoạn văn bản thành danh sách các vector."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Số chiều của không gian vector."""
        pass


class LightweightEmbedding(BaseEmbedding):
    """
    Mô hình Embedding nhẹ dùng giải thuật Hash + N-gram + L2 Normalization.
    
    Ưu điểm học tập:
    - Chạy hoàn toàn bằng Python thuần, 0 giây chờ đợi, không tốn RAM/VRAM.
    - Giữ trọn vẹn bản chất toán học: Vector chuẩn hóa có độ dài L2 = 1.0,
      tính Dot Product ra chính xác Cosine Similarity.
    """

    def __init__(self, dimension: int = 256):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self._dim

        vector = [0.0] * self._dim

        # 1. Tính tần suất từ đơn (Unigram)
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            vector[idx] += 1.0

        # 2. N-gram 2 từ liên tiếp (Bigram) với trọng số cao hơn để bắt ngữ cảnh
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            h = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            vector[idx] += 2.0

        # 3. Chuẩn hóa L2 (L2 Normalization): ||V|| = 1.0
        # Đảm bảo vector nằm trên mặt cầu đơn vị để Cosine Similarity = Dot Product trong dải [0.0, 1.0]
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 1e-9:
            vector = [x / norm for x in vector]

        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def _tokenize(self, text: str) -> List[str]:
        cleaned = text.lower()
        return [w for w in re.split(r"[^\w\d_]+", cleaned) if w]


class LocalSentenceTransformerEmbedding(BaseEmbedding):
    """Mô hình Embedding chuyên sâu sử dụng thư viện sentence-transformers (Deep Learning)."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "Vui lòng cài đặt: pip install sentence-transformers để dùng Local Deep Learning Embedding!"
                )

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        self._load_model()
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]
