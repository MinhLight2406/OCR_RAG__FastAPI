"""
Quản lý Vector Database (ChromaDB Vector Store).

MỤC ĐÍCH HỌC TẬP:
- Hiểu cách lưu trữ và lập chỉ mục Vector cùng Metadata trong Vector Database.
- Hiểu cách thực hiện phép truy vấn Top-K Nearest Neighbors (k lân cận gần nhất).
- Hỗ trợ lưu trữ bền vững (Persistent Storage trên đĩa) và tự động fallback nếu chưa cài đặt thư viện ngoài.
"""

from typing import List, Dict, Any, Optional
import json
import math
from pathlib import Path
from app.models.rag_models import DocumentChunk, SourceCitation
from app.core.config import settings


class InMemoryVectorDB:
    """Vector Store thuần Python trong bộ nhớ (dùng để học tập hoặc khi chưa cài ChromaDB)."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def query(
        self,
        query_vector: List[float],
        top_k: int = 4,
        filter_document_id: Optional[str] = None
    ) -> List[SourceCitation]:
        scores = []
        for i, emb in enumerate(self.embeddings):
            chunk = self.chunks[i]
            if filter_document_id and chunk.document_id != filter_document_id:
                continue

            # Tính Cosine Similarity (Dot Product vì vector đã chuẩn hóa L2)
            dot_product = sum(a * b for a, b in zip(query_vector, emb))
            sim_score = max(0.0, min(1.0, dot_product))  # Lấy trực tiếp độ tương đồng Cosine dương
            scores.append((sim_score, chunk))

        # Sắp xếp giảm dần theo điểm tương đồng
        scores.sort(key=lambda x: x[0], reverse=True)
        top_results = scores[:top_k]

        citations = []
        for score, chunk in top_results:
            citations.append(
                SourceCitation(
                    document_id=chunk.document_id,
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text_snippet=chunk.text,
                    relevance_score=round(score, 4),
                    bbox=chunk.bbox
                )
            )
        return citations

    def count(self) -> int:
        return len(self.chunks)


class ChromaVectorStore:
    """Lớp giao tiếp chính với ChromaDB Persistent Vector Store."""

    def __init__(self, persist_directory: Path = settings.VECTOR_DB_DIR, collection_name: str = settings.CHROMA_COLLECTION_NAME):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.fallback_db = InMemoryVectorDB()
        self._init_db()

    def _init_db(self):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            # Fallback sang InMemoryVectorDB nếu chưa cài chromadb
            self.collection = None

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Lưu trữ danh sách chunk và vector nhúng vào database."""
        if not chunks:
            return

        if self.collection is not None:
            ids = [c.chunk_id for c in chunks]
            documents = [c.text for c in chunks]
            metadatas = [
                {
                    "document_id": c.document_id,
                    "source_file": c.source_file,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "bbox_json": json.dumps(c.bbox) if c.bbox else ""
                }
                for c in chunks
            ]
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            self.fallback_db.add_chunks(chunks, embeddings)

    def query(
        self,
        query_vector: List[float],
        top_k: int = 4,
        filter_document_id: Optional[str] = None
    ) -> List[SourceCitation]:
        """Truy vấn Top-K đoạn tài liệu có độ tương đồng Cosine cao nhất."""
        if self.collection is not None:
            where_clause = {"document_id": filter_document_id} if filter_document_id else None
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            citations = []
            if results and results.get("documents") and len(results["documents"][0]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0]

                for doc_text, meta, dist in zip(docs, metas, distances):
                    # Chroma tính khoảng cách cosine distance (1 - cosine_similarity)
                    similarity = max(0.0, 1.0 - float(dist))
                    bbox_str = meta.get("bbox_json", "")
                    bbox = json.loads(bbox_str) if bbox_str else None

                    citations.append(
                        SourceCitation(
                            document_id=meta.get("document_id", "unknown"),
                            source_file=meta.get("source_file", "unknown"),
                            page_number=int(meta.get("page_number", 1)),
                            chunk_index=int(meta.get("chunk_index", 0)),
                            text_snippet=doc_text,
                            relevance_score=round(similarity, 4),
                            bbox=bbox
                        )
                    )
            return citations
        else:
            return self.fallback_db.query(query_vector, top_k, filter_document_id)

    def count(self) -> int:
        if self.collection is not None:
            return self.collection.count()
        return self.fallback_db.count()
