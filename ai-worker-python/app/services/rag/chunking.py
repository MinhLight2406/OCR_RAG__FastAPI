"""
Module Cắt Đoạn Văn Bản (Text Chunking).

MỤC ĐÍCH HỌC TẬP:
- Hiểu tại sao phải cắt nhỏ văn bản trước khi nhúng (Embed) thành Vector.
- Hiểu thuật toán Recursive Character Chunking: ưu tiên ngắt ở ranh giới tự nhiên (Đoạn văn \n\n -> Dòng \n -> Câu . -> Từ ' ')
- Hiểu cơ chế Chunk Overlap để tránh mất thông tin ở điểm cắt.
"""

from abc import ABC, abstractmethod
from typing import List
from app.models.rag_models import DocumentChunk


class BaseChunker(ABC):
    """Lớp trừu tượng cho tất cả các giải thuật Chunking."""

    @abstractmethod
    def split_text(
        self,
        text: str,
        document_id: str,
        source_file: str,
        page_number: int = 1
    ) -> List[DocumentChunk]:
        pass


class RecursiveCharacterChunker(BaseChunker):
    """
    Thuật toán cắt văn bản đệ quy theo thứ tự ưu tiên các ký tự phân cách (Separators).
    
    Quy tắc hoạt động:
    1. Thử tách văn bản bằng separator đầu tiên (thường là '\n\n' - ngắt đoạn).
    2. Nếu một đoạn vẫn dài hơn `chunk_size`, tiếp tục tách nó bằng separator tiếp theo ('\n' - ngắt dòng).
    3. Cứ tiếp tục với '. ' (ngắt câu), ' ' (ngắt từ), và cuối cùng là từng ký tự ''.
    4. Ghép các đoạn nhỏ lại sao cho độ dài không vượt quá `chunk_size`, đồng thời giữ lại `chunk_overlap` ký tự gối đầu.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size để tránh lặp vô tận!")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]

    def split_text(
        self,
        text: str,
        document_id: str,
        source_file: str,
        page_number: int = 1
    ) -> List[DocumentChunk]:
        """Chia văn bản thô thành danh sách các DocumentChunk chuẩn hóa."""
        cleaned_text = self._clean_whitespace(text)
        if not cleaned_text:
            return []

        raw_chunks = self._recursive_split(cleaned_text, self.separators)
        merged_chunks = self._merge_chunks_with_overlap(raw_chunks)

        document_chunks = []
        for idx, chunk_str in enumerate(merged_chunks):
            chunk_id = f"{document_id}_p{page_number}_c{idx}"
            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_str,
                    document_id=document_id,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_index=idx,
                    extra_metadata={"char_count": len(chunk_str)}
                )
            )

        return document_chunks

    def _clean_whitespace(self, text: str) -> str:
        """Làm sạch các khoảng trắng dư thừa sau khi OCR."""
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join([line for line in lines if line])

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Tách đệ quy dựa trên danh sách separator."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        for s in splits:
            if not s:
                continue
            if len(s) <= self.chunk_size:
                final_chunks.append(s)
            else:
                if new_separators:
                    other_chunks = self._recursive_split(s, new_separators)
                    final_chunks.extend(other_chunks)
                else:
                    final_chunks.append(s)

        return final_chunks

    def _merge_chunks_with_overlap(self, splits: List[str]) -> List[str]:
        """Ghép các mẩu nhỏ lại thành từng chunk có kích thước mong muốn kèm overlap."""
        docs = []
        current_doc = []
        total_len = 0

        for piece in splits:
            piece_len = len(piece)
            if total_len + piece_len > self.chunk_size and current_doc:
                doc_text = " ".join(current_doc).strip()
                if doc_text:
                    docs.append(doc_text)
                
                # Giữ lại một phần cuối cho overlap
                while total_len > self.chunk_overlap and current_doc:
                    removed = current_doc.pop(0)
                    total_len -= (len(removed) + 1)
            
            current_doc.append(piece)
            total_len += piece_len + 1

        if current_doc:
            doc_text = " ".join(current_doc).strip()
            if doc_text:
                docs.append(doc_text)

        return docs
