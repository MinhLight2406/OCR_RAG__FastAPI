"""
Module Tạo Sinh Câu Trả Lời (LLM Generation & Prompt Engineering).

MỤC ĐÍCH HỌC TẬP:
- Hiểu cách cấu trúc System Prompt để ép buộc LLM chỉ trả lời dựa trên Context (chống Hallucination).
- Hiểu cách format trích dẫn nguồn (Source Citation) chuẩn mực.
- Hỗ trợ kết nối Google Gemini API và có cơ chế fallback thông minh khi chạy offline.
"""

from typing import List
from app.models.rag_models import SourceCitation
from app.core.config import settings


class PromptBuilder:
    """Xây dựng Prompt chặt chẽ cho LLM."""

    @staticmethod
    def build_system_prompt() -> str:
        return (
            "Bạn là trợ lý AI chuyên gia phân tích và trích xuất thông tin tài liệu OCR.\n"
            "Nhiệm vụ của bạn là đọc kỹ phần NGỮ CẢNH (CONTEXT) được trích xuất từ tài liệu "
            "và trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn, trung thực.\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. CHỈ sử dụng thông tin có trong phần NGỮ CẢNH để trả lời.\n"
            "2. Nếu phần NGỮ CẢNH không chứa đủ thông tin để trả lời, bạn BẮT BUỘC phải nói rõ: "
            "'Tài liệu không có đủ thông tin để trả lời câu hỏi này.' Tuyệt đối không tự suy diễn.\n"
            "3. Cuối câu trả lời, hãy liệt kê rõ thông tin được trích dẫn từ [Trang số X] của tệp tài liệu nào."
        )

    @staticmethod
    def build_user_prompt(query: str, citations: List[SourceCitation]) -> str:
        context_blocks = []
        for i, cit in enumerate(citations, 1):
            block = (
                f"--- [Đoạn trích {i}] (Tệp: {cit.source_file} | Trang: {cit.page_number}) ---\n"
                f"{cit.text_snippet}\n"
            )
            context_blocks.append(block)

        full_context = "\n".join(context_blocks) if context_blocks else "Không tìm thấy đoạn văn bản liên quan nào."

        return (
            f"=== NGỮ CẢNH TÀI LIỆU (CONTEXT) ===\n"
            f"{full_context}\n"
            f"====================================\n\n"
            f"CÂU HỎI: {query}\n\n"
            f"CÂU TRẢ LỜI CỦA BẠN:"
        )


class LLMGenerator:
    """Bộ tạo sinh câu trả lời sử dụng LLM."""

    def __init__(self, api_key: str = settings.GEMINI_API_KEY, model_name: str = settings.GEMINI_MODEL_NAME):
        self.api_key = api_key
        self.model_name = model_name

    def generate_answer(self, query: str, citations: List[SourceCitation]) -> str:
        """Gửi prompt tới LLM hoặc tổng hợp offline nếu chưa có API Key."""
        # 1. Thử gọi Google Gemini nếu có API Key
        if self.api_key:
            try:
                import google.generativeai as genai
                system_prompt = PromptBuilder.build_system_prompt()
                user_prompt = PromptBuilder.build_user_prompt(query, citations)
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_prompt
                )
                response = model.generate_content(user_prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception:
                # Log lỗi nếu gọi API thất bại và chuyển sang fallback
                pass

        # 2. Fallback ngoại tuyến (Dành cho việc học tập & demo nhanh không cần internet)
        return self._offline_synthesize(query, citations)

    def _offline_synthesize(self, query: str, citations: List[SourceCitation]) -> str:
        """Tổng hợp thông tin trực tiếp từ các đoạn văn bản tương đồng cao nhất."""
        if not citations:
            return "Tài liệu được cung cấp không có đủ thông tin để trả lời câu hỏi này."

        best = citations[0]
        # Kiểm tra ngưỡng tương đồng tối thiểu (Relevance Threshold) để chống trả lời bừa khi câu hỏi lạc đề
        if best.relevance_score < 0.15:
            return "Tài liệu được cung cấp không chứa thông tin liên quan đến câu hỏi này."

        summary = (
            f"Dựa trên tài liệu '{best.source_file}' (Trang {best.page_number}):\n\n"
            f"> \"{best.text_snippet}\"\n\n"
            f"📌 Trích dẫn nguồn: Tệp '{best.source_file}', Trang {best.page_number} "
            f"(Độ khớp: {best.relevance_score * 100:.1f}%)."
        )
        return summary
