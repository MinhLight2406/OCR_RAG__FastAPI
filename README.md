# OCR + RAG + FastAPI System

Hệ thống xử lý và trích xuất thông tin tài liệu thông minh kết hợp Nhận diện chữ qua ảnh (OCR), Truy xuất tăng cường cho mô hình ngôn ngữ lớn (RAG) và phục vụ qua FastAPI.

---

## 📁 Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
OCR_RAG_FastAPI/
├── ai-worker-python/           # Dịch vụ lõi AI & FastAPI (OCR, RAG, API)
│   ├── app/
│   │   ├── api/                # API Endpoints (v1/ocr, v1/rag, v1/documents)
│   │   ├── core/               # Cấu hình hệ thống, logging, settings
│   │   ├── db/                 # Kết nối Vector Database (Chroma / Qdrant)
│   │   ├── models/             # Pydantic schemas (Request/Response models)
│   │   ├── services/           # Nghiệp vụ xử lý chính:
│   │   │   ├── ocr/            # Engine OCR (PaddleOCR, Tesseract, VLM)
│   │   │   ├── rag/            # Pipeline RAG (Chunking, Embedding, Retriever, LLM)
│   │   │   └── pipeline/       # Điều phối quy trình xử lý end-to-end
│   │   ├── utils/              # Tiền xử lý ảnh (OpenCV), làm sạch văn bản
│   │   └── main.py             # Điểm khởi động ứng dụng FastAPI
│   ├── tests/                  # Unit tests & Integration tests
│   ├── Dockerfile              # Đóng gói container cho AI Worker
│   └── requirements.txt        # Danh sách các thư viện Python
│
├── backend-java/               # (Tùy chọn) Backend Gateway / Quản lý nghiệp vụ doanh nghiệp (Spring Boot)
│   └── src/                    # Mã nguồn Java
│
├── frontend/                   # Giao diện người dùng Web (Upload tài liệu & Chat)
│   └── src/                    # Mã nguồn giao diện
│
├── storage/                    # Nơi lưu trữ dữ liệu cục bộ
│   ├── raw_documents/          # File tài liệu gốc (ảnh, PDF) tải lên
│   ├── processed_images/       # Ảnh sau khi tiền xử lý (deskew, binarized)
│   ├── vector_db/              # Dữ liệu nhúng của Vector Database (ChromaDB)
│   └── exports/                # Kết quả xuất dữ liệu (JSON, CSV, Markdown)
│
├── research/                   # Thử nghiệm & Nghiên cứu mô hình
│   ├── notebooks/              # Jupyter Notebooks thử nghiệm OCR & RAG
│   ├── experiments/            # Ghi chép kết quả benchmark mô hình
│   └── sample_data/            # Dữ liệu ảnh mẫu để thử nghiệm
│
├── learning/                   # 📚 TÀI LIỆU HỌC TẬP & KIẾN TRÚC CHUYÊN SÂU
│   ├── 01-tong-quan-kien-truc-he-thong.md
│   ├── 02-so-sanh-va-lua-chon-cong-nghe-ocr.md
│   ├── 03-chuyen-sau-rag-cho-tai-lieu-ocr.md
│   ├── 04-kien-truc-fastapi-va-backend.md
│   ├── 05-lo-trinh-trien-khai-tung-buoc.md
│   └── 06-giao-dien-web-frontend-boundingbox-va-chat.md
│
├── docker-compose.yml          # Cấu hình chạy toàn bộ hệ thống bằng Docker
└── README.md                   # Tài liệu hướng dẫn chung
```

---

## 📚 Thư Mục Học Tập (`learning/`)

Để hiểu rõ **"Tại sao phải làm vậy?"**, **"Còn cách nào khác không?"**, và **"Ưu nhược điểm từng giải pháp là gì?"**, hãy xem các tài liệu hướng dẫn chuyên sâu:

1. [01. Tổng quan kiến trúc hệ thống](learning/01-tong-quan-kien-truc-he-thong.md)
2. [02. So sánh và lựa chọn công nghệ OCR & Bounding Box](learning/02-so-sanh-va-lua-chon-cong-nghe-ocr.md)
3. [03. Chuyên sâu RAG cho tài liệu OCR](learning/03-chuyen-sau-rag-cho-tai-lieu-ocr.md)
4. [04. Kiến trúc FastAPI và mô hình Backend](learning/04-kien-truc-fastapi-va-backend.md)
5. [05. Lộ trình triển khai từng bước](learning/05-lo-trinh-trien-khai-tung-buoc.md)
6. [06. Giao diện Web Frontend, Bounding Box tương tác & Chat RAG](learning/06-giao-dien-web-frontend-boundingbox-va-chat.md)

---

## 🚀 Khởi Động Nhanh (Quick Start)

### 1. Khởi chạy FastAPI Server:
```powershell
cd ai-worker-python
.\.venv\Scripts\uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Trải nghiệm hệ thống:
* 💻 **Giao diện Web Trực quan (Drag & Drop, Bounding Box, Chat):** Truy cập [http://localhost:8000/app](http://localhost:8000/app)
* 📑 **Tài liệu API tương tác (Swagger UI):** Truy cập [http://localhost:8000/docs](http://localhost:8000/docs)
