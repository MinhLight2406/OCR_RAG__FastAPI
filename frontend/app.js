/**
 * DocuMind Frontend Application Logic (OCR & RAG Interaction).
 * 
 * Tính năng chính:
 * 1. Kéo thả tải tệp (Drag-and-Drop) & Preview ảnh tức thì.
 * 2. Gọi API FastAPI Backend (/api/v1/ocr/extract & /api/v1/rag/query).
 * 3. Tính toán và vẽ Bounding Box đè lên ảnh theo tọa độ chuẩn hóa %.
 * 4. Tương tác hai chiều: Hover dòng chữ bôi sáng BBox và ngược lại.
 * 5. Khung Chat hỏi đáp tài liệu thông minh kèm trích dẫn nguồn có thể bấm để xem vùng tài liệu.
 */

const API_BASE_URL = "http://localhost:8000";

// DOM Elements
const backendStatusEl = document.getElementById("backend-status");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const viewerContainer = document.getElementById("viewer-container");
const documentImage = document.getElementById("document-image");
const bboxLayer = document.getElementById("bbox-layer");
const ocrResultsContainer = document.getElementById("ocr-results-container");
const ocrLinesList = document.getElementById("ocr-lines-list");
const linesCountBadge = document.getElementById("lines-count-badge");
const ocrStatsEl = document.getElementById("ocr-stats");
const btnToggleBbox = document.getElementById("btn-toggle-bbox");
const btnRemoveDoc = document.getElementById("btn-remove-doc");

const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const btnSend = document.getElementById("btn-send");
const quickSuggestions = document.getElementById("quick-suggestions");

let currentDocumentId = null;
let currentLines = [];
let isBboxVisible = true;

// ==========================================================================
// 1. Kiểm tra trạng thái Backend (Health Check)
// ==========================================================================
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            backendStatusEl.innerHTML = `<span class="status-dot online"></span> Backend Online`;
            backendStatusEl.classList.add("badge-accent");
        } else {
            throw new Error();
        }
    } catch (e) {
        backendStatusEl.innerHTML = `<span class="status-dot"></span> Backend Offline (Chưa bật)`;
        backendStatusEl.classList.remove("badge-accent");
    }
}

setInterval(checkBackendHealth, 5000);
checkBackendHealth();

// ==========================================================================
// 2. Xử lý Kéo Thả & Tải Tệp (Drag and Drop Upload)
// ==========================================================================
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
    }
});

async function handleFileUpload(file) {
    // 1. Preview ảnh ngay lập tức
    const reader = new FileReader();
    reader.onload = (e) => {
        documentImage.src = e.target.result;
        dropZone.classList.add("hidden");
        viewerContainer.classList.remove("hidden");
        ocrResultsContainer.classList.remove("hidden");
    };
    reader.readAsDataURL(file);

    ocrStatsEl.textContent = "⏳ Đang chạy OCR & Nạp Vector DB...";
    currentDocumentId = "DOC_" + Date.now();

    // 2. Gửi file lên API OCR & Ingest
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Bước A: Chạy OCR để lấy Bounding Boxes
        const ocrRes = await fetch(`${API_BASE_URL}/api/v1/ocr/extract`, {
            method: "POST",
            body: formData
        });

        if (!ocrRes.ok) {
            throw new Error("Lỗi khi gọi API OCR");
        }

        const ocrData = await ocrRes.json();
        const page = ocrData.pages[0];
        currentLines = page.lines;

        // Bước B: Tự động nạp vào Vector Database cho RAG
        await fetch(`${API_BASE_URL}/api/v1/rag/ingest`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                document_id: currentDocumentId,
                filename: file.name,
                content_text: page.full_text,
                page_number: page.page_number
            })
        });

        // 3. Hiển thị Bounding Box & Danh sách dòng
        renderBoundingBoxes(currentLines);
        renderOcrLinesList(currentLines);

        ocrStatsEl.textContent = `✅ Đã OCR: ${file.name} (${page.lines.length} dòng)`;
        linesCountBadge.textContent = `${page.lines.length} dòng`;

        // Kích hoạt khung Chat
        chatInput.disabled = false;
        btnSend.disabled = false;
        quickSuggestions.classList.remove("hidden");
        chatInput.focus();

        appendBotMessage(`Đã bóc tách thành công tài liệu **${file.name}** và nạp vào cơ sở dữ liệu! Bạn có thể đặt câu hỏi ở khung chat bên dưới.`);

    } catch (error) {
        console.error(error);
        ocrStatsEl.textContent = "❌ Lỗi kết nối Backend!";
        alert("Không thể kết nối với Backend tại http://localhost:8000. Hãy đảm bảo bạn đã khởi động server FastAPI!");
    }
}

// ==========================================================================
// 3. Thuật Toán Vẽ Bounding Box Đè Lên Ảnh & Tương Tác Hai Chiều
// ==========================================================================
function renderBoundingBoxes(lines) {
    bboxLayer.innerHTML = "";

    lines.forEach((line) => {
        if (!line.bbox || !line.bbox.normalized) return;

        const [x1, y1, x2, y2] = line.bbox.normalized;
        const boxEl = document.createElement("div");
        boxEl.className = "bbox-box";
        boxEl.id = `bbox-line-${line.line_number}`;

        // Công thức tính vị trí CSS % dựa trên tọa độ chuẩn hóa
        boxEl.style.left = `${x1 * 100}%`;
        boxEl.style.top = `${y1 * 100}%`;
        boxEl.style.width = `${(x2 - x1) * 100}%`;
        boxEl.style.height = `${(y2 - y1) * 100}%`;
        boxEl.title = `Dòng ${line.line_number}: ${line.text} (Độ khớp: ${Math.round(line.confidence * 100)}%)`;

        // Tương tác hover: Khi rê chuột vào BBox trên ảnh -> Highlight dòng chữ bên dưới
        boxEl.addEventListener("mouseenter", () => highlightLine(line.line_number));
        boxEl.addEventListener("mouseleave", () => unhighlightLine(line.line_number));

        bboxLayer.appendChild(boxEl);
    });
}

function renderOcrLinesList(lines) {
    ocrLinesList.innerHTML = "";

    lines.forEach((line) => {
        const item = document.createElement("div");
        item.className = "ocr-line-item";
        item.id = `ocr-line-item-${line.line_number}`;
        item.innerHTML = `
            <span class="line-num">#${line.line_number}</span>
            <span class="line-text">${line.text}</span>
            <span class="line-conf">${Math.round(line.confidence * 100)}%</span>
        `;

        // Tương tác hover: Khi rê chuột vào dòng chữ -> Highlight BBox trên ảnh
        item.addEventListener("mouseenter", () => highlightLine(line.line_number));
        item.addEventListener("mouseleave", () => unhighlightLine(line.line_number));

        ocrLinesList.appendChild(item);
    });
}

function highlightLine(lineNum) {
    const bboxEl = document.getElementById(`bbox-line-${lineNum}`);
    const itemEl = document.getElementById(`ocr-line-item-${lineNum}`);

    if (bboxEl) bboxEl.classList.add("active");
    if (itemEl) {
        itemEl.classList.add("active");
        itemEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

function unhighlightLine(lineNum) {
    const bboxEl = document.getElementById(`bbox-line-${lineNum}`);
    const itemEl = document.getElementById(`ocr-line-item-${lineNum}`);

    if (bboxEl) bboxEl.classList.remove("active");
    if (itemEl) itemEl.classList.remove("active");
}

// Nút ẩn/hiện Bounding Box
btnToggleBbox.addEventListener("click", () => {
    isBboxVisible = !isBboxVisible;
    bboxLayer.style.display = isBboxVisible ? "block" : "none";
    btnToggleBbox.textContent = isBboxVisible ? "👁️ Ẩn Bounding Box" : "👁️ Hiện Bounding Box";
});

// Nút đổi tài liệu khác
btnRemoveDoc.addEventListener("click", () => {
    dropZone.classList.remove("hidden");
    viewerContainer.classList.add("hidden");
    ocrResultsContainer.classList.add("hidden");
    fileInput.value = "";
    ocrStatsEl.textContent = "Chưa có tài liệu";
    chatInput.disabled = true;
    btnSend.disabled = true;
    quickSuggestions.classList.add("hidden");
});

// ==========================================================================
// 4. Khung Chat Hỏi Đáp RAG (Chatbot & Source Citation)
// ==========================================================================
chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    appendUserMessage(query);
    chatInput.value = "";

    // Hiển thị trạng thái đang suy nghĩ
    const thinkingMessage = appendBotMessage("🤖 Đang tra cứu tài liệu và suy nghĩ...");

    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/rag/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                top_k: 3,
                filter_document_id: currentDocumentId,
                use_hybrid_search: true
            })
        });

        if (!response.ok) {
            throw new Error("Lỗi khi truy vấn RAG");
        }

        const data = await response.json();
        
        // Cập nhật câu trả lời kèm thẻ trích dẫn nguồn
        updateBotMessageWithCitations(thinkingMessage, data.answer, data.sources);

    } catch (err) {
        console.error(err);
        thinkingMessage.querySelector(".message-body").textContent = 
            "❌ Không thể kết nối với RAG API. Vui lòng kiểm tra lại server!";
    }
});

function appendUserMessage(text) {
    const msg = document.createElement("div");
    msg.className = "message user-message";
    msg.innerHTML = `
        <div class="message-avatar">👤</div>
        <div class="message-body">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendBotMessage(text) {
    const msg = document.createElement("div");
    msg.className = "message bot-message";
    msg.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-body">${formatMarkdown(text)}</div>
    `;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
}

function updateBotMessageWithCitations(msgElement, answer, sources) {
    const bodyEl = msgElement.querySelector(".message-body");
    let html = formatMarkdown(answer);

    if (sources && sources.length > 0) {
        html += `<div class="citation-box">`;
        html += `<div style="font-size: 11px; color: var(--text-subtle); margin-bottom: 4px;">📌 Căn cứ trích dẫn nguồn:</div>`;
        sources.forEach((s, idx) => {
            html += `
                <div class="citation-chip" onclick="highlightCitation('${s.document_id}')">
                    📄 ${s.source_file} (Trang ${s.page_number}) • Khớp ${Math.round(s.relevance_score * 100)}%
                </div>
            `;
        });
        html += `</div>`;
    }

    bodyEl.innerHTML = html;
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

window.highlightCitation = function(docId) {
    // Nhấp nháy toàn bộ Bounding Boxes để người dùng chú ý
    document.querySelectorAll(".bbox-box").forEach(b => {
        b.classList.add("active");
        setTimeout(() => b.classList.remove("active"), 1200);
    });
};

// Gợi ý câu hỏi nhanh
document.querySelectorAll(".suggestion-chip").forEach(chip => {
    chip.addEventListener("click", () => {
        chatInput.value = chip.textContent;
        chatForm.dispatchEvent(new Event("submit"));
    });
});

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    return text
        .replace(/\n\n/g, "<br><br>")
        .replace(/\n/g, "<br>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/> "(.*?)"/g, "<blockquote style='border-left: 2px solid var(--accent-amber); padding-left: 8px; color: #fde68a;'>\"$1\"</blockquote>");
}
