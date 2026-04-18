# Medical Chatbot

Medical Chatbot là dự án hỗ trợ **đọc phiếu xét nghiệm máu (CBC)** từ ảnh, chuyển dữ liệu bằng **OCR**, cho phép người dùng **xác nhận/chỉnh sửa các chỉ số**, sau đó dùng **RAG + LLM** để phân tích bất thường và trả lời câu hỏi tiếp theo theo ngữ cảnh của đúng phiếu xét nghiệm đó.

## Tính năng chính

- Tải ảnh phiếu xét nghiệm CBC từ giao diện web.
- OCR tự động bóc tách bảng chỉ số từ ảnh.
- Cho phép người dùng chỉnh sửa lại chỉ số, khoảng tham chiếu và trạng thái trước khi gửi phân tích.
- Phân tích các chỉ số bất thường bằng RAG + LLM.
- Hỗ trợ chat tiếp theo theo ngữ cảnh của chính report đã tải lên.
- Lưu session hội thoại theo `session_id` để follow-up liên tục.
- Tự động tạo collection `cbc` trong Qdrant và tự nạp knowledge base nếu chưa có dữ liệu.

---

## Kiến trúc hệ thống

Dự án hiện chạy theo kiến trúc **2 backend service + 1 frontend tĩnh + 1 vector database**:

1. **OCR Service** (`ocr_api.py`)  
   Chạy cổng **8001**, nhận ảnh xét nghiệm và trả về bảng dữ liệu OCR.

2. **Chatbot Service** (`chatbot_api.py`)  
   Chạy cổng **8002**, có 2 nhiệm vụ:
   - Phân tích report đã xác nhận (`/api/v1/analyze`)
   - Trả lời chat tự do hoặc chat follow-up (`/api/v1/chat`)

3. **Frontend** (`index.html`)  
   Giao diện web gọi trực tiếp:
   - OCR API tại `http://127.0.0.1:8001/api/v1`
   - Chat API tại `http://127.0.0.1:8002/api/v1`

4. **Qdrant**  
   Chạy cổng **6333**, dùng làm vector database cho phần RAG.

---

## Cấu trúc file chính

```bash
.
├── index.html              # Frontend giao diện người dùng
├── ocr_api.py              # FastAPI cho OCR Service
├── ocr_service.py          # Pipeline OCR + xử lý bảng CBC
├── chatbot_api.py          # FastAPI cho Chatbot Service
├── chatbot_service.py      # Điều phối hội thoại, session memory, intent
├── rag_service.py          # RAG, embedding, Qdrant, gọi LLM
├── .env                    # Biến môi trường API key
├── demo_cases_all_clean.json
├── cbc_kb_v8.json          # Knowledge base để nạp vào Qdrant
├── ontology_tests.json
├── ontology_units.json
└── outputs/                # Ảnh debug / JSON OCR xuất ra
```

---

## Luồng hoạt động

### 1. OCR phiếu xét nghiệm

- Người dùng tải ảnh từ giao diện web.
- Frontend gửi file tới `POST /api/v1/extract` của OCR Service.
- `ocr_api.py` lưu tạm ảnh, gọi `ocr_service.run_end_to_end_pipeline(...)` để xử lý, sau đó trả về `ocr_table`.

### 2. Người dùng xác nhận dữ liệu

- Dữ liệu OCR được hiển thị thành bảng ở frontend.
- Người dùng có thể chỉnh tên xét nghiệm, giá trị, đơn vị, khoảng tham chiếu.
- Frontend tự tính lại trạng thái `High / Low / Normal` sau khi chỉnh sửa.

### 3. Phân tích y khoa

- Sau khi xác nhận, frontend gửi danh sách chỉ số tới `POST /api/v1/analyze`.
- Chatbot Service gọi `rag_service.analyze_indicators_with_llm(...)` để:
  - lấy các chỉ số bất thường,
  - match pattern,
  - truy vấn Qdrant,
  - tạo prompt,
  - gọi LLM,
  - trả về bản tóm tắt y khoa.
- Đồng thời service lưu `active_report` và `active_report_summary` vào session.

### 4. Chat follow-up

- Khi người dùng hỏi tiếp, frontend gọi `POST /api/v1/chat`.
- Chatbot Service phân loại intent:
  - `general_chat`
  - `medical_knowledge`
  - `report_followup`
- Nếu là follow-up về phiếu xét nghiệm, service sẽ dùng lại:
  - report hiện tại,
  - lịch sử chat,
  - evidence từ Qdrant,
  để tạo câu trả lời đúng ngữ cảnh.

---

## Công nghệ sử dụng

### Backend
- FastAPI
- Uvicorn
- Pydantic

### OCR
- PaddleOCR
- VietOCR
- OpenCV
- PIL
- NumPy

### RAG / NLP
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Qdrant
- Google GenAI
- OpenRouter / DeepSeek

### Frontend
- HTML
- CSS
- JavaScript thuần

---

## Yêu cầu môi trường

- Python 3.10+ khuyến nghị
- Docker (để chạy Qdrant)
- GPU NVIDIA + CUDA phù hợp nếu muốn chạy bản OCR/PyTorch/Paddle GPU

---

## Cài đặt dependencies

Ưu tiên dùng file `requirements.txt`:

```bash
pip install -r requirements.txt
```

Sau đó cài riêng các gói GPU theo đúng **CUDA của máy**.

Ví dụ với **CUDA 11.8**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

> Gợi ý: nên kiểm tra đúng phiên bản CUDA trước khi cài. Nếu máy bạn không dùng CUDA 11.8 thì đổi `cu118` sang bản phù hợp. Nếu chạy CPU-only thì có thể bỏ qua 2 lệnh GPU này.

---

## Cấu hình `.env`

Tạo file `.env` cùng cấp với `rag_service.py`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Trong code hiện tại, hệ thống sẽ ưu tiên gọi **DeepSeek qua OpenRouter**, nếu lỗi thì fallback sang **Gemini**.

---

## Chạy dự án

### 1. Chạy Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Chạy OCR Service

```bash
uvicorn ocr_api:app --port 8001 --reload
```

### 3. Chạy Chatbot Service

```bash
uvicorn chatbot_api:app --port 8002 --reload
```

### 4. Mở frontend

Có thể mở `index.html` trực tiếp bằng trình duyệt, hoặc chạy một static server local.  
Do frontend đang hard-code API base là:

- `http://127.0.0.1:8001/api/v1`
- `http://127.0.0.1:8002/api/v1`

nên khi deploy sang internet bạn cần sửa lại các URL này.

Ví dụ chạy local server nhanh bằng Python:

```bash
python -m http.server 5500
```

Sau đó truy cập:

```text
http://127.0.0.1:5500/index.html
```

---

## API chính

### OCR Service

#### `POST /api/v1/extract`
Nhận file ảnh và trả về bảng dữ liệu OCR.

Request:
- `multipart/form-data`
- field: `file`

Response thành công:

```json
{
  "status": "success",
  "ocr_table": [
    {
      "test_name": "WBC",
      "value": 12.5,
      "unit": "10^9/L",
      "ref_range": {
        "ref_min": 4.0,
        "ref_max": 10.0
      },
      "status": "High"
    }
  ]
}
```

#### `POST /api/v1/free-ram`
Giải phóng OCR model khỏi RAM nếu cần.

---

### Chatbot Service

#### `POST /api/v1/analyze`
Nhận dữ liệu chỉ số đã xác nhận và trả về bản phân tích y khoa.

Request:

```json
{
  "indicators": [...],
  "session_id": "sess_xxx"
}
```

#### `POST /api/v1/chat`
Chat tự do hoặc hỏi follow-up theo report hiện tại.

Request:

```json
{
  "text": "Chỉ số WBC của tôi có đáng lo không?",
  "session_id": "sess_xxx"
}
```

---

## Session memory

`chatbot_service.py` đang lưu session bằng biến in-memory:

- `history`
- `active_report`
- `active_report_summary`
- `last_topic`
- `last_medical_context`

Điều này có nghĩa là:

- session sẽ **mất khi restart server**,
- phù hợp để demo/local,
- chưa phù hợp cho production nhiều người dùng.

Nếu muốn production, nên chuyển sang Redis hoặc database.

---

## Một số điểm kỹ thuật đáng chú ý

### OCR pipeline

`ocr_service.py` hiện kết hợp:
- PaddleOCR cho detection/recognition,
- VietOCR cho recognition bổ sung,
- deskew / deshear ảnh,
- ontology cho tên xét nghiệm và đơn vị,
- hậu xử lý để dựng lại bảng CBC sạch hơn.

Ngoài dữ liệu JSON trả về API, service còn ghi thêm ảnh debug và file JSON vào thư mục `outputs/`.

### RAG pipeline

`rag_service.py` hiện:
- load pattern từ `demo_cases_all_clean.json`,
- load knowledge base từ `cbc_kb_v8.json`,
- load embedding model `all-MiniLM-L6-v2`,
- kết nối Qdrant tại `localhost:6333`,
- tự kiểm tra collection `cbc`,
- tự tạo collection nếu chưa có,
- tự nạp vector vào Qdrant nếu collection chưa tồn tại,
- search evidence,
- sinh prompt tiếng Việt theo format y khoa,
- gọi LLM để trả về phân tích cuối cùng.

### Intent routing

Chatbot hiện chia câu hỏi thành 3 nhóm:
- hỏi chung,
- hỏi kiến thức y khoa,
- hỏi follow-up về report hiện tại.

Cách này giúp phần follow-up bám đúng phiếu xét nghiệm mà người dùng vừa upload.

---

## Hạn chế hiện tại

- Session đang lưu in-memory, chưa bền vững.
- Frontend đang hard-code URL local `127.0.0.1`.
- Chưa có cơ chế auth.
- Chưa có Docker Compose tổng thể cho toàn dự án.
- Bước nạp knowledge base vào Qdrant hiện diễn ra khi `rag_service.py` khởi động lần đầu, nên lần chạy đầu có thể mất thêm thời gian để encode vector.

---

## Gợi ý cải tiến

- Tạo `docker-compose.yml` cho toàn bộ stack.
- Chuyển session sang Redis.
- Tách cấu hình API base ra file config hoặc biến môi trường frontend.
- Thêm endpoint health check cho từng service.
- Tách script ingest knowledge base riêng nếu muốn quản lý dữ liệu Qdrant rõ hơn.
- Viết thêm script deploy cho Cloudflare Pages + backend public API.

---

## Cách demo nhanh

1. Chạy Qdrant.
2. Chạy `ocr_api` ở cổng 8001.
3. Chạy `chatbot_api` ở cổng 8002.
4. Mở `index.html`.
5. Upload ảnh phiếu xét nghiệm CBC.
6. Kiểm tra/chỉnh lại bảng OCR.
7. Bấm **Gửi cho Bác sĩ AI**.
8. Tiếp tục đặt câu hỏi như:
   - `WBC của tôi có cao không?`
   - `Kết quả này gợi ý gì?`
   - `Tôi có cần khám thêm không?`

---

## Lưu ý

Dự án này mang tính **hỗ trợ tham khảo**, không thay thế chẩn đoán của bác sĩ.  
Mọi kết luận lâm sàng cần được đối chiếu với triệu chứng, tiền sử bệnh và đánh giá chuyên môn thực tế.
