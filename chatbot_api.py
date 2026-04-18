from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback
import chatbot_service
import rag_service

app = FastAPI(title="Unified Medical Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConfirmedData(BaseModel):
    indicators: list
    session_id: str

class ChatMessage(BaseModel):
    text: str
    session_id: str

# ==========================================
# ENDPOINT 1: PHÂN TÍCH & NẠP VÀO BỘ NHỚ
# ==========================================
@app.options("/api/v1/analyze")
async def analyze_options(): return {"status": "ok"}

@app.post("/api/v1/analyze")
async def analyze_report(data: ConfirmedData):
    try:
        # 1. Gọi RAG Service Phân tích chuyên sâu (Bản V11)
        summary = rag_service.analyze_indicators_with_llm(data.indicators)
        
        # 2. LƯU BỘ NHỚ CHO CHATBOT:
        # Lưu cả Bảng dữ liệu thô và Bản tóm tắt vào phiên của User này
        chatbot_service.update_session_memory(
            session_id=data.session_id,
            active_report_data=data.indicators,
            report_summary=summary
        )
        
        return {"status": "success", "summary": summary}
    except Exception as e:
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

# ==========================================
# ENDPOINT 2: CHAT TỰ DO
# ==========================================
@app.options("/api/v1/chat")
async def chat_options(): return {"status": "ok"}

@app.post("/api/v1/chat")
async def chat_bot(msg: ChatMessage):
    try:
        # Gửi Text và ID. Chatbot sẽ tự lục lọi bộ nhớ (Session) để trả lời.
        result = chatbot_service.handle_chat(text=msg.text, session_id=msg.session_id)
        
        return {
            "status": "success",
            "answer": result["answer"],
            "intent": result["intent"]
        }
    except Exception as e:
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}