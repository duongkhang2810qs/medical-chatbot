import os
from dotenv import load_dotenv
load_dotenv()

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
        print(f"\n[API] Nhận yêu cầu phân tích phiếu từ session: {data.session_id}")
        summary = rag_service.analyze_indicators_with_llm(data.indicators, session_id=data.session_id)
        
        session = chatbot_service.update_session_memory(
            session_id=data.session_id,
            active_report_data=data.indicators,
            report_summary=summary
        )
        
        # [FIX] Đã XÓA 2 dòng gọi session["history"] vì LangGraph đã tự động quản lý bộ nhớ
        # report_summary đã được lưu vào memory và truyền tự động vào Report Node.
        
        # [FIX] Đã gỡ bỏ langfuse_context.flush() ở đây để tránh treo API
        print("[API] Đã phân tích xong, trả kết quả về UI.")
        return {"status": "success", "summary": summary}
    except Exception as e:
        print(f"[API LỖI] {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

# ==========================================
# ENDPOINT 2: CHAT TỰ DO
# ==========================================
@app.options("/api/v1/chat")
async def chat_options(): return {"status": "ok"}

@app.post("/api/v1/chat")
async def chat_bot(msg: ChatMessage):
    try:
        print(f"\n[API] Nhận tin nhắn chat: '{msg.text}'")
        result = chatbot_service.handle_chat(text=msg.text, session_id=msg.session_id)
        
        # [FIX] Đã gỡ bỏ langfuse_context.flush() ở đây
        print("[API] Trả lời chat thành công.")
        return {
            "status": "success",
            "answer": result["answer"],
            "intent": result["intent"]
        }
    except Exception as e:
        print(f"[API LỖI] {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}