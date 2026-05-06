import os
from dotenv import load_dotenv
load_dotenv()

import json
from typing import Any, Dict, List, Optional
import rag_service
from langfuse.decorators import observe, langfuse_context

SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "session_id": session_id,
            "user_name": "Bạn",
            "history": [],
            "active_report": None,
            "active_report_summary": None,
            "last_topic": None,
            "last_medical_context": None
        }
    return SESSIONS[session_id]

def update_session_memory(session_id: str, text: str = None, active_report_data: list = None, report_summary: str = None):
    session = get_session(session_id)
    if text:
        session["history"].append({"role": "user", "content": text})
    if active_report_data is not None:
        session["active_report"] = {
            "id": "Web_Upload_Report",
            "data": active_report_data
        }
    if report_summary is not None:
        session["active_report_summary"] = report_summary
    return session

@observe(as_type="span", name="Detect Intent (Semantic)")
def detect_intent(text: str, session: Dict) -> str:
    print("[LOGIC] Bắt đầu Semantic Routing...")
    has_report = session.get("active_report") is not None
    intent = rag_service.semantic_route_intent(text, has_report=has_report)
    print(f"[LOGIC] Đã điều hướng sang luồng: {intent}")
    return intent

# ========================================================
# CÁC HÀM XÂY DỰNG PROMPT
# ========================================================
def _format_history(session: Dict[str, Any], limit: int = 10) -> str:
    history = session.get("history", [])[-limit:]
    if not history:
        return "(không có lịch sử)"

    lines = []
    for item in history:
        role = item.get("role", "unknown")
        content = str(item.get("content", "")).replace("\n", " ").strip()
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"- [{role}] {content}")
    return "\n".join(lines)

def _format_full_report(report_data: List[Dict[str, Any]]) -> str:
    if not report_data:
        return "(report rỗng)"

    lines = []
    for item in report_data:
        test_name = item.get("test_name", "?")
        value = item.get("value", "?")
        unit = item.get("unit", "")
        status = item.get("status")
        
        status_str = status if status is not None else "Không rõ"
        unit_str = unit if str(unit).strip() else "[Không ghi đơn vị]"

        ref = item.get("ref_range", {}) or {}
        ref_min = ref.get("ref_min")
        ref_max = ref.get("ref_max")

        if ref_min is None and ref_max is None:
            ref_str = "Không có sẵn"
        elif ref_min is None:
            ref_str = f"< {ref_max}"
        elif ref_max is None:
            ref_str = f"> {ref_min}"
        else:
            ref_str = f"{ref_min} - {ref_max}"

        lines.append(
            f"- {test_name}: Kết quả = {value} {unit_str} | Tham chiếu = {ref_str} | Trạng thái = {status_str}"
        )
    return "\n".join(lines)

def build_memory_block(session: Dict[str, Any]) -> str:
    user_name = session.get("user_name") or "Unknown"
    active_report = session.get("active_report")
    last_topic = session.get("last_topic")
    last_medical_context = session.get("last_medical_context")

    report_info = "None"
    if active_report:
        report_info = active_report.get("id", "unknown_report")

    recent_history = _format_history(session, limit=8)

    return f"""
User: {user_name}
Active report: {report_info}
Last topic: {last_topic}
Last medical context: {last_medical_context}

Recent history:
{recent_history}
""".strip()

def build_general_prompt(query: str, session: Dict[str, Any]) -> str:
    memory_block = build_memory_block(session)
    return f"""
Bạn là trợ lý thân thiện, trả lời bằng tiếng Việt.

{memory_block}

User:
{query}
""".strip()

def build_medical_prompt(
    query: str,
    evidence: List[Dict[str, Any]],
    session: Dict[str, Any],
) -> str:
    memory_block = build_memory_block(session)

    ev_text = "\n".join(
        [
            f"[{i+1}] \"{e.get('text', '')[:200]}\" "
            f"({e.get('source', 'unknown')}, p.{e.get('page', '?')})"
            for i, e in enumerate(evidence[:5])
        ]
    ) or "(không có evidence)"

    return f"""
You are a hematology specialist. Answer in Vietnamese.

{memory_block}

User question:
{query}

Evidence:
{ev_text}

Rules:
- Answer in Vietnamese
- MUST cite evidence like [1], [2]
- ONLY use evidence when making medical claims
- DO NOT hallucinate
- If evidence is weak, say it clearly

Structure:
1. Trả lời ngắn
2. Giải thích
3. Khuyến nghị
""".strip()

def build_report_followup_prompt(
    query: str,
    report_summary: str,
    abnormal_items: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    session: Dict[str, Any],
    full_report: Optional[Dict[str, Any]] = None,
) -> str:
    memory_block = build_memory_block(session)

    abnormal_text = "\n".join(
        [
            f"- {item['test_name']}: {item['value']} {item.get('unit', '[Không ghi đơn vị]')} "
            f"(Tham chiếu: {item['ref_range'].get('ref_min', 'Không có')} - {item['ref_range'].get('ref_max', 'Không có')}) => {item['status']}"
            for item in abnormal_items
        ]
    ) or "Không có chỉ số bất thường."

    ev_text = "\n".join(
        [
            f"[{i+1}] \"{e.get('text', '')[:220]}\" "
            f"({e.get('source', 'unknown')}, p.{e.get('page', '?')})"
            for i, e in enumerate(evidence[:5])
        ]
    ) or "(không có evidence)"

    report_id = None
    full_report_block = "(không có full report)"
    total_indices = 0
    
    if full_report:
        report_id = full_report.get("id")
        report_data = full_report.get("data", [])
        total_indices = len(report_data)
        full_report_block = _format_full_report(report_data)

    return f"""
You are a friendly and empathetic hematology specialist. Answer in Vietnamese.

{memory_block}

CURRENT REPORT ID:
{report_id}

TOTAL INDICES EXTRACTED:
Bệnh nhân có tổng cộng {total_indices} chỉ số trong phiếu xét nghiệm này.

CURRENT CBC REPORT SUMMARY:
{report_summary}

📊 TOÀN BỘ CHỈ SỐ BẤT THƯỜNG ĐƯỢC PHÁT HIỆN:
{abnormal_text}

FULL CBC REPORT:
{full_report_block}

USER FOLLOW-UP QUESTION:
{query}

EVIDENCE:
{ev_text}

Rules:
- Answer in a friendly, conversational Vietnamese tone.
- Combine user's report data + recent history + evidence
- The FULL CBC REPORT and the ABNORMAL ITEMS list are the absolute source of truth for this patient.
- CRITICAL: NEVER mention internal prompt variable names like "FULL CBC REPORT" or "TOÀN BỘ CHỈ SỐ BẤT THƯỜNG ĐƯỢC PHÁT HIỆN". Refer to them naturally as "phiếu xét nghiệm của bạn" hoặc "các chỉ số bất thường".
- CRITICAL: BẮT BUỘC liệt kê ĐẦY ĐỦ 100% các chỉ số bất thường liên quan đến câu hỏi, tuyệt đối không lược bỏ.
- CRITICAL: DO NOT use markdown headers (`#` or `###`). Use bold text (`**...**`) for emphasis.
- CRITICAL: Use friendly Vietnamese names for tests (e.g., Hồng cầu (RBC), Bạch cầu (WBC), Tiểu cầu (PLT)) instead of dry acronyms.
- CRITICAL ANTI-HALLUCINATION 1: If a reference range is "Không có sẵn" in the FULL CBC REPORT, you MUST firmly state that the report does not provide it. DO NOT invent or hallucinate a reference range using your external medical knowledge.
- CRITICAL ANTI-HALLUCINATION 2: If the user challenges you about a missing reference range (e.g., "Are you sure it doesn't have one?"), politely stand your ground. Confirm that based on the uploaded report data, it is truly missing.
- CRITICAL ANTI-HALLUCINATION 3: If a test result's unit is "[Không ghi đơn vị]", you MUST NOT assume, invent, or append any unit (like '%', 'g/L') to the value. Just use the exact number provided in the report.
- MUST cite evidence like [1], [2] when making medical claims.
- ONLY use citation indices that actually exist in EVIDENCE.
- DO NOT hallucinate external evidence.

Structure:
**1. Trả lời ngắn gọn:** Trả lời trực tiếp và thân thiện.
**2. Liên hệ với phiếu xét nghiệm:** Phân tích dữ liệu từ báo cáo.
**3. Giải thích y khoa (nếu cần):** Giải thích dễ hiểu, tránh hàn lâm.
**4. Lời khuyên:** Khuyến nghị thực tế cho bệnh nhân.
""".strip()

# ========================================================
# HÀM XỬ LÝ CHAT CHÍNH (ĐIỀU PHỐI)
# ========================================================
@observe(name="Chat Workflow")
def handle_chat(text: str, session_id: str) -> Dict:
    langfuse_context.update_current_trace(
        session_id=session_id,
        tags=["chat_interaction"],
        user_id="patient_anonymous"
    )

    session = update_session_memory(session_id, text=text)
    intent = detect_intent(text, session)
    
    if intent == "general_chat":
        prompt = build_general_prompt(text, session)
    elif intent == "medical_knowledge":
        print(f"[LOGIC] Tra cứu Qdrant cho Medical Knowledge: {text}")
        evidence = rag_service.search_qdrant(text, rag_service.embedding_model, rag_service.qdrant_client, top_k=5)
        prompt = build_medical_prompt(text, evidence, session)
    elif intent == "report_followup":
        print(f"[LOGIC] Tra cứu Qdrant cho Report Follow-up: {text}")
        evidence = rag_service.search_qdrant(text + " clinical interpretation cbc", rag_service.embedding_model, rag_service.qdrant_client, top_k=5)
        
        report_summary = session.get("active_report_summary") or "Chưa có tóm tắt."
        full_report_dict = session.get("active_report")
        report_data = full_report_dict.get("data", []) if full_report_dict else []
        abnormal_items = [i for i in report_data if i.get("status") in ["High", "Low"]]
        
        prompt = build_report_followup_prompt(
            query=text,
            report_summary=report_summary,
            abnormal_items=abnormal_items,
            evidence=evidence,
            session=session,
            full_report=full_report_dict
        )

    print("[LOGIC] Bắt đầu gọi LLM trả lời chat...")
    answer = rag_service.call_llm(prompt)
    session["history"].append({"role": "assistant", "content": answer})
    print("[LOGIC] Trả lời LLM thành công.")
    
    # Đã gỡ bỏ flush() gây treo ở đây
    return {"intent": intent, "answer": answer}