import json
from typing import Any, Dict, List
import rag_service

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

def detect_intent(text: str, session: Dict) -> str:
    report_cues = ["của tôi", "kết quả", "report", "phiếu", "chỉ số này", "nó", "cái này", "bất thường", "có sao không", "xét nghiệm", "chỉ số"]
    if session.get("active_report") and any(cue in text.lower() for cue in report_cues):
        return "report_followup"
    
    medical_keywords = ["wbc", "rbc", "hgb", "hct", "mcv", "mch", "mchc", "plt", "neut", "lym", "mono", "eos", "baso", "ig", "máu", "xét nghiệm", "cbc", "bạch cầu", "hồng cầu", "thiếu máu", "tiểu cầu", "bệnh"]
    if any(k in text.lower() for k in medical_keywords):
        return "medical_knowledge"
        
    return "general_chat"

# ========================================================
# CÁC HÀM XÂY DỰNG PROMPT (GIỮ NGUYÊN 100% NHƯ BẠN YÊU CẦU)
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
        status = item.get("status", "Unknown")

        ref = item.get("ref_range", {}) or {}
        ref_min = ref.get("ref_min", "?")
        ref_max = ref.get("ref_max", "?")

        lines.append(
            f"- {test_name}: {value} {unit} "
            f"(ref {ref_min} - {ref_max}) => {status}"
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
    full_report: Dict[str, Any] | None = None,
) -> str:
    memory_block = build_memory_block(session)

    abnormal_text = "\n".join(
        [
            f"- {item['test_name']}: {item['value']} {item['unit']} "
            f"(ref {item['ref_range']['ref_min']} - {item['ref_range']['ref_max']}) => {item['status']}"
            for item in abnormal_items
        ]
    ) or "Không có chỉ số bất thường rõ ràng."

    ev_text = "\n".join(
        [
            f"[{i+1}] \"{e.get('text', '')[:220]}\" "
            f"({e.get('source', 'unknown')}, p.{e.get('page', '?')})"
            for i, e in enumerate(evidence[:5])
        ]
    ) or "(không có evidence)"

    report_id = None
    full_report_block = "(không có full report)"
    if full_report:
        report_id = full_report.get("id")
        full_report_block = _format_full_report(full_report.get("data", []))

    return f"""
You are a hematology specialist. Answer in Vietnamese.

{memory_block}

CURRENT REPORT ID:
{report_id}

CURRENT CBC REPORT SUMMARY:
{report_summary}

ABNORMAL ITEMS:
{abnormal_text}

FULL CBC REPORT:
{full_report_block}

USER FOLLOW-UP QUESTION:
{query}

EVIDENCE:
{ev_text}

Rules:
- Answer in Vietnamese
- Combine user's report data + recent history + evidence
- The FULL CBC REPORT is the source of truth for this patient
- If the user asks about a normal index, you MUST check it from FULL CBC REPORT
- If the user asks something referring to prior conversation, use Recent history
- MUST cite evidence like [1], [2] when making medical claims
- ONLY use citation indices that actually exist in EVIDENCE
- Each citation [i] must correspond exactly to evidence item [i]
- DO NOT hallucinate
- If the question is answerable from report data only, answer directly and say no external evidence is needed
- If the question cannot be answered from current report/history/evidence, say so clearly

Structure:
1. Trả lời ngắn
2. Liên hệ trực tiếp với report hiện tại
3. Nếu cần, liên hệ với lịch sử hội thoại
4. Giải thích y khoa
5. Khuyến nghị
""".strip()

# ========================================================
# HÀM XỬ LÝ CHAT CHÍNH (ĐIỀU PHỐI)
# ========================================================
def handle_chat(text: str, session_id: str) -> Dict:
    # 1. Cập nhật câu hỏi của User vào bộ nhớ
    session = update_session_memory(session_id, text=text)
    
    # 2. Phân loại câu hỏi
    intent = detect_intent(text, session)
    
    # 3. Kịch bản 1: Trò chuyện chung
    if intent == "general_chat":
        prompt = build_general_prompt(text, session)
        
    # 4. Kịch bản 2: Hỏi y khoa chung
    elif intent == "medical_knowledge":
        evidence = rag_service.search_qdrant(text, rag_service.embedding_model, rag_service.qdrant_client, top_k=5)
        prompt = build_medical_prompt(text, evidence, session)
        
    # 5. Kịch bản 3: Hỏi về tờ phiếu xét nghiệm
    elif intent == "report_followup":
        # Tìm kiếm tài liệu y khoa
        evidence = rag_service.search_qdrant(text + " clinical interpretation cbc", rag_service.embedding_model, rag_service.qdrant_client, top_k=5)
        
        # Lấy tóm tắt và dữ liệu từ bộ nhớ
        report_summary = session.get("active_report_summary") or "Chưa có tóm tắt."
        full_report_dict = session.get("active_report")
        report_data = full_report_dict.get("data", []) if full_report_dict else []
        abnormal_items = [i for i in report_data if i.get("status") in ["High", "Low"]]
        
        # Gắn vào Prompt đúng y như Format của bạn
        prompt = build_report_followup_prompt(
            query=text,
            report_summary=report_summary,
            abnormal_items=abnormal_items,
            evidence=evidence,
            session=session,
            full_report=full_report_dict
        )

    # 6. Gọi LLM và lưu vào lịch sử
    answer = rag_service.call_llm(prompt)
    session["history"].append({"role": "assistant", "content": answer})
    
    return {"intent": intent, "answer": answer}