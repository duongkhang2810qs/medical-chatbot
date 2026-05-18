import json
import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langfuse.decorators import observe, langfuse_context

import rag_service
import lab_core

# =========================================================
# QUẢN LÝ DỮ LIỆU PHIẾU TẠM THỜI
# (LangGraph quản lý lịch sử chat, SESSIONS chỉ dùng lưu phiếu)
# =========================================================
SESSIONS: Dict[str, Dict[str, Any]] = {}

def update_session_memory(session_id: str, active_report_data=None, report_summary=None, text=None):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "active_report": None,
            "report_summary": "Chưa có tóm tắt.",
        }
    
    if active_report_data is not None:
        SESSIONS[session_id]["active_report"] = {"data": active_report_data}
    if report_summary is not None:
        SESSIONS[session_id]["report_summary"] = report_summary
        
    return SESSIONS[session_id]

# =========================================================
# 1. KHAI BÁO GRAPH STATE (TRẠNG THÁI ĐỒ THỊ)
# =========================================================
class ChatState(TypedDict):
    session_id: str
    messages: List[Dict[str, str]]   # Lịch sử hội thoại
    current_input: str               # Câu hỏi hiện tại của user
    intent: str                      # Ý định (do Router gán)
    active_report: Optional[Dict]    # Dữ liệu phiếu OCR
    report_summary: str              # Tóm tắt phiếu OCR
    evidence: List[Dict]             # Chứng cứ tìm được (RAG)
    final_answer: str                # Câu trả lời cuối cùng

# =========================================================
# 2. XÂY DỰNG CÁC NODE (CÁC TRẠM XỬ LÝ)
# =========================================================

@observe(as_type="generation", name="LLM Router Node")
def route_node(state: ChatState):
    """Sử dụng LLM để đọc hiểu câu hỏi và gán nhãn rẽ nhánh"""
    input_text = state["current_input"]
    has_report = "CÓ" if state.get("active_report") else "KHÔNG"
    
    prompt = f"""
    Bạn là hệ thống định tuyến (Router) cho một Chatbot Y khoa.
    Trạng thái: Người dùng đã tải phiếu xét nghiệm lên chưa? {has_report}
    Câu hỏi của người dùng: "{input_text}"

    Hãy phân loại câu hỏi vào ĐÚNG 1 trong 3 nhóm:
    1. "report_followup": Hỏi về kết quả xét nghiệm của họ, chỉ số của họ, bệnh của họ (VD: "wbc của tôi", "tại sao giảm", "có sao không").
    2. "medical_knowledge": Hỏi kiến thức y khoa chung chung (VD: "hồng cầu là gì", "nguyên nhân tiểu đường").
    3. "general_chat": Giao tiếp thông thường (VD: "xin chào", "cảm ơn", "ok").

    BẮT BUỘC CHỈ trả về một chuỗi JSON hợp lệ, không kèm văn bản nào khác:
    {{"intent": "tên_nhóm"}}
    """
    
    raw_response = rag_service.call_llm(prompt)
    intent = "general_chat" # Mặc định an toàn
    
    # Ép kiểu an toàn (đề phòng LLM trả về markdown dạng ```json ... ```)
    try:
        match = re.search(r'\{.*\}', raw_response.replace('\n', ''), re.IGNORECASE)
        if match:
            parsed = json.loads(match.group())
            if "intent" in parsed and parsed["intent"] in ["report_followup", "medical_knowledge", "general_chat"]:
                intent = parsed["intent"]
    except Exception as e:
        print(f"[ROUTER ERROR] Không thể parse JSON: {e}. Dùng default: {intent}")

    print(f"🚥 [Router] Điều hướng tới nhánh: {intent}")
    return {"intent": intent}


@observe(as_type="generation", name="General Chat Node")
def general_chat_node(state: ChatState):
    """Xử lý giao tiếp bình thường (Chào hỏi, cảm ơn)"""
    hist = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"][-4:]])
    prompt = f"""
    Lịch sử chat:
    {hist}
    
    Người dùng vừa nói: "{state['current_input']}"
    Hãy trả lời như một trợ lý y khoa lịch sự, thân thiện. Trả lời ngắn gọn (1-2 câu).
    """
    ans = rag_service.call_llm(prompt)
    return {"final_answer": ans}


@observe(as_type="generation", name="Medical Knowledge Node")
def knowledge_node(state: ChatState):
    """Tra cứu Qdrant để trả lời kiến thức Y khoa"""
    query = state["current_input"]
    print(f"📚 [Knowledge] Tra cứu Qdrant: {query}")
    evidence = lab_core.qdrant_search(query, top_k=5)
    
    ev_text = "\n".join([f"- {e['text']} (Nguồn: {e['source']})" for e in evidence])
    
    prompt = f"""
    Bạn là bác sĩ tư vấn. Hãy trả lời câu hỏi y khoa sau: "{query}"
    Dùng kiến thức y khoa chuẩn và tham khảo tài liệu sau (nếu có ích):
    {ev_text}
    
    Không cần bịa tài liệu. Trả lời rõ ràng, dễ hiểu cho bệnh nhân.
    """
    ans = rag_service.call_llm(prompt)
    return {"final_answer": ans, "evidence": evidence}


@observe(as_type="generation", name="Clinical Report Node")
def report_node(state: ChatState):
    """Phân tích chuyên sâu dựa trên phiếu xét nghiệm của user"""
    query = state["current_input"]
    
    # [FIX] Đảm bảo report luôn là dict, ngay cả khi state["active_report"] là None
    report = state.get("active_report") or {}
    report_summary = state.get("report_summary") or ""
    
    # Trích xuất các chỉ số bất thường
    abnormal_items = [i for i in report.get("data", []) if str(i.get("status")) in ["High", "Low"]]
    abn_text = "\n".join([f"- {i['test_name']}: {i['value']} {i['unit']} (Bất thường: {i['status']})" for i in abnormal_items])
    
    print(f"🔬 [Report Followup] Tra cứu Qdrant: {query}")
    evidence = lab_core.qdrant_search(query + " clinical interpretation cbc biochem", top_k=5)
    ev_text = "\n".join([f"- {e['text']}" for e in evidence])
    
    hist = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"][-3:]])

    prompt = f"""
    Bạn là bác sĩ tư vấn kết quả xét nghiệm. Người dùng đang hỏi về phiếu xét nghiệm của họ.
    
    LỊCH SỬ CHAT GẦN NHẤT:
    {hist}
    
    TÓM TẮT PHIẾU CỦA NGƯỜI DÙNG:
    {report_summary}
    
    CÁC CHỈ SỐ BẤT THƯỜNG TRONG PHIẾU:
    {abn_text if abn_text else "Không có chỉ số nào vượt ngưỡng tham chiếu."}
    
    TÀI LIỆU Y KHOA THAM KHẢO (EVIDENCE):
    {ev_text}
    
    CÂU HỎI HIỆN TẠI: "{query}"
    
    YÊU CẦU:
    - Trả lời trực tiếp vào câu hỏi.
    - Dựa chặt chẽ vào các chỉ số bất thường của họ để giải thích.
    - Nhấn mạnh việc cần tham khảo ý kiến bác sĩ trực tiếp.
    - Không kê đơn thuốc.
    """
    ans = rag_service.call_llm(prompt)
    return {"final_answer": ans, "evidence": evidence}


# =========================================================
# 3. KẾT NỐI LANGGRAPH
# =========================================================

def decide_next_node(state: ChatState):
    """Hàm quyết định rẽ nhánh dựa vào intent"""
    intent = state.get("intent", "general_chat")
    if intent == "report_followup": return "report"
    if intent == "medical_knowledge": return "knowledge"
    return "general"

# Khởi tạo đồ thị
builder = StateGraph(ChatState)

# Thêm Node
builder.add_node("router", route_node)
builder.add_node("general", general_chat_node)
builder.add_node("knowledge", knowledge_node)
builder.add_node("report", report_node)

# Bắt đầu chạy vào Router
builder.add_edge(START, "router")

# Từ Router rẽ nhánh có điều kiện
builder.add_conditional_edges("router", decide_next_node)

# Các nhánh xử lý xong thì Kết thúc
builder.add_edge("general", END)
builder.add_edge("knowledge", END)
builder.add_edge("report", END)

# Biên dịch Graph với Memory (Trí nhớ)
memory = MemorySaver()
app_graph = builder.compile(checkpointer=memory)

# =========================================================
# 4. HÀM CHÍNH GỌI TỪ API
# =========================================================

@observe(name="Chat Workflow (LangGraph)")
def handle_chat(text: str, session_id: str) -> Dict:
    langfuse_context.update_current_trace(
        session_id=session_id,
        tags=["langgraph_chat"],
        user_id="patient_anonymous"
    )

    # Lấy dữ liệu phiếu từ Session tạm thời (nếu user đã upload)
    session_data = SESSIONS.get(session_id, {})
    
    # Cấu hình thread_id để LangGraph tự động lôi lịch sử chat cũ ra
    config = {"configurable": {"thread_id": session_id}}
    
    # Khởi tạo Input State cho lượt chạy này
    inputs = {
        "session_id": session_id,
        "current_input": text,
        "active_report": session_data.get("active_report"),
        "report_summary": session_data.get("report_summary", ""),
        "messages": [{"role": "user", "content": text}]  # Graph tự cộng dồn vào messages cũ
    }
    
    print("\n" + "="*50)
    print(f"🚀 [LangGraph] Bắt đầu luồng xử lý mới (Session: {session_id})")
    
    # Chạy đồ thị
    for output in app_graph.stream(inputs, config=config):
        # In ra màn hình xem đang chạy tới Node nào
        for key, value in output.items():
            print(f"   => Đã chạy qua Node: [{key.upper()}]")
            
    # Lấy State cuối cùng sau khi Graph chạy xong
    final_state = app_graph.get_state(config).values
    
    print("✅ [LangGraph] Hoàn tất luồng.")
    print("="*50 + "\n")
    
    return {
        "answer": final_state.get("final_answer", "Lỗi xử lý luồng AI."),
        "intent": final_state.get("intent", "unknown")
    }