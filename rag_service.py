import os
import json
import uuid
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load biến môi trường
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# =========================================================
# KẾT NỐI CÁC EXTERNAL SERVICES (LLM, NEO4J)
# =========================================================
from google import genai
from google.genai import types
from neo4j import GraphDatabase
from langfuse.decorators import observe, langfuse_context

# Import Core Engine
import config
import lab_core

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

print("🔧 Khởi tạo AI Services (Neo4j)...")
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    print("✅ Đã kết nối Neo4j Knowledge Graph thành công!")
except Exception as e:
    print(f"⚠️ Lỗi kết nối Neo4j: {e}")
    neo4j_driver = None

# Ghi chú: Đã gỡ bỏ SentenceTransformer vì LangGraph LLM Router đã đảm nhiệm việc định tuyến.

# =========================================================
# GỌI LLM (DYNAMIC PROVIDER ROUTING)
# =========================================================
@observe(as_type="generation", name="Gemini Generation")
def call_gemini(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    res = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=2000)
    )
    return res.text.strip()

@observe(as_type="generation", name="OpenRouter Generation")
def call_deepseek(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 2000}
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=40)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()

@observe(as_type="generation", name="Local LLM Generation")
def call_local_llm(prompt):
    payload = {
        "model": LOCAL_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 2000
        }
    }
    res = requests.post(LOCAL_LLM_URL, json=payload, timeout=60)
    res.raise_for_status()
    return res.json().get("response", "").strip()

# Từ điển ánh xạ tên provider với hàm thực thi tương ứng
PROVIDER_FUNCTIONS = {
    "gemini": call_gemini,
    "deepseek": call_deepseek,
    "local": call_local_llm
}

# Đọc cấu hình từ .env (Nếu không có thì dùng mặc định là deepseek -> gemini -> local)
LLM_PROVIDER_ORDER = os.getenv("LLM_PROVIDER_ORDER", "deepseek,gemini,local").split(",")

@observe(as_type="span", name="LLM Execution")
def call_llm(prompt):
    # Chuẩn hóa danh sách
    order = [p.strip().lower() for p in LLM_PROVIDER_ORDER if p.strip()]
    
    for provider in order:
        if provider not in PROVIDER_FUNCTIONS:
            print(f"⚠️ Cảnh báo: LLM Provider '{provider}' không hợp lệ. Bỏ qua.")
            continue
            
        print(f"[LLM] Đang gọi {provider.upper()}...")
        try:
            target_function = PROVIDER_FUNCTIONS[provider]
            return target_function(prompt)
        except Exception as e:
            print(f"⚠️ {provider.upper()} lỗi ({e}) -> Fallback sang model tiếp theo...")
            
    print("❌ Tất cả các model LLM đều thất bại.")
    return "Hệ thống AI đang bận hoặc mất kết nối. Vui lòng thử lại sau."


# =========================================================
# NEO4J QUERY CORE
# =========================================================
@observe(as_type="retrieval", name="Neo4j Cypher Retrieval")
def fetch_evidence_from_neo4j(abnormal_tests: list, conditions: list) -> list:
    if not neo4j_driver or (not abnormal_tests and not conditions):
        return []
    
    # Query đồ thị: Tìm sách y khoa (Evidence) hỗ trợ các Bệnh lý (Condition) hoặc có nhắc tới Chỉ số (Test)
    query = """
    MATCH (e:Evidence)
    OPTIONAL MATCH (e)-[:MENTIONS_TEST]->(t:Test)
    OPTIONAL MATCH (e)-[:SUPPORTS]->(c:Condition)
    WHERE t.test_code IN $tests OR c.name IN $conditions
    RETURN DISTINCT e.id AS evidence_id, e.text AS text, e.source AS source, 
                    e.page AS page, e.score AS kb_score, e.panel AS panel
    ORDER BY kb_score DESC
    LIMIT 6
    """
    evidence_list = []
    with neo4j_driver.session() as session:
        result = session.run(query, tests=abnormal_tests, conditions=conditions)
        for r in result:
            evidence_list.append({
                "evidence_id": r["evidence_id"],
                "text": r["text"],
                "source": r["source"],
                "page": r["page"],
                "kb_score": r["kb_score"],
                "panel": r["panel"],
                "tests": abnormal_tests,
                "conditions": conditions,
                "score": r["kb_score"] + 0.5  # Boost điểm vì lấy từ Graph có độ ưu tiên cao
            })
            
    langfuse_context.update_current_observation(output={"nodes_found": len(evidence_list)})
    return evidence_list


# =========================================================
# LUỒNG PHÂN TÍCH REPORT ĐẦU TIÊN
# =========================================================
def format_ui_data_to_case(user_indicators: list, session_id: str) -> dict:
    case_data = {
        "case_id": session_id or f"realtime_{uuid.uuid4().hex[:6]}",
        "data": []
    }
    for item in user_indicators:
        ref_min = item.get('ref_range', {}).get('ref_min', '')
        ref_max = item.get('ref_range', {}).get('ref_max', '')
        ref_str = f"{ref_min} - {ref_max}".strip(" -")
        
        case_data["data"].append({
            "test_name": item.get("test_name"),
            "value": item.get("value"),
            "unit": item.get("unit", ""),
            "status": item.get("status"),
            "reference_range": ref_str
        })
    return case_data


@observe(name="GraphRAG Report Analysis")
def analyze_indicators_with_llm(user_indicators: list, session_id: str = None) -> str:
    if session_id:
        langfuse_context.update_current_trace(session_id=session_id, tags=["graphrag_analysis"])
        
    print("\n[AI SERVICE] Bắt đầu phân tích GraphRAG...")

    # 1. Chuyển đổi Dữ liệu & Xây dựng Context Suy luận
    case_dict = format_ui_data_to_case(user_indicators, session_id)
    ctx = lab_core.build_reasoning_context(case_dict, 0)
    
    # Nạp Static Patterns từ files
    cbc_demo = lab_core.load_jsonl(config.CBC_DEMO_PATTERN_PATH) if config.CBC_DEMO_PATTERN_PATH.exists() else []
    biochem_patt = lab_core.load_json(config.BIOCHEM_PATTERN_PATH) if config.BIOCHEM_PATTERN_PATH.exists() else {}
    
    ctx = lab_core.augment_reasoning_context_with_static_patterns(ctx, cbc_demo, biochem_patt)

    abnormal_tests = ctx.get("abnormal_tests", [])
    conditions = ctx.get("conditions", [])

    if not ctx.get("abnormal_items"):
        return "Kết quả xét nghiệm của bạn nằm trong giới hạn tham chiếu. Không phát hiện chỉ số bất thường nào."

    # 2. Truy xuất Kiến thức: Kết hợp Graph (Neo4j) + Vector (Qdrant trong lab_core)
    print(f"[GraphRAG] Đang truy vấn Neo4j cho bệnh lý: {conditions}...")
    graph_evidence = fetch_evidence_from_neo4j(abnormal_tests, conditions)
    
    print("[GraphRAG] Đang truy vấn Vector Qdrant (Fallback bổ sung)...")
    vector_evidence = lab_core.retrieve_evidence(ctx)
    
    # Gộp và lọc trùng lặp chứng cứ
    combined_evidence = lab_core.dedup_evidence(graph_evidence + vector_evidence)
    final_evidence = lab_core.rerank_evidence(combined_evidence, ctx)[:config.MAX_FINAL_EVIDENCE]

    # 3. Tạo đường dẫn suy luận Graph (Reasoning Paths)
    graph_reasoning_paths = lab_core.enrich_reasoning_paths(ctx, final_evidence)

    # 4. Sinh Prompt Siêu cấp từ lab_core
    prompt = lab_core.build_final_prompt(
        reasoning_context=ctx,
        evidence=final_evidence,
        reasoning_paths=graph_reasoning_paths
    )

    # 5. Gọi LLM và Dọn dẹp Output
    raw_answer = call_llm(prompt)
    
    final_answer = lab_core.build_user_visible_answer(raw_answer, ctx, final_evidence)

    return final_answer