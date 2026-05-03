import os
from pathlib import Path
from dotenv import load_dotenv

# ==============================
# PATH & CONFIG
# ==============================
BASE_DIR = Path(__file__).resolve().parent
PATTERN_PATH = BASE_DIR / "demo_cases_all_clean.json"
KB_PATH = BASE_DIR / "cbc_kb_v8.json"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

# [FIX] Đọc tên LLM từ file .env để tránh hardcode, có giá trị mặc định nếu quên khai báo
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

import json
import uuid
import requests
import numpy as np
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langfuse.decorators import observe, langfuse_context

# ==============================
# 1. KHỞI TẠO QDRANT & MODEL
# ==============================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==============================
# 2. SEMANTIC ROUTING (ĐIỀU HƯỚNG NGỮ NGHĨA)
# ==============================
ROUTE_EXAMPLES = {
    "report_followup": [
        "của tôi", "kết quả", "report", "phiếu", "chỉ số này", "nó", "cái này", 
        "bất thường", "có sao không", "xét nghiệm của tôi", "đọc giúp tôi",
        "chỉ số wbc của tôi cao quá", "tại sao giảm", "kiểm tra lại", "bị bệnh gì"
    ],
    "medical_knowledge": [
        "wbc là gì", "rbc là gì", "hgb", "hct", "mcv", "mch", "mchc", "plt", "neut", 
        "lym", "mono", "eos", "baso", "ig", "máu", "cbc", "bạch cầu là gì", 
        "hồng cầu", "bệnh thiếu máu", "tiểu cầu", "triệu chứng", "nguyên nhân"
    ],
    "general_chat": [
        "xin chào", "hello", "hi", "cảm ơn", "thank you", "bạn là ai", 
        "tạm biệt", "chào bác sĩ", "ok", "dạ", "chào"
    ]
}

ROUTE_EMBEDDINGS = {}

def init_semantic_router():
    global ROUTE_EMBEDDINGS
    if embedding_model:
        for intent, examples in ROUTE_EXAMPLES.items():
            ROUTE_EMBEDDINGS[intent] = embedding_model.encode(examples)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@observe(as_type="span", name="Semantic Routing")
def semantic_route_intent(query: str, has_report: bool = False) -> str:
    if not embedding_model or not ROUTE_EMBEDDINGS:
        return "report_followup" if has_report else "general_chat"

    query_emb = embedding_model.encode(query)
    best_intent = "general_chat"
    best_score = -1

    for intent, embs in ROUTE_EMBEDDINGS.items():
        scores = [cosine_similarity(query_emb, e) for e in embs]
        max_score = max(scores)
        if max_score > best_score:
            best_score = max_score
            best_intent = intent

    langfuse_context.update_current_observation(output={"intent": best_intent, "confidence": float(best_score)})

    if has_report and best_intent == "medical_knowledge" and best_score < 0.6:
        return "report_followup"
    if best_score < 0.35:
        return "general_chat"

    return best_intent


print("🔧 Loading Qdrant and Models for AI Service...")
try:
    PATTERN_DATA = load_json(PATTERN_PATH)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(host="localhost", port=6333)
    
    collections_response = qdrant_client.get_collections()
    collection_names = [c.name for c in collections_response.collections]
    
    if "cbc" not in collection_names:
        print("🚀 Collection 'cbc' chưa tồn tại. Đang tự động nạp dữ liệu từ cbc_kb_v8.json...")
        qdrant_client.create_collection(
            collection_name="cbc",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        if KB_PATH.exists():
            kb_data = load_json(KB_PATH)
            points = []
            print(f"⏳ Đang mã hóa vector cho {len(kb_data)} đoạn text. Vui lòng đợi...")
            for item in kb_data:
                text = item.get("text", "")
                if not text.strip(): continue
                vector = embedding_model.encode(text).tolist()
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={"text": text, "source": item.get("source", "Unknown"), "page": item.get("page", 0)}
                    )
                )
            if points:
                qdrant_client.upsert(collection_name="cbc", points=points)
                print(f"✅ Đã nạp thành công {len(points)} đoạn kiến thức y khoa vào Qdrant!")
    
    init_semantic_router()
    print("✅ Đã kết nối Qdrant thành công (AI Service)!")
except Exception as e:
    print(f"⚠️ Lỗi khởi tạo: {e}")
    PATTERN_DATA, embedding_model, qdrant_client = [], None, None


def normalize_tag(tag):
    return tag.replace("_PERCENT", "").replace("_ABS", "")

def extract_abnormal(user_indicators):
    tags = []
    abnormal_items = []
    for item in user_indicators:
        st = item.get("status", "Normal")
        if st != "Normal" and st is not None:
            tag = normalize_tag(f"{item.get('test_name')}_{st}")
            tags.append(tag)
            abnormal_items.append({
                "test": item.get("test_name"),
                "value": item.get("value"),
                "status": st
            })
    return tags, abnormal_items

@observe(as_type="span", name="Match Patterns")
def match_patterns(tags, pattern_data):
    if not pattern_data: return []
    matched = []
    for case in pattern_data:
        case_tags = {normalize_tag(t) for t in case.get("input_tags", [])}
        overlap = case_tags & set(tags)
        if len(overlap) >= 2:
            matched.append({
                "pattern_name": case.get("pattern_name"),
                "interpretation": case.get("combined_interpretation"),
                "confidence": round(len(overlap) / len(case_tags), 2)
            })
    matched.sort(key=lambda x: x["confidence"], reverse=True)
    return matched[:3]

def search_qdrant(query, model, client, top_k=5):
    if model is None or client is None: return []
    emb = model.encode(query).tolist()
    results = client.query_points(collection_name="cbc", query=emb, limit=top_k)
    output = []
    for r in results.points:
        output.append({
            "score": float(r.score),
            "text": r.payload.get("text"),
            "source": r.payload.get("source"),
            "page": r.payload.get("page")
        })
    return output

def expand_query(query: str) -> list:
    return [query, f"{query} clinical interpretation", f"{query} medical meaning"]

@observe(as_type="retrieval", name="Retrieve Qdrant Evidence")
def retrieve_multi(query: str, model, client, top_k=5):
    langfuse_context.update_current_observation(input=query)
    all_results = []
    for q in expand_query(query):
        all_results.extend(search_qdrant(q, model, client, top_k=top_k))

    seen = set()
    final = []
    for e in sorted(all_results, key=lambda x: x.get("score", 0.0), reverse=True):
        text_key = (e.get("text") or "")[:120]
        if text_key and text_key not in seen:
            seen.add(text_key)
            final.append(e)

    result = final[:top_k]
    langfuse_context.update_current_observation(output=[item.get("text") for item in result])
    return result

def filter_evidence(evidence, abnormal_items):
    keywords = [x["test"].lower() for x in abnormal_items]
    filtered = []
    for e in evidence:
        text = e["text"].lower()
        if any(k in text for k in keywords) or any(syn in text for syn in ["blood", "cell", "hemoglobin", "leukocyte"]):
            filtered.append(e)
    return filtered if len(filtered) >= 2 else evidence[:3]

def calculate_confidence(patterns, evidence):
    if not evidence: return 0.2
    score = 0.3
    avg = sum([e["score"] for e in evidence[:3]]) / len(evidence[:3])
    score += avg * 0.5
    if patterns: score += 0.2
    return round(min(score, 0.95), 2)

@observe(as_type="span", name="Build System Prompt")
def build_prompt(abnormal_items, patterns, evidence):
    abnormal_text = "\n".join([f"- 🔴 {x['test']}: {x['status']} ({x['value']})" for x in abnormal_items])
    pattern_text = "\n".join([f"- {p['interpretation']}" for p in patterns])
    if not pattern_text: pattern_text = "Không phát hiện pattern rõ ràng, dựa vào từng chỉ số riêng lẻ."
    evidence_text = "\n".join([f"[{i+1}] \"{e['text'][:200]}\" ({e.get('source')}, p.{e.get('page')})" for i, e in enumerate(evidence[:3])])

    return f"""
You are a friendly and professional hematology specialist with strong clinical reasoning.

Your task:
Analyze CBC test abnormalities and provide evidence-based interpretation in a patient-friendly way.

=====================
📊 TOÀN BỘ CHỈ SỐ BẤT THƯỜNG ĐƯỢC PHÁT HIỆN:
{abnormal_text}

🧠 PATTERN GỢI Ý:
{pattern_text}

📚 TÀI LIỆU THAM KHẢO (EVIDENCE):
{evidence_text}
=====================

IMPORTANT RULES (MUST FOLLOW STRICTLY):

1. 🔴 CRITICAL: Trong phần "1. Tóm tắt tình trạng", bạn BẮT BUỘC PHẢI liệt kê ĐẦY ĐỦ 100% các chỉ số đã xuất hiện trong danh sách "TOÀN BỘ CHỈ SỐ BẤT THƯỜNG" ở trên. Tuyệt đối không được lược bỏ bất kỳ chỉ số nào (kể cả các chỉ số tỷ lệ % và tuyệt đối #).
2. 🔴 ONLY use information from the "EVIDENCE"
3. 🔴 DO NOT hallucinate or invent information
4. 🔴 ONLY cite from the provided EVIDENCE items
5. 🔴 DO NOT fabricate sources or page numbers
6. 🔴 Each citation [i] MUST correspond exactly to EVIDENCE [i]
7. 🔴 If you use evidence [1], cite as [1] only
8. 🔴 NEVER cite [4], [5], ... if only [1], [2], [3] are provided
9. 🔴 If evidence is insufficient, explicitly say the evidence is limited
10. 🔴 Ignore irrelevant evidence
11. 🔴 DO NOT use markdown headers like `###` or `#`. Use bold text `**...**` for formatting.
12. 🔴 Translate dry test acronyms into friendly Vietnamese names (e.g., Hồng cầu (RBC), Bạch cầu (WBC), Tiểu cầu (PLT), Huyết sắc tố (HGB)).

CITATION FORMAT RULES:

- Use citation format like [1], [2], [3]
- The citation index must match the exact EVIDENCE item number
- Do not invent book names or page numbers outside the provided EVIDENCE
- When mentioning a medical claim, attach the matching citation right in the sentence

=====================

CLINICAL REQUIREMENTS:

- Use proper medical reasoning
- Connect abnormal findings logically
- Explain in a way that is easy for a patient to understand.
- If possible:
  → Suggest likely diagnoses (NOT definitive)
  → Assess severity (mild / moderate / severe)

=====================

OUTPUT FORMAT (WRITE IN VIETNAMESE, NO '###' OR '#' HEADERS, USE BOLD INSTEAD):

**1. Tóm tắt tình trạng:**
- BẮT BUỘC liệt kê TOÀN BỘ các chỉ số bất thường (dùng 🔴) kèm tên tiếng Việt dễ hiểu. Mỗi chỉ số một dòng. Không được gom nhóm hay bỏ sót ở mục này.

**2. Giải thích chi tiết:** (có citation)
- Gom nhóm các bất thường liên quan và giải thích một cách thân thiện.
- MUST include citation in sentence
- ONLY use citation numbers that exist in EVIDENCE

**3. Nguyên nhân khả dĩ:**
- Liệt kê nguyên nhân
- If evidence is weak, say it clearly

**4. Chẩn đoán gợi ý:**
- In đậm (**...**)
- Lưu ý người bệnh đây chỉ là gợi ý, không thay thế chẩn đoán của bác sĩ.

**5. Đánh giá mức độ:**
- Nhẹ / Trung bình / Nặng
- Giải thích ngắn

**6. Lời khuyên & Lưu ý:**
- Đề xuất xét nghiệm thêm
- Cảnh báo quan trọng

**7. Kết luận chung:**
- Viết 1 câu kết luận ngắn gọn, súc tích và thân thiện (1 dòng duy nhất)

=====================

NOTES:

- Final answer MUST be in Vietnamese
- Prioritize accuracy over length
- Use professional but friendly, empathetic clinical tone
"""

# ==============================
# HÀM GỌI LLM (ĐÃ CHUYỂN ƯU TIÊN SANG GEMINI)
# ==============================

@observe(as_type="generation", name="Gemini Fallback Generation")
def call_gemini(prompt):
    langfuse_context.update_current_observation(
        model=GEMINI_MODEL,
        input=prompt,
        model_parameters={"temperature": 0.2, "max_output_tokens": 2000}
    )
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    res = client.models.generate_content(
        model=GEMINI_MODEL, 
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2000
        )
    )
    content = res.text.strip()
    langfuse_context.update_current_observation(output=content)
    return content

@observe(as_type="generation", name="OpenRouter Generation")
def call_deepseek(prompt):
    langfuse_context.update_current_observation(
        model=OPENROUTER_MODEL,
        input=[{"role": "user", "content": prompt}],
        model_parameters={"temperature": 0.2, "max_tokens": 2000}
    )
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": OPENROUTER_MODEL, 
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2000
    }
    
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=40)
    res.raise_for_status()
    
    response_json = res.json()
    content = response_json["choices"][0]["message"]["content"]
    
    usage = response_json.get("usage", {})
    if usage:
        langfuse_context.update_current_observation(
            usage={
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0)
            }
        )
        
    langfuse_context.update_current_observation(output=content)
    return content

@observe(as_type="span", name="LLM Routing")
def call_llm(prompt):
    try:
        print(f"[LLM] Đang gọi OpenRouter ({OPENROUTER_MODEL})...")
        return call_deepseek(prompt)
    except Exception as e:
        print(f"⚠️ OpenRouter failed ({e}) -> Đang gọi Gemini fallback ({GEMINI_MODEL})...")
        try:
            return call_gemini(prompt)
        except Exception as ex:
            print(f"⚠️ Cả OpenRouter và Gemini đều thất bại: {ex}")
            return "Hệ thống AI đang quá tải. Vui lòng thử lại sau ít phút."

def get_book_intro():
    return "\n📚 Nguồn tài liệu:\n- Harrison's Principles of Internal Medicine\n- Clinical Hematology\n"

@observe(name="Analyze Report Workflow")
def analyze_indicators_with_llm(user_indicators: list, session_id: str = None) -> str:
    if session_id:
        langfuse_context.update_current_trace(
            session_id=session_id,
            tags=["report_analysis"],
            user_id="patient_anonymous"
        )
        
    print("\n[AI SERVICE] Bắt đầu phân tích RAG (Monitored by Langfuse)...")
    
    if embedding_model is None or qdrant_client is None:
        return "⚠️ Lỗi Hệ Thống: Không thể kết nối DB Y khoa."

    tags, abnormal_items = extract_abnormal(user_indicators)
    if not abnormal_items:
        return "Kết quả xét nghiệm của bạn nằm trong giới hạn tham chiếu. Không phát hiện chỉ số bất thường nào."

    print("[RAG] Đang matching pattern...")
    patterns = match_patterns(tags, PATTERN_DATA)
    
    print("[RAG] Đang truy xuất Qdrant...")
    query = " ".join(tags)
    evidence = retrieve_multi(query, embedding_model, qdrant_client, top_k=5)
    evidence = filter_evidence(evidence, abnormal_items)

    print("[RAG] Đang xây dựng Prompt...")
    prompt = build_prompt(abnormal_items, patterns, evidence)
    
    answer = call_llm(prompt)
    confidence = calculate_confidence(patterns, evidence)

    final = answer
    if patterns:
        final += "\n\n🧠 Pattern phát hiện:\n"
        for p in patterns: final += f"- {p['interpretation']} (confidence: {p['confidence']})\n"
    else:
        final += "\n\n🧠 Pattern phát hiện:\n- Không phát hiện pattern rõ ràng\n"
    
    final += f"\n🎯 Confidence: {confidence}\n\n📚 References:\n"
    for i, e in enumerate(evidence[:3], 1):
        final += f"[{i}] {e.get('source')} - p.{e.get('page')}\n"
    final += get_book_intro()
    return final