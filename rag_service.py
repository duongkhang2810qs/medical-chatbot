import json
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

BASE_DIR = Path(__file__).resolve().parent
PATTERN_PATH = BASE_DIR / "demo_cases_all_clean.json"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return []

print("🔧 Loading Qdrant and Models for RAG Service...")
try:
    PATTERN_DATA = load_json(PATTERN_PATH)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(host="localhost", port=6333)
    print("✅ Đã kết nối Qdrant thành công!")
except Exception as e:
    print(f"⚠️ Lỗi khởi tạo RAG: {e}")
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
                "status": st,
                "unit": item.get("unit")
            })
    return tags, abnormal_items

def match_patterns(tags, pattern_data):
    matched = []
    for case in pattern_data:
        case_tags = {normalize_tag(t) for t in case.get("input_tags", [])}
        overlap = case_tags & set(tags)
        if len(overlap) >= min(2, max(len(case_tags), 1)):
            matched.append({
                "pattern_name": case.get("pattern_name"),
                "interpretation": case.get("combined_interpretation"),
                "confidence": round(len(overlap) / max(len(case_tags), 1), 2)
            })
    matched.sort(key=lambda x: x["confidence"], reverse=True)
    return matched[:3]

def search_qdrant(query, model, client, top_k=5):
    if not model or not client: return []
    emb = model.encode([query])[0].tolist()
    
    try:
        results = client.query_points(collection_name="cbc", query=emb, limit=top_k)
        hits = results.points
    except AttributeError:
        hits = client.search(collection_name="cbc", query_vector=emb, limit=top_k)

    output = []
    for r in hits:
        output.append({
            "score": float(r.score),
            "text": r.payload.get("text"),
            "source": r.payload.get("source"),
            "page": r.payload.get("page")
        })
    return output

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

def build_prompt(abnormal_items, patterns, evidence):
    abnormal_text = "\n".join([f"- 🔴 {x['test']}: {x['status']} ({x['value']})" for x in abnormal_items])
    pattern_text = "\n".join([f"- {p['interpretation']}" for p in patterns])
    if not pattern_text: pattern_text = "Không phát hiện pattern rõ ràng, dựa vào từng chỉ số riêng lẻ."
    evidence_text = "\n".join([f"[{i+1}] \"{e['text'][:200]}\" ({e.get('source')}, p.{e.get('page')})" for i, e in enumerate(evidence[:3])])

    return f"""
You are a hematology specialist with strong clinical reasoning.

Your task:
Analyze CBC test abnormalities and provide evidence-based interpretation.

=====================
📊 ABNORMAL FINDINGS
{abnormal_text}

🧠 PATTERN
{pattern_text}

📚 EVIDENCE
{evidence_text}
=====================

IMPORTANT RULES:

1. 🔴 ONLY use information from the "EVIDENCE"
2. 🔴 DO NOT hallucinate or invent information
3. 🔴 ONLY cite from the provided EVIDENCE items
4. 🔴 DO NOT fabricate sources or page numbers
5. 🔴 Each citation [i] MUST correspond exactly to EVIDENCE [i]
6. 🔴 If you use evidence [1], cite as [1] only
7. 🔴 NEVER cite [4], [5], ... if only [1], [2], [3] are provided
8. 🔴 If evidence is insufficient, explicitly say the evidence is limited
9. 🔴 Ignore irrelevant evidence

CITATION FORMAT RULES:

- Use citation format like [1], [2], [3]
- The citation index must match the exact EVIDENCE item number
- Do not invent book names or page numbers outside the provided EVIDENCE
- When mentioning a medical claim, attach the matching citation right in the sentence

=====================

CLINICAL REQUIREMENTS:

- Use proper medical reasoning
- Connect abnormal findings logically
- If possible:
  → Suggest likely diagnoses (NOT definitive)
  → Assess severity (mild / moderate / severe)

=====================

OUTPUT FORMAT (WRITE IN VIETNAMESE):

### 1. Tóm tắt
- Liệt kê bất thường chính (dùng 🔴)

### 2. Giải thích (có citation)
- Giải thích từng bất thường
- MUST include citation in sentence
- ONLY use citation numbers that exist in EVIDENCE

### 3. Nguyên nhân
- Liệt kê nguyên nhân khả dĩ
- If evidence is weak, say it clearly

### 4. Chẩn đoán gợi ý
- In đậm (**...**)
- Không khẳng định tuyệt đối

### 5. Đánh giá mức độ
- Nhẹ / Trung bình / Nặng
- Giải thích ngắn

### 6. Lưu ý lâm sàng
- Đề xuất xét nghiệm thêm
- Cảnh báo quan trọng

### 7. Clinical impression
- Viết 1 câu kết luận ngắn gọn, súc tích (1 dòng duy nhất)

=====================

NOTES:

- Final answer MUST be in Vietnamese
- Prioritize accuracy over length
- Use professional clinical tone
"""

def call_deepseek(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": "deepseek/deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def call_gemini(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return res.text.strip()

def call_llm(prompt):
    try: return call_deepseek(prompt)
    except: return call_gemini(prompt)

def analyze_indicators_with_llm(user_indicators: list) -> str:
    tags, abnormal_items = extract_abnormal(user_indicators)
    if not abnormal_items:
        return "Kết quả xét nghiệm của bạn nằm trong giới hạn tham chiếu. Không phát hiện chỉ số bất thường nào."

    patterns = match_patterns(tags, PATTERN_DATA)
    query = " ".join(tags) + " clinical interpretation cbc"
    evidence = search_qdrant(query, embedding_model, qdrant_client)
    evidence = filter_evidence(evidence, abnormal_items)

    prompt = build_prompt(abnormal_items, patterns, evidence)
    answer = call_llm(prompt)
    confidence = calculate_confidence(patterns, evidence)

    final = answer
    if patterns:
        final += "\n\n🧠 Pattern phát hiện:\n"
        for p in patterns: final += f"- {p['interpretation']} (confidence: {p['confidence']})\n"
    else:
        final += "\n\n🧠 Pattern phát hiện:\n- Không phát hiện pattern đặc thù\n"
    
    final += f"\n🎯 Confidence: {confidence}\n\n📚 References:\n"
    for i, e in enumerate(evidence[:3], 1):
        final += f"[{i}] {e.get('source')} - p.{e.get('page')}\n"
    return final