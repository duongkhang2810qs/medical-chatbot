import json
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

CBC50_JSON_PATH = Path(__file__).with_name("cbc_50_book_evidence_pack_v1.json")

_CBC50_PACK = None


CBC_TEST_CODES = [
    "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW",
    "WBC", "NEUT", "LYMPH", "MONO", "EOS", "BASO", "IG", "PLT"
]


def _load_pack():
    global _CBC50_PACK
    if _CBC50_PACK is None:
        with open(CBC50_JSON_PATH, "r", encoding="utf-8") as f:
            _CBC50_PACK = json.load(f)
    return _CBC50_PACK


def _normalize_text(s: str) -> str:
    s = s or ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^\w\s%]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _detect_test_code(text: str):
    upper = (text or "").upper()

    # Ưu tiên match token độc lập
    for code in CBC_TEST_CODES:
        if re.search(rf"\b{re.escape(code)}\b", upper):
            return code

    # Một số alias tiếng Việt / tiếng Anh phổ biến
    aliases = {
        "bạch cầu": "WBC",
        "bach cau": "WBC",
        "hồng cầu": "RBC",
        "hong cau": "RBC",
        "tiểu cầu": "PLT",
        "tieu cau": "PLT",
        "huyết sắc tố": "HGB",
        "huyet sac to": "HGB",
        "hemoglobin": "HGB",
        "hematocrit": "HCT",
    }

    norm = _normalize_text(text)
    for k, v in aliases.items():
        if _normalize_text(k) in norm:
            return v

    return None


def _intent_kind(text: str):
    norm = _normalize_text(text)

    if any(x in norm for x in ["la gi", "là gì", "viet tat", "viết tắt", "meaning", "definition"]):
        return "definition"

    if any(x in norm for x in ["cao", "tang", "tăng", "high"]):
        return "high"

    if any(x in norm for x in ["thap", "thấp", "giam", "giảm", "low"]):
        return "low"

    if any(x in norm for x in ["binh thuong", "bình thường", "normal", "range", "tham chieu"]):
        return "normal_range"

    return None


def find_cbc50_case(user_text: str):
    pack = _load_pack()
    cases = pack.get("cases", [])

    q_norm = _normalize_text(user_text)
    code = _detect_test_code(user_text)
    intent = _intent_kind(user_text)

    # 1. Exact normalized match
    for case in cases:
        if _normalize_text(case.get("question", "")) == q_norm:
            return case

    # 2. Fuzzy question match
    best_case = None
    best_score = 0.0
    for case in cases:
        cq = _normalize_text(case.get("question", ""))
        score = SequenceMatcher(None, q_norm, cq).ratio()
        if score > best_score:
            best_score = score
            best_case = case

    if best_case and best_score >= 0.82:
        return best_case

    # 3. Match theo chỉ số + intent
    if code:
        candidates = [
            c for c in cases
            if code in [str(x).upper() for x in c.get("tests", [])]
            or code == str(c.get("indicator", "")).upper()
        ]

        if intent:
            intent_candidates = [
                c for c in candidates
                if intent in _normalize_text(c.get("question", ""))
                or (
                    intent == "definition"
                    and any(x in _normalize_text(c.get("question", "")) for x in ["la gi", "là gì"])
                )
                or (
                    intent == "high"
                    and any(x in _normalize_text(c.get("question", "")) for x in ["cao", "tang", "tăng"])
                )
                or (
                    intent == "low"
                    and any(x in _normalize_text(c.get("question", "")) for x in ["thap", "thấp", "giam", "giảm"])
                )
            ]

            if intent_candidates:
                return intent_candidates[0]

        # Nếu không rõ intent, ưu tiên câu định nghĩa
        for c in candidates:
            cq = _normalize_text(c.get("question", ""))
            if "la gi" in cq or "là gì" in cq:
                return c

        if candidates:
            return candidates[0]

    return None


def get_cbc50_evidence_for_question(user_text: str):
    case = find_cbc50_case(user_text)
    if not case:
        return None

    evidence = case.get("evidence", [])
    normalized = []

    for ev in evidence:
        normalized.append({
            "source": ev.get("source", "clinical_hematology.pdf"),
            "page": ev.get("page"),
            "text": ev.get("text") or ev.get("quote") or ev.get("content") or "",
            "id": ev.get("id") or ev.get("evidence_id"),
            "origin": "cbc50_local_json"
        })

    return normalized