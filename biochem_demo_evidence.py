import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

# Đặt file này cùng thư mục với chatbot_service.py
# và đặt biochem_50_book_evidence_pack_v1.json cùng thư mục.
_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_PACK_PATH = _BASE_DIR / "biochem_50_book_evidence_pack_v1.json"

_BIOCHEM50_PACK: Optional[Dict[str, Any]] = None

BIOCHEM_TEST_CODES = [
    "AST", "ALT", "UREA", "BUN", "CREATININE", "NA", "K", "CL",
    "GLUCOSE", "HBA1C", "CHOLESTEROL", "TRIGLYCERIDE", "HDL_C", "LDL_C",
    "CK_MB", "TROPONIN_T", "PRO_BNP", "BNP", "NT_PROBNP", "FERRITIN",
    "ALBUMIN", "CALCIUM_ION", "PTH", "URIC_ACID",
]

ALIASES = {
    "men gan": ["AST", "ALT"],
    "ast": ["AST"],
    "alt": ["ALT"],
    "urea": ["UREA"],
    "bun": ["UREA"],
    "creatinine": ["CREATININE"],
    "creatinin": ["CREATININE"],
    "natri": ["NA"],
    "sodium": ["NA"],
    "na": ["NA"],
    "kali": ["K"],
    "potassium": ["K"],
    "clo": ["CL"],
    "chloride": ["CL"],
    "glucose": ["GLUCOSE"],
    "duong mau": ["GLUCOSE"],
    "đường máu": ["GLUCOSE"],
    "hba1c": ["HBA1C"],
    "cholesterol": ["CHOLESTEROL"],
    "triglyceride": ["TRIGLYCERIDE"],
    "triglycerid": ["TRIGLYCERIDE"],
    "hdl": ["HDL_C"],
    "hdl c": ["HDL_C"],
    "ldl": ["LDL_C"],
    "ldl c": ["LDL_C"],
    "lipid": ["LDL_C", "HDL_C", "TRIGLYCERIDE", "CHOLESTEROL"],
    "ck mb": ["CK_MB"],
    "ck-mb": ["CK_MB"],
    "troponin": ["TROPONIN_T"],
    "troponin t": ["TROPONIN_T"],
    "pro bnp": ["PRO_BNP"],
    "pro-bnp": ["PRO_BNP"],
    "nt probnp": ["PRO_BNP"],
    "nt-probnp": ["PRO_BNP"],
    "bnp": ["PRO_BNP"],
    "ferritin": ["FERRITIN"],
    "albumin": ["ALBUMIN"],
    "canxi ion": ["CALCIUM_ION"],
    "calcium ion": ["CALCIUM_ION"],
    "ionized calcium": ["CALCIUM_ION"],
    "pth": ["PTH"],
    "acid uric": ["URIC_ACID"],
    "uric acid": ["URIC_ACID"],
    "gout": ["URIC_ACID"],
}


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _normalize_text(text: str) -> str:
    text = _strip_accents(text or "").lower().strip()
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9\s%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_biochem50_pack(path: Optional[str] = None) -> Dict[str, Any]:
    global _BIOCHEM50_PACK
    if _BIOCHEM50_PACK is not None:
        return _BIOCHEM50_PACK

    pack_path = Path(path) if path else _DEFAULT_PACK_PATH
    if not pack_path.exists():
        print(f"⚠️ Không tìm thấy BIOCHEM50 evidence pack: {pack_path}")
        _BIOCHEM50_PACK = {"cases": [], "evidence_library": {}}
        return _BIOCHEM50_PACK

    with pack_path.open("r", encoding="utf-8") as f:
        _BIOCHEM50_PACK = json.load(f)

    return _BIOCHEM50_PACK


def _detect_tests(user_text: str) -> List[str]:
    norm = _normalize_text(user_text)
    upper = (user_text or "").upper()
    found: List[str] = []

    # Match mã xét nghiệm rõ ràng trước.
    token_map = {
        "AST": "AST", "ALT": "ALT", "BUN": "UREA", "UREA": "UREA",
        "CREATININE": "CREATININE", "NA": "NA", "K": "K", "CL": "CL",
        "HBA1C": "HBA1C", "CK-MB": "CK_MB", "CK_MB": "CK_MB",
        "PTH": "PTH", "BNP": "PRO_BNP",
    }
    for token, code in token_map.items():
        if re.search(rf"\b{re.escape(token)}\b", upper):
            if code not in found:
                found.append(code)

    for alias, codes in ALIASES.items():
        if _normalize_text(alias) in norm:
            for code in codes:
                if code not in found:
                    found.append(code)

    return found


def _intent_kind(user_text: str) -> Optional[str]:
    norm = _normalize_text(user_text)
    if any(x in norm for x in ["la gi", "khai niem", "viet tat", "khac gi", "definition", "meaning"]):
        return "definition"
    if any(x in norm for x in ["cao", "tang", "high"]):
        return "high"
    if any(x in norm for x in ["thap", "giam", "low"]):
        return "low"
    if any(x in norm for x in ["chac chan", "co phai", "khong", "benh khong"]):
        return "safety"
    return None


def _case_tests(case: Dict[str, Any]) -> List[str]:
    tests = []
    for x in case.get("tests", []) or []:
        x = str(x).upper().replace("-", "_")
        if x:
            tests.append(x)
    ind = str(case.get("indicator", "")).upper().replace("-", "_")
    for part in re.split(r"[+/\s]+", ind):
        if part and part not in tests:
            tests.append(part)
    return tests


def find_biochem50_case(user_text: str) -> Optional[Dict[str, Any]]:
    """Match câu hỏi demo sinh hóa. Có exact, fuzzy và match theo chỉ số + intent."""
    pack = load_biochem50_pack()
    cases = pack.get("cases", []) or []
    q_norm = _normalize_text(user_text)
    detected = _detect_tests(user_text)
    intent = _intent_kind(user_text)

    # 1) exact normalized question
    for case in cases:
        if _normalize_text(case.get("question", "")) == q_norm:
            return case

    # 2) fuzzy question
    best_case = None
    best_score = 0.0
    for case in cases:
        score = SequenceMatcher(None, q_norm, _normalize_text(case.get("question", ""))).ratio()
        if score > best_score:
            best_score = score
            best_case = case
    if best_case and best_score >= 0.82:
        return best_case

    # 3) indicator/test + intent
    if detected:
        candidates = []
        for case in cases:
            ct = _case_tests(case)
            if any(code in ct for code in detected):
                candidates.append(case)

        if candidates and intent:
            def qn(c): return _normalize_text(c.get("question", ""))
            if intent == "definition":
                for c in candidates:
                    if any(x in qn(c) for x in ["la gi", "khai niem", "khac gi"]):
                        return c
            if intent == "high":
                for c in candidates:
                    if any(x in qn(c) for x in ["cao", "tang"]):
                        return c
            if intent == "low":
                for c in candidates:
                    if any(x in qn(c) for x in ["thap", "giam"]):
                        return c
            if intent == "safety":
                for c in candidates:
                    if any(x in qn(c) for x in ["chac chan", "co phai", "benh khong"]):
                        return c

        if candidates:
            # Nếu không rõ intent, ưu tiên câu khái niệm.
            for c in candidates:
                if "la gi" in _normalize_text(c.get("question", "")):
                    return c
            return candidates[0]

    return None


def get_biochem50_evidence_for_question(user_text: str) -> List[Dict[str, Any]]:
    """Trả về evidence list đúng format chatbot_service.format_evidence_for_prompt đang dùng."""
    case = find_biochem50_case(user_text)
    if not case:
        return []

    evidence: List[Dict[str, Any]] = []
    for ev in case.get("evidence", []) or []:
        evidence.append({
            "id": ev.get("evidence_id") or ev.get("id"),
            "source": ev.get("source", "tietz.pdf"),
            "page": ev.get("page"),
            "panel": ev.get("panel", "BIOCHEM"),
            "text": ev.get("text", ""),
            "tests": ev.get("tests") or case.get("tests") or [],
            "topics": ev.get("topics") or [],
            "conditions": ev.get("conditions") or [],
            "origin": "biochem50_local_json",
            "_biochem50_case_id": case.get("id"),
            "_expected_answer_vi": case.get("expected_answer_vi", ""),
        })

    return evidence


def build_biochem50_prompt_context(user_text: str) -> Optional[str]:
    """Dùng nếu muốn đưa thêm expected answer + evidence vào prompt."""
    case = find_biochem50_case(user_text)
    if not case:
        return None

    lines = [
        f"BIOCHEM50_DEMO_CASE_ID: {case.get('id')}",
        f"Question: {case.get('question')}",
        f"Expected answer guide: {case.get('expected_answer_vi', '')}",
        "Evidence bắt buộc dùng khi liên quan:",
    ]

    for i, ev in enumerate(case.get("evidence", []) or [], start=1):
        lines.append(
            f"[{i}] Source={ev.get('source', 'tietz.pdf')}; "
            f"Page={ev.get('page')}; Panel={ev.get('panel', 'BIOCHEM')}; "
            f"Tests={', '.join(ev.get('tests') or case.get('tests') or [])}; "
            f"Quote=\"{ev.get('text', '')}\""
        )

    return "\n".join(lines)
