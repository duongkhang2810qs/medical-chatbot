# lab_core.py
from __future__ import annotations
import time
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv

try:
    from google import genai
except Exception:
    genai = None

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import (
    TEST_NORMALIZATION,
    TEST_LABELS,
    STATUS_NORMALIZATION,
    SOURCE_TRUST,
    CBC_TEST_MAP,
    BIOCHEM_TEST_MAP,
    CBC_TOPIC_KEYWORDS,
    BIOCHEM_TOPIC_KEYWORDS,
    TYPE_PATTERNS,
    PANEL_PATTERNS,
    CROSS_PANEL_PATTERNS,
    BOUNDARY_REGEX_TEXT,
    MIN_WORDS,
    MAX_WORDS,
    PDF_SKIP_FIRST_PAGES,
    PDF_SKIP_PAGES_BY_FILE,
    KB_QUALITY_THRESHOLD,
    MAX_CHUNKS_PER_PDF,
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    EMBEDDING_MODEL_NAME,
    TOP_K_PER_QUERY,
    MAX_RAW_EVIDENCE,
    MAX_FINAL_EVIDENCE,
    REQUEST_TIMEOUT,
    COLAB_MAX_NEW_TOKENS,
    COLAB_TEMPERATURE,
)


_EMBEDDING_MODEL: SentenceTransformer | None = None
_QDRANT_CLIENT: QdrantClient | None = None
_BOUNDARY_REGEX = re.compile(BOUNDARY_REGEX_TEXT, flags=re.IGNORECASE)


# =========================================================
# BASIC IO
# =========================================================

def load_json(path: Path | str) -> Any:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    data: list[dict] = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except Exception as exc:
                print(f"Warning: cannot parse JSONL line {line_no} in {path}: {exc}")
    return data

def append_jsonl(path: Path | str, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def stable_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()[:n]

def slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None: return default
    try: return float(value)
    except: return default

def ensure_list(value: Any) -> list:
    if value is None: return []
    if isinstance(value, list): return value
    return [value]


# =========================================================
# TEXT CLEANING / PDF PROCESSING
# =========================================================

def clean_text(text: str) -> str:
    text = str(text or "").replace("\x00", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).lower().strip()

def split_sentences(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text: return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def get_pdf_skip_pages(pdf_path: Path | str) -> int:
    pdf_path = Path(pdf_path)
    return int(PDF_SKIP_PAGES_BY_FILE.get(pdf_path.name, PDF_SKIP_FIRST_PAGES))

def read_pdf_pages(pdf_path: Path | str, skip_first_pages: int | None = None) -> list[dict]:
    pdf_path = Path(pdf_path)
    if skip_first_pages is None:
        skip_first_pages = get_pdf_skip_pages(pdf_path)
    pages: list[dict] = []
    if not pdf_path.exists():
        print(f"Warning: PDF not found: {pdf_path}")
        return pages
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page_no = page_index + 1
        if page_no <= skip_first_pages: continue
        try:
            page = doc.load_page(page_index)
            text = clean_text(page.get_text("text"))
            if text: pages.append({"page": page_no, "text": text})
        except Exception as exc:
            print(f"Warning: cannot read page {page_no} from {pdf_path.name}: {exc}")
    doc.close()
    return pages


# =========================================================
# KB BUILD: DETECTION
# =========================================================

def get_panel_test_map(panel: str) -> dict[str, list[str]]:
    panel = str(panel).upper()
    if panel == "CBC": return CBC_TEST_MAP
    if panel == "BIOCHEM": return BIOCHEM_TEST_MAP
    raise ValueError(f"Unknown panel: {panel}")

def get_panel_topic_keywords(panel: str) -> dict[str, list[str]]:
    panel = str(panel).upper()
    if panel == "CBC": return CBC_TOPIC_KEYWORDS
    if panel == "BIOCHEM": return BIOCHEM_TOPIC_KEYWORDS
    raise ValueError(f"Unknown panel: {panel}")

def detect_tests_in_text(text: str, panel: str) -> list[str]:
    text_l = str(text or "").lower()
    found: list[str] = []
    for test, keywords in get_panel_test_map(panel).items():
        for kw in keywords:
            kw_l = str(kw).lower().strip()
            if not kw_l: continue
            pattern = rf"(?<![a-z0-9]){re.escape(kw_l)}(?![a-z0-9])" if len(kw_l) <= 2 else rf"\b{re.escape(kw_l)}\b"
            if re.search(pattern, text_l):
                found.append(test)
                break
    return sorted(set(found))

def detect_topics_in_text(text: str, panel: str) -> list[str]:
    text_l = str(text or "").lower()
    topics: list[str] = []
    for topic, keywords in get_panel_topic_keywords(panel).items():
        for kw in keywords:
            if str(kw).lower().strip() in text_l:
                topics.append(topic)
                break
    return sorted(set(topics))

def classify_chunk_type(text: str) -> str:
    text_l = str(text or "").lower()
    for ctype, patterns in TYPE_PATTERNS.items():
        for pattern in patterns:
            if str(pattern).lower() in text_l:
                return ctype
    return "general"

def extract_keywords(text: str, panel: str) -> list[str]:
    text_l = str(text or "").lower()
    if panel == "CBC":
        candidates = ["anemia", "iron", "deficiency", "infection", "bacterial", "viral", "inflammation", "bleeding", "bone marrow", "thalassemia", "polycythemia", "neutrophilia", "neutropenia", "lymphocytosis", "thrombocytopenia", "macrocytic", "microcytic", "hypochromic", "megaloblastic", "hemolytic", "sepsis", "coagulation", "left shift", "pancytopenia", "leukocytosis", "leukopenia", "eosinophilia", "basophilia", "erythrocytosis"]
    else:
        candidates = ["hepatitis", "hepatocellular", "transaminase", "aminotransferase", "azotemia", "uremia", "acute kidney injury", "chronic kidney disease", "renal function", "electrolyte", "hyponatremia", "hypernatremia", "hypokalemia", "hyperkalemia", "diabetes", "hyperglycemia", "hypoglycemia", "glycemic control", "hba1c", "dyslipidemia", "atherosclerosis", "cardiovascular risk", "lipoprotein", "myocardial infarction", "acute coronary syndrome", "heart failure", "troponin", "natriuretic peptide", "ferritin", "iron deficiency", "albumin", "hypoalbuminemia", "malnutrition", "protein loss", "ionized calcium", "parathyroid hormone", "hypercalcemia", "hypocalcemia", "urate", "gout", "hyperuricemia"]
    return sorted({kw for kw in candidates if kw in text_l})

def infer_conditions_from_text(text: str, panel: str, tests: list[str], topics: list[str], keywords: list[str]) -> list[str]:
    text_l = str(text or "").lower()
    conditions: set[str] = set()

    topic_map = {
        "anemia": ["anemia"], "infection": ["infection"], "inflammation": ["inflammation"], "bleeding": ["bleeding_risk"], "bone_marrow": ["bone_marrow_disorder"], "thalassemia": ["thalassemia"], "polycythemia": ["polycythemia"], "coagulation": ["coagulation_disorder"],
        "liver_injury": ["liver_injury", "hepatocellular_injury"], "renal_function": ["renal_impairment", "reduced_kidney_function"], "electrolyte_disorder": ["electrolyte_disorder"], "diabetes": ["hyperglycemia", "diabetes_risk"], "dyslipidemia": ["dyslipidemia", "cardiovascular_risk"], "cardiac_biomarker": ["myocardial_injury", "cardiac_stress"], "iron_metabolism": ["iron_store_abnormality"], "protein_nutrition": ["hypoalbuminemia"], "bone_mineral": ["bone_mineral_disorder"], "purine_metabolism": ["hyperuricemia", "gout_risk"],
    }

    for topic in [str(x).lower() for x in topics]:
        for cond in topic_map.get(topic, []): conditions.add(cond)

    trigger_map = [
        (r"iron deficiency|iron deficient", "iron_deficiency_anemia"), (r"microcytic", "microcytic_anemia"), (r"macrocytic|megaloblastic|vitamin b12|folate", "macrocytic_anemia"), (r"thalassemia|thalassaemia", "thalassemia"), (r"hemolytic|hemolysis", "hemolytic_anemia"), (r"neutrophilia|bacterial infection", "bacterial_infection"), (r"lymphocytosis|viral infection", "viral_infection"), (r"left shift|immature granulocyte|band cell", "left_shift_stress_response"), (r"thrombocytopenia", "thrombocytopenia"), (r"thrombocytosis", "thrombocytosis"), (r"polycythemia|erythrocytosis", "polycythemia"), (r"leukocytosis", "leukocytosis"), (r"leukopenia", "leukopenia"), (r"neutropenia", "neutropenia"), (r"eosinophilia", "eosinophilia"), (r"basophilia", "basophilia"),
        (r"hepatocellular|aminotransferase|transaminase|hepatitis", "hepatocellular_injury"), (r"renal|kidney|creatinine|urea|azotemia|uremia", "renal_impairment"), (r"hyperglycemia|diabetes|glycemic", "hyperglycemia"), (r"dyslipidemia|lipoprotein|cardiovascular risk|atherosclerosis", "dyslipidemia"), (r"hyperkalemia|hypokalemia|hyponatremia|hypernatremia", "electrolyte_disorder"), (r"hypoalbuminemia|albumin", "hypoalbuminemia"), (r"ferritin|iron stores|iron storage", "iron_store_abnormality"), (r"urate|uric acid|gout|hyperuricemia", "hyperuricemia"), (r"troponin|ck-mb|myocardial|acute coronary syndrome", "myocardial_injury"), (r"natriuretic peptide|heart failure|pro-bnp|bnp", "cardiac_stress"), (r"parathyroid|pth|calcium", "bone_mineral_disorder"),
    ]

    combined = " ".join([text_l] + [str(x).lower() for x in topics] + [str(x).lower() for x in keywords])
    for pattern, condition in trigger_map:
        if re.search(pattern, combined): conditions.add(condition)

    test_condition_hints = {"HGB": ["anemia", "polycythemia"], "RBC": ["anemia", "polycythemia"], "HCT": ["anemia", "polycythemia"], "MCV": ["microcytic_anemia", "macrocytic_anemia"], "MCH": ["microcytic_anemia"], "MCHC": ["red_cell_index_abnormality"], "RDW": ["anisocytosis", "iron_deficiency_anemia"], "WBC": ["infection", "leukocytosis", "leukopenia"], "NEUT": ["bacterial_infection", "neutrophilia", "neutropenia"], "LYMPH": ["viral_infection", "lymphocytosis", "lymphopenia"], "PLT": ["thrombocytopenia", "thrombocytosis"], "IG": ["left_shift_stress_response"]} if panel == "CBC" else {"AST": ["liver_injury"], "ALT": ["liver_injury"], "UREA": ["renal_impairment"], "CREATININE": ["renal_impairment"], "NA": ["electrolyte_disorder"], "K": ["electrolyte_disorder"], "CL": ["electrolyte_disorder"], "GLUCOSE": ["hyperglycemia", "diabetes_risk"], "HBA1C": ["diabetes_risk", "poor_glycemic_control"], "CHOLESTEROL": ["dyslipidemia", "cardiovascular_risk"], "TRIGLYCERIDE": ["dyslipidemia", "cardiovascular_risk"], "HDL_C": ["dyslipidemia", "cardiovascular_risk"], "LDL_C": ["dyslipidemia", "cardiovascular_risk"], "CK_MB": ["myocardial_injury"], "TROPONIN_T": ["myocardial_injury"], "PRO_BNP": ["cardiac_stress"], "FERRITIN": ["iron_store_abnormality"], "ALBUMIN": ["hypoalbuminemia"], "CALCIUM_ION": ["bone_mineral_disorder"], "PTH": ["bone_mineral_disorder"], "URIC_ACID": ["hyperuricemia", "gout_risk"]}
    for test in tests:
        for cond in test_condition_hints.get(test, []):
            if cond.replace("_", " ") in combined or cond in combined: conditions.add(cond)
    return sorted(conditions)

def generate_query_hints(panel: str, tests: list[str], topics: list[str], conditions: list[str]) -> list[str]:
    queries: list[str] = []
    for test in tests:
        queries.extend([f"{test.lower()} abnormal interpretation", f"{test.lower()} clinical significance", f"{test.lower()} causes of increase or decrease"])
        queries.append(f"{test.lower()} complete blood count finding" if panel == "CBC" else f"{test.lower()} clinical chemistry interpretation")
    for topic in topics:
        queries.append(f"{topic} cbc finding" if panel == "CBC" else f"{topic} biochemistry interpretation")
    for condition in conditions:
        queries.append(f"{condition} laboratory interpretation")
    seen = set()
    return [q for q in queries if q.strip() and not (q.strip() in seen or seen.add(q.strip()))]

def is_noise_text(text: str) -> bool:
    text_l = str(text or "").lower().strip()
    words = text_l.split()
    if len(words) < MIN_WORDS: return True
    if any(x in text_l for x in ["table of contents", "contents", "copyright", "isbn", "bibliography", "references", "index", "chapter contents", "permission", "all rights reserved", "printed in", "editorial"]) and len(words) < 80: return True
    numeric_tokens = sum(1 for w in words if w.replace(".", "").replace(",", "").replace("-", "").replace("–", "").isdigit())
    if numeric_tokens / max(len(words), 1) > 0.45: return True
    if text_l.count(" et al") >= 3 or text_l.count(" doi") >= 1: return True
    return False

def score_chunk(text: str, ctype: str, tests: list[str], topics: list[str], conditions: list[str]) -> float:
    text_l = str(text or "").lower()
    words = len(text_l.split())
    points = {"interpretation": 3.0, "cause": 2.0, "definition": 1.0}.get(ctype, 0.0)
    points += 2.5 if len(tests) >= 3 else 1.8 if len(tests) == 2 else 1.0 if len(tests) == 1 else 0.0
    points += min(1.5, len(topics) * 0.5) + min(2.0, len(conditions) * 0.6)
    if any(marker in text_l for marker in ["indicates", "suggests", "consistent with", "associated with", "reflects", "is seen in", "caused by", "due to", "leads to", "result of", "elevation of", "decrease in", "low levels", "high levels"]): points += 1.0
    if 25 <= words <= MAX_WORDS: points += 1.0
    elif words > MAX_WORDS: points += 0.4
    if not tests and not topics: points -= 1.0
    if is_noise_text(text_l): points -= 2.0
    return round(max(0.0, min(1.0, points / 10.0)), 2)

def semantic_chunk_page_text(text: str) -> list[str]:
    sentences = split_sentences(text)
    chunks, current = [], ""
    for sentence in sentences:
        if not sentence.strip() or is_noise_text(sentence.strip()): continue
        if _BOUNDARY_REGEX.search(sentence) is not None and len(current.split()) >= 25:
            if current.strip(): chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip()
        if len(current.split()) >= MAX_WORDS:
            chunks.append(current.strip())
            current = ""
    if len(current.split()) >= MIN_WORDS: chunks.append(current.strip())
    return [c for c in [re.sub(r"\s+", " ", ch).strip() for ch in chunks] if c and not is_noise_text(c) and len(c.split()) >= MIN_WORDS]

def build_kb_from_pdf_paths(pdf_paths: list[Path], panel: str) -> list[dict]:
    panel = str(panel).upper()
    kb: list[dict] = []
    for pdf_path in pdf_paths:
        source = Path(pdf_path).name
        trust = SOURCE_TRUST.get(source, 0.8)
        if not Path(pdf_path).exists(): continue
        pages = read_pdf_pages(pdf_path, skip_first_pages=get_pdf_skip_pages(pdf_path))
        source_count = 0
        for page_obj in pages:
            for chunk in semantic_chunk_page_text(page_obj["text"]):
                tests, topics, keywords = detect_tests_in_text(chunk, panel), detect_topics_in_text(chunk, panel), extract_keywords(chunk, panel)
                ctype = classify_chunk_type(chunk)
                conditions = infer_conditions_from_text(chunk, panel, tests, topics, keywords)
                score = score_chunk(chunk, ctype, tests, topics, conditions)
                if score < KB_QUALITY_THRESHOLD: continue
                item = {
                    "evidence_id": f"ev_{panel.lower()}_{len(kb):06d}_{stable_hash(source + str(page_obj['page']) + chunk)}",
                    "panel": panel, "text": chunk, "bm25_text": chunk, "tests": tests, "topics": topics,
                    "keywords": keywords, "conditions": conditions, "query_hints": generate_query_hints(panel, tests, topics, conditions),
                    "type": ctype, "score": score, "trust": trust, "source": source, "page": page_obj["page"], "source_type": "pdf_book",
                }
                item["embedding_text"] = build_embedding_text(item)
                kb.append(item)
                source_count += 1
                if MAX_CHUNKS_PER_PDF is not None and source_count >= MAX_CHUNKS_PER_PDF: break
            if MAX_CHUNKS_PER_PDF is not None and source_count >= MAX_CHUNKS_PER_PDF: break
    return kb


# =========================================================
# NORMALIZATION FOR CASE INPUT
# =========================================================

def normalize_raw_test_name(test_name: str) -> str:
    return str(test_name or "").strip().replace("-", "_").replace(" ", "_").upper()

def guess_panel(test_name: str) -> str:
    raw = normalize_raw_test_name(test_name)
    for panel, mapping in TEST_NORMALIZATION.items():
        if raw in mapping: return panel
    cbc_tests = {"RBC", "HGB", "HB", "HCT", "MCV", "MCH", "MCHC", "RDW", "RDW_SD", "RDW_CV", "WBC", "NEUT", "LYM", "LYMPH", "MONO", "EOS", "BASO", "PLT", "IG", "NEUT_PERCENT", "NEUT_ABS", "LYM_PERCENT", "LYM_ABS", "MONO_PERCENT", "MONO_ABS", "EOS_PERCENT", "EOS_ABS", "BASO_PERCENT", "BASO_ABS", "IG_PERCENT", "IG_ABS", "PCT", "MPV", "PDW"}
    return "CBC" if raw in cbc_tests else "BIOCHEM"

def normalize_test_name(test_name: str, panel: str | None = None) -> str:
    raw = normalize_raw_test_name(test_name)
    panel = str(panel).upper() if panel else guess_panel(raw)
    return TEST_NORMALIZATION.get(panel, {}).get(raw, raw)

def normalize_status(status: Any) -> str:
    return STATUS_NORMALIZATION.get(str(status or "").strip().lower(), str(status or "").strip().lower())

def normalize_panel(panel: Any, test_name: str = "") -> str:
    p = str(panel).strip().upper() if panel else ""
    if p in {"CBC", "HEMATOLOGY", "HAEMATOLOGY"}: return "CBC"
    if p in {"BIOCHEM", "BIOCHEMISTRY", "CHEMISTRY", "CLINICAL_CHEMISTRY"}: return "BIOCHEM"
    return guess_panel(test_name)

def get_case_id(case: dict, idx: int = 0) -> str:
    for key in ["case_id", "id", "image_id", "img_id", "file_name", "filename"]:
        if case.get(key): return str(case[key])
    return f"case_{idx:04d}"

def format_ref_range(ref_range: Any) -> str:
    if not ref_range: return ""
    if isinstance(ref_range, dict):
        ref_min, ref_max = ref_range.get("ref_min"), ref_range.get("ref_max")
        if ref_min is not None and ref_max is not None: return f"{ref_min} - {ref_max}"
        return json.dumps(ref_range, ensure_ascii=False)
    return str(ref_range)

def extract_case_items(case: dict | list, default_panel: str | None = None) -> list[dict]:
    raw_items = case if isinstance(case, list) else (case.get("data") or case.get("results") or case.get("items") or case.get("lab_results") or [])
    output: list[dict] = []
    for item in raw_items:
        if not isinstance(item, dict): continue
        raw_test = item.get("test_name") or item.get("name") or item.get("test") or item.get("parameter") or item.get("analyte") or ""
        if not raw_test: continue
        panel = normalize_panel(item.get("panel") or default_panel, raw_test)
        test = normalize_test_name(raw_test, panel)
        output.append({
            "panel": panel, "raw_test": raw_test, "test": test, "test_label": TEST_LABELS.get(test, test),
            "value": item.get("value"), "raw_value": item.get("raw_value"), "unit": item.get("unit", ""),
            "status": normalize_status(item.get("status", "")),
            "reference_range": format_ref_range(item.get("reference_range") or item.get("ref_range") or item.get("ref") or item.get("range") or ""),
            "raw_text_line": item.get("raw_text_line", ""), "source_case_item": item,
        })
    return output

def normalize_single_case(case: dict, idx: int = 0, default_panel: str | None = None) -> dict:
    return {"case_id": get_case_id(case, idx), "data": extract_case_items(case, default_panel=default_panel), "raw_case": case}

def merge_case_lists(cbc_cases: list[dict], biochem_cases: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for panel, cases in [("CBC", cbc_cases), ("BIOCHEM", biochem_cases)]:
        for idx, case in enumerate(cases):
            norm = normalize_single_case(case, idx, default_panel=panel)
            case_id = norm["case_id"]
            merged.setdefault(case_id, {"case_id": case_id, "data": [], "raw_sources": {}})
            merged[case_id]["data"].extend(norm["data"])
            merged[case_id]["raw_sources"][panel] = case
    return sorted(list(merged.values()), key=lambda x: str(x.get("case_id", "")))

def extract_lab_items(case: dict) -> list[dict]:
    output: list[dict] = []
    for item in case.get("data", []):
        if not isinstance(item, dict): continue
        raw_test = item.get("raw_test") or item.get("test_name") or item.get("test") or ""
        test = item.get("test") or normalize_test_name(raw_test, item.get("panel"))
        panel = normalize_panel(item.get("panel"), raw_test or test)
        output.append({
            "panel": panel, "raw_test": raw_test or test, "test": test, "test_label": TEST_LABELS.get(test, test),
            "value": item.get("value"), "raw_value": item.get("raw_value"), "unit": item.get("unit", ""),
            "status": normalize_status(item.get("status", "")),
            "reference_range": format_ref_range(item.get("reference_range") or item.get("ref_range") or item.get("ref") or ""),
            "raw_text_line": item.get("raw_text_line", ""), "source_case_item": item.get("source_case_item", item),
        })
    return output

def extract_abnormal_items(items: list[dict]) -> list[dict]:
    return [dict(item, status=normalize_status(item.get("status"))) for item in items if normalize_status(item.get("status")) not in ["", "normal"]]

# =========================================================
# STATIC PATTERNS & SAFETY WARNINGS (Chuyển từ run_final.py sang)
# =========================================================

def slug_condition(text: str) -> str:
    text = str(text or "").lower().strip()
    out, prev_is_sep = [], False
    for ch in text:
        if ch.isalnum():
            out.append(ch); prev_is_sep = False
        else:
            if not prev_is_sep: out.append("_"); prev_is_sep = True
    return "".join(out).strip("_")

def build_static_evidence_from_context(ctx: dict, max_items: int = 3) -> list[dict]:
    evidence = []
    for idx, pattern in enumerate(ctx.get("static_context", [])[:max_items], start=1):
        text_parts = []
        if pattern.get("pattern_name"): text_parts.append(str(pattern["pattern_name"]))
        if pattern.get("description"): text_parts.append(str(pattern["description"]))
        extra = pattern.get("extra", {}) or {}
        if extra.get("clinical_flags"): text_parts.append("Lưu ý lâm sàng: " + "; ".join(extra["clinical_flags"][:2]))
        if extra.get("next_steps"): text_parts.append("Gợi ý đánh giá tiếp: " + "; ".join(extra["next_steps"][:2]))
        text = ". ".join([x for x in text_parts if x]).strip()
        if not text: continue
        evidence.append({
            "evidence_id": f"static_pattern_{idx}", "panel": pattern.get("panel", "STATIC"), "text": text,
            "tests": [], "conditions": pattern.get("conditions", []), "topics": [], "keywords": [],
            "type": "static_pattern_context", "score": 1.0, "final_score": 2.0, "trust": 0.95,
            "source": pattern.get("source", "static_pattern_file"), "page": "static", "source_type": "static_pattern_context", "is_static": True,
        })
    return evidence

def has_case_finding(ctx: dict, panel: str, test: str, status: str) -> bool:
    panel = str(panel).upper()
    test = normalize_test_name(test, panel)
    status = normalize_status(status)
    for item in ctx.get("abnormal_items", []):
        if item.get("panel") == panel and item.get("test") == test and normalize_status(item.get("status")) == status:
            return True
    return False

def case_simple_tags(ctx: dict) -> set[str]:
    tags = set()
    for item in ctx.get("abnormal_items", []):
        test, status = item.get("test"), normalize_status(item.get("status"))
        if test and status: tags.add(f"{test}_{status.capitalize()}")
    return tags

def match_cbc_demo_patterns(ctx: dict, demo_patterns: list[dict]) -> list[dict]:
    matches = []
    for row in demo_patterns:
        raw_input = row.get("input", {}) or {}
        if not isinstance(raw_input, dict): continue
        required = [(normalize_test_name(t, "CBC"), normalize_status(s)) for t, s in raw_input.items()]
        if not required: continue
        hit = sum(1 for t, s in required if has_case_finding(ctx, "CBC", t, s))
        ratio = hit / max(len(required), 1)
        if hit == 0 or (len(required) >= 2 and ratio < 0.6) or (len(required) == 1 and ratio < 1.0): continue
        
        patterns = row.get("patterns", []) or []
        if not patterns:
            patterns = [{"pattern_name": row.get("case_id", "CBC demo pattern"), "interpretation": row.get("combined_interpretation", ""), "confidence": row.get("confidence", "medium"), "match_score": ratio, "matched_conditions": []}]
        for pattern in patterns:
            pattern_name = pattern.get("pattern_name", "CBC demo pattern")
            matches.append({
                "pattern_id": f"cbc_static_{slug_condition(pattern_name)}", "pattern_name": pattern_name,
                "panel": "CBC", "conditions": [slug_condition(pattern_name)],
                "description": pattern.get("interpretation") or row.get("combined_interpretation", ""),
                "confidence": round(safe_float(pattern.get("match_score"), ratio), 2),
                "matched_required": [f"{t}_{s}" for t, s in required], "matched_optional": [], "source": "cbc_demo_cases", "static_rule": True,
            })
    return matches

def match_biochem_static_patterns(ctx: dict, biochem_patterns: dict) -> list[dict]:
    matches = []
    tags = case_simple_tags(ctx)
    single = biochem_patterns.get("single_test_patterns", {}) or {}

    for item in ctx.get("abnormal_items", []):
        if item.get("panel") != "BIOCHEM": continue
        test, status = item.get("test"), normalize_status(item.get("status"))
        if not test or not status: continue
        status_rule = single.get(test, {}).get(status, {}) or {}
        if not status_rule: continue
        label = status_rule.get("label") or f"{test} {status}"
        matches.append({
            "pattern_id": f"biochem_single_{test}_{status}", "pattern_name": label, "panel": "BIOCHEM",
            "conditions": [slug_condition(label)], "description": status_rule.get("clinical_meaning") or status_rule.get("note") or "",
            "confidence": 0.85, "matched_required": [f"{test}_{status}"], "matched_optional": [], "source": "biochem_patterns_single", "static_rule": True,
            "extra": {"associated_tests": status_rule.get("associated_tests", []), "clinical_flags": status_rule.get("clinical_flags", []), "causes": status_rule.get("causes", [])},
        })

    for combo in biochem_patterns.get("pattern_combinations", []) or []:
        req, opt, conf_req = combo.get("required_tags", []), combo.get("optional_tags", []), int(combo.get("confidence_required", 1) or 1)
        req_hits, opt_hits = [t for t in req if t in tags], [t for t in opt if t in tags]
        if (req and len(req_hits) < len(req)) or (not req and len(opt_hits) < conf_req): continue
        match_score = (len(req_hits) + len(opt_hits)) / max(len(req) + len(opt), 1)
        matches.append({
            "pattern_id": combo.get("pattern_id", "biochem_combo_pattern"), "pattern_name": combo.get("name", "BIOCHEM combination pattern"), "panel": "BIOCHEM",
            "conditions": [slug_condition(combo.get("pattern_id", combo.get("name", "biochem_pattern")))],
            "description": combo.get("interpretation", ""), "confidence": round(match_score, 2),
            "matched_required": [x.lower() for x in req_hits], "matched_optional": [x.lower() for x in opt_hits], "source": "biochem_patterns_combo", "static_rule": True,
            "extra": {"next_steps": combo.get("next_steps", []), "sources": combo.get("sources", []), "severity_escalators": combo.get("severity_escalators", {})},
        })
    return matches

def build_safety_warnings(ctx: dict, biochem_patterns: dict) -> list[dict]:
    warnings = []
    for item in ctx.get("abnormal_items", []):
        if item.get("panel") != "BIOCHEM": continue
        test, value = item.get("test"), safe_float(item.get("value"), None)
        if value is None: continue
        if test == "K" and value >= 6.5: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Kali ≥ 6.5 mmol/L: nguy cơ rối loạn nhịp tim đe dọa tính mạng, cần ECG và xử trí cấp cứu."})
        if test == "K" and value <= 2.5: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Kali ≤ 2.5 mmol/L: nguy cơ loạn nhịp và yếu/liệt cơ, cần xử trí y tế."})
        if test == "NA" and value <= 120: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Natri ≤ 120 mmol/L: hạ natri nặng, nguy cơ phù não/co giật."})
        if test == "NA" and value >= 160: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Natri ≥ 160 mmol/L: tăng natri nặng, nguy cơ rối loạn tri giác/hôn mê."})
        if test == "GLUCOSE" and value < 3.0: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Glucose < 3.0 mmol/L: hạ đường huyết nặng, cần xử trí ngay."})
        if test == "GLUCOSE" and value >= 20: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Glucose ≥ 20 mmol/L: tăng đường huyết nặng, cần đánh giá DKA/HHS."})
        if test == "CALCIUM_ION" and value >= 1.75: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Canxi ion hóa ≥ 1.75 mmol/L: tăng canxi nặng, có thể là cấp cứu."})
        if test == "CALCIUM_ION" and value <= 0.80: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Canxi ion hóa ≤ 0.80 mmol/L: giảm canxi nặng, nguy cơ co giật/loạn nhịp."})
        if test == "TRIGLYCERIDE" and value >= 11.3: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Triglyceride ≥ 11.3 mmol/L: nguy cơ viêm tụy cấp rất cao."})
        if test == "CREATININE" and value >= 354: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Creatinine ≥ 354 µmol/L: gợi ý suy thận nặng hoặc AKI nặng, cần đánh giá thận học."})
        if test == "UREA" and value >= 25: warnings.append({"test": test, "value": value, "unit": item.get("unit", ""), "level": "critical", "message": "Urê ≥ 25 mmol/L: tăng urê rất cao, cần đánh giá kèm triệu chứng và điện giải."})
    return warnings

def augment_reasoning_context_with_static_patterns(ctx: dict, cbc_demo_patterns: list[dict], biochem_patterns: dict) -> dict:
    ctx = dict(ctx)
    static_context = match_cbc_demo_patterns(ctx, cbc_demo_patterns) + match_biochem_static_patterns(ctx, biochem_patterns)
    detected_patterns = ctx.get("detected_patterns", [])
    ctx["detected_patterns"] = detected_patterns
    ctx["conditions"] = sorted({condition for pattern in detected_patterns for condition in pattern.get("conditions", [])})
    ctx["static_context"] = static_context
    ctx["safety_warnings"] = build_safety_warnings(ctx, biochem_patterns)
    return ctx

# =========================================================
# PATTERN DETECTION
# =========================================================

def has_finding(abnormal_items: list[dict], panel: str, test: str, status: str) -> bool:
    panel = str(panel).upper()
    test = normalize_test_name(test, panel)
    status = normalize_status(status)
    for item in abnormal_items:
        if str(item.get("panel", "")).upper() == panel and item.get("test") == test and normalize_status(item.get("status")) == status: return True
    return False

def detect_panel_patterns(abnormal_items: list[dict]) -> list[dict]:
    detected: list[dict] = []
    for panel, rules in PANEL_PATTERNS.items():
        if not [x for x in abnormal_items if x.get("panel") == panel]: continue
        for rule in rules:
            requires, optional = rule.get("requires", []), rule.get("optional", [])
            req_hits = [f"{t}_{s}" for t, s in requires if has_finding(abnormal_items, panel, t, s)]
            if requires and len(req_hits) < len(requires): continue
            opt_hits = [f"{t}_{s}" for t, s in optional if has_finding(abnormal_items, panel, t, s)]
            confidence = min(0.95, max(0.6, (len(req_hits) + 0.5 * len(opt_hits)) / (len(requires) + max(len(optional), 1))))
            detected.append({"pattern_id": rule["pattern_id"], "pattern_name": rule["name"], "panel": panel, "conditions": rule.get("conditions", []), "description": rule.get("description", ""), "confidence": round(confidence, 2), "matched_required": req_hits, "matched_optional": opt_hits, "source": "panel_rule"})
    return sorted(detected, key=lambda x: x.get("confidence", 0), reverse=True)

def detect_cross_panel_patterns(abnormal_items: list[dict]) -> list[dict]:
    if not {"CBC", "BIOCHEM"}.issubset({item.get("panel") for item in abnormal_items}): return []
    detected: list[dict] = []
    for rule in CROSS_PANEL_PATTERNS:
        requires, optional = rule.get("requires", []), rule.get("optional", [])
        req_hits = [f"{p}_{t}_{s}" for p, t, s in requires if has_finding(abnormal_items, p, t, s)]
        if requires and len(req_hits) < len(requires): continue
        opt_hits = [f"{p}_{t}_{s}" for p, t, s in optional if has_finding(abnormal_items, p, t, s)]
        confidence = min(0.95, max(0.65, (len(req_hits) + 0.5 * len(opt_hits)) / (len(requires) + max(len(optional), 1))))
        detected.append({"pattern_id": rule["pattern_id"], "pattern_name": rule["name"], "panel": "CROSS_PANEL", "conditions": rule.get("conditions", []), "description": rule.get("description", ""), "confidence": round(confidence, 2), "matched_required": req_hits, "matched_optional": opt_hits, "source": "cross_panel_rule"})
    return sorted(detected, key=lambda x: x.get("confidence", 0), reverse=True)

def build_reasoning_context(case: dict, idx: int = 0) -> dict:
    case_id = get_case_id(case, idx)
    items = extract_lab_items(case)
    abnormal_items = extract_abnormal_items(items)
    detected_patterns = sorted(detect_panel_patterns(abnormal_items) + detect_cross_panel_patterns(abnormal_items), key=lambda x: x.get("confidence", 0), reverse=True)
    return {
        "case_id": case_id, "panels": sorted({x["panel"] for x in items if x.get("panel")}),
        "items": items, "abnormal_items": abnormal_items, "abnormal_tests": sorted({x["test"] for x in abnormal_items if x.get("test")}),
        "detected_patterns": detected_patterns, "conditions": sorted({condition for pattern in detected_patterns for condition in pattern.get("conditions", [])}),
    }

# =========================================================
# KB NORMALIZATION / EMBEDDING TEXT
# =========================================================

def build_embedding_text(item: dict) -> str:
    parts = [f"Panel: {item.get('panel', '')}", f"Clinical text: {item.get('text', '')}"]
    if item.get("tests"): parts.append("Tests: " + ", ".join(item["tests"]))
    if item.get("topics"): parts.append("Topics: " + ", ".join(item["topics"]))
    if item.get("conditions"): parts.append("Conditions: " + ", ".join(item["conditions"]))
    if item.get("keywords"): parts.append("Keywords: " + ", ".join(item["keywords"]))
    if item.get("type"): parts.append("Type: " + str(item["type"]))
    if item.get("tests"): parts.append("Clinical interpretation of " + ", ".join(item["tests"]))
    return " | ".join(parts)

def normalize_kb_item(item: dict, panel: str, idx: int = 0) -> dict:
    x = dict(item)
    panel, source, text = str(panel).upper(), str(x.get("source", "Unknown")), str(x.get("text", ""))
    x.update({
        "panel": panel, "evidence_id": x.get("evidence_id") or f"ev_{panel.lower()}_{idx:06d}_{stable_hash(text)}",
        "tests": sorted({normalize_test_name(t, panel) for t in x.get("tests", []) if t}), "topics": x.get("topics", []), "keywords": x.get("keywords", []),
        "type": x.get("type", "general"), "score": safe_float(x.get("score"), 0.3), "trust": safe_float(x.get("trust"), SOURCE_TRUST.get(source, 0.8)), "source_type": x.get("source_type", "pdf_book"),
    })
    if not x.get("conditions"): x["conditions"] = infer_conditions_from_text(text=text, panel=panel, tests=x["tests"], topics=x["topics"], keywords=x["keywords"])
    x["embedding_text"] = x.get("embedding_text") or build_embedding_text(x)
    return x

# =========================================================
# QUERY BUILDING & QDRANT RETRIEVAL
# =========================================================

def build_query_hints(reasoning_context: dict) -> list[str]:
    queries = []
    for item in reasoning_context.get("abnormal_items", []):
        panel, test, status = item.get("panel"), item.get("test"), item.get("status")
        if not test or not status: continue
        queries.extend([f"{panel} {test} {status} interpretation", f"{test} {status} clinical significance", f"{test} {status} causes"])
        queries.append(f"{test} {status} complete blood count interpretation" if panel == "CBC" else f"{test} {status} clinical chemistry interpretation")
    for pattern in reasoning_context.get("detected_patterns", []):
        if pattern.get("pattern_name"): queries.append(pattern["pattern_name"])
        if pattern.get("description"): queries.append(pattern["description"])
        for condition in pattern.get("conditions", []): queries.append(f"{condition} laboratory interpretation")
    if reasoning_context.get("abnormal_tests"): queries.append(" ".join(reasoning_context.get("abnormal_tests")) + " laboratory abnormal pattern")
    queries.extend(["complete blood count interpretation", "clinical chemistry interpretation", "laboratory test abnormal pattern interpretation"])
    seen = set()
    return [q for q in queries if q.strip() and not (q.strip() in seen or seen.add(q.strip()))]

def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None: _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL

def get_qdrant_client() -> QdrantClient:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None: _QDRANT_CLIENT = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _QDRANT_CLIENT

def qdrant_search(query: str, top_k: int = TOP_K_PER_QUERY) -> list[dict]:
    client, vector = get_qdrant_client(), get_embedding_model().encode([f"Clinical laboratory interpretation query: {query}"])[0].tolist()
    results = client.search(collection_name=COLLECTION_NAME, query_vector=vector, limit=top_k, with_payload=True) if hasattr(client, "search") else client.query_points(collection_name=COLLECTION_NAME, query=vector, limit=top_k, with_payload=True).points
    evidence = []
    for hit in results:
        payload, hit_score = hit.payload or {}, float(getattr(hit, "score", 0) or 0)
        evidence.append({
            "evidence_id": payload.get("evidence_id") or f"ev_{stable_hash(payload.get('text', ''))}", "panel": payload.get("panel", "UNKNOWN"),
            "text": payload.get("text", ""), "bm25_text": payload.get("bm25_text", ""), "tests": payload.get("tests", []), "topics": payload.get("topics", []),
            "conditions": payload.get("conditions", []), "keywords": payload.get("keywords", []), "type": payload.get("type", "general"),
            "score": hit_score, "kb_score": payload.get("score", 0), "trust": float(payload.get("trust", 0.8) or 0.8),
            "source": payload.get("source", "Unknown source"), "page": payload.get("page", ""), "source_type": payload.get("source_type", "pdf_book"), "is_static": False,
        })
    return evidence

def dedup_evidence(evidence: list[dict]) -> list[dict]:
    seen, output = set(), []
    for e in evidence:
        fp = re.sub(r"\s+", " ", str(e.get("text", "")).lower()).strip()[:260]
        if fp and fp not in seen:
            seen.add(fp); output.append(e)
    return output

def rerank_evidence(evidence: list[dict], reasoning_context: dict) -> list[dict]:
    abnormal_tests, panels, conditions = set(reasoning_context.get("abnormal_tests", [])), set(reasoning_context.get("panels", [])), set(reasoning_context.get("conditions", []))
    ranked = []
    for e in evidence:
        score = safe_float(e.get("score"), 0.0) + safe_float(e.get("trust"), 0.8) * 0.10 + safe_float(e.get("kb_score"), 0.0) * 0.12
        if set(e.get("tests", [])) & abnormal_tests: score += 0.30
        if set(e.get("conditions", [])) & conditions: score += 0.30
        if e.get("panel") in panels: score += 0.18
        score += {"interpretation": 0.12, "cause": 0.08, "definition": 0.03}.get(e.get("type"), 0)
        x = dict(e); x["final_score"] = round(score, 4); ranked.append(x)
    return sorted(ranked, key=lambda x: x.get("final_score", 0), reverse=True)

def retrieve_evidence(reasoning_context: dict) -> list[dict]:
    queries, all_evidence = build_query_hints(reasoning_context), []
    for query in queries[:10]:
        try: all_evidence.extend(qdrant_search(query, top_k=TOP_K_PER_QUERY))
        except Exception as exc: print(f"Qdrant search failed for query='{query}': {exc}")
    return rerank_evidence(dedup_evidence(all_evidence), reasoning_context)[:MAX_RAW_EVIDENCE]

# =========================================================
# REASONING PATHS / GRAPH EXPLAINABILITY
# =========================================================

def build_reasoning_paths(reasoning_context: dict, evidence: list[dict]) -> list[dict]:
    paths = []
    for item in reasoning_context.get("abnormal_items", []):
        finding = {"panel": item.get("panel"), "test": item.get("test"), "test_label": item.get("test_label"), "status": item.get("status"), "value": item.get("value"), "raw_value": item.get("raw_value"), "unit": item.get("unit", ""), "reference_range": item.get("reference_range", "")}
        related_patterns = [{"pattern_id": p.get("pattern_id"), "pattern_name": p.get("pattern_name"), "panel": p.get("panel"), "conditions": p.get("conditions", []), "confidence": p.get("confidence"), "description": p.get("description")} for p in reasoning_context.get("detected_patterns", []) if p.get("panel") in [item.get("panel"), "CROSS_PANEL"]]
        related_evidence = [{"evidence_id": e.get("evidence_id"), "panel": e.get("panel"), "source": e.get("source"), "page": e.get("page"), "score": e.get("final_score", e.get("score"))} for e in evidence if item.get("test") in e.get("tests", [])]
        paths.append({"finding": finding, "patterns": related_patterns[:3], "evidence": related_evidence[:3]})
    return paths

def enrich_reasoning_paths(ctx: dict, evidence: list[dict]) -> list[dict]:
    enriched = []
    for path in build_reasoning_paths(ctx, evidence):
        finding, patterns = path.get("finding", {}), path.get("patterns", [])
        candidate_conditions = {cond for p in patterns for cond in p.get("conditions", [])}
        related_evidence = []
        for e in evidence:
            same_test, same_condition = finding.get("test") in set(e.get("tests", [])), bool(candidate_conditions & set(e.get("conditions", [])))
            if same_test or same_condition: related_evidence.append({"evidence_id": e.get("evidence_id"), "panel": e.get("panel"), "source": e.get("source"), "page": e.get("page"), "tests": e.get("tests", []), "conditions": e.get("conditions", []), "score": e.get("final_score", e.get("score")), "link_reason": "same_test" if same_test else "same_condition"})
        enriched.append({"case_id": ctx.get("case_id"), "finding": finding, "patterns": patterns[:3], "conditions": sorted(candidate_conditions), "evidence": related_evidence[:4], "path_text": build_path_text(ctx.get("case_id"), finding, patterns, related_evidence)})
    return enriched

def build_path_text(case_id: str, finding: dict, patterns: list[dict], evidence: list[dict]) -> str:
    finding_text = f"Case {case_id} → Finding {finding.get('panel')} {finding.get('test')} {finding.get('status')}"
    pattern_text = " → ".join([f"Pattern {p.get('pattern_name')}" for p in patterns[:2]]) if patterns else "No matched pattern"
    conditions = sorted({c for p in patterns for c in p.get("conditions", [])})
    condition_text = "Condition " + ", ".join(conditions[:3]) if conditions else "No condition"
    evidence_text = f"Evidence {evidence[0].get('evidence_id')} ({evidence[0].get('source')}, p.{evidence[0].get('page')})" if evidence else "No direct evidence"
    return f"{finding_text} → {pattern_text} → {condition_text} → {evidence_text}"

def format_reasoning_paths_for_prompt(reasoning_paths: list[dict]) -> str:
    if not reasoning_paths: 
        return "- Không có graph reasoning path rõ ràng."
    
    lines = []
    for idx, path in enumerate(reasoning_paths[:8], start=1):
        finding = path.get("finding", {})
        f_text = f"{finding.get('panel')} {finding.get('test')} {finding.get('status')} ({finding.get('value')} {finding.get('unit', '')})"
        
        p_list = []
        for p in path.get("patterns", [])[:2]:
            c_text = ", ".join(p.get("conditions", []))
            p_list.append(f"{p.get('pattern_name')} -> {c_text}")
        p_text = "; ".join(p_list) if p_list else "No matched pattern"
        
        e_list = []
        for e in path.get("evidence", [])[:2]:
            e_list.append(f"{e.get('evidence_id')} ({e.get('source')}, p.{e.get('page')})")
        e_text = "; ".join(e_list) if e_list else "No direct evidence linked"
        
        lines.append(f"{idx}. Finding: {f_text} -> Pattern/Condition: {p_text} -> Evidence: {e_text}")
        
    return "\n".join(lines)

# =========================================================
# PROMPT FORMAT / USER VISIBLE STRINGS
# =========================================================

def clean_quote(text: str, max_len: int = 350) -> str:
    text = " ".join(str(text or "").replace("\n", " ").split()).strip()
    return text[:max_len].rstrip() + "..." if len(text) > max_len else text

def clean_reference_quote(text: str, max_len: int = 360) -> str:
    text = " ".join(str(text or "").replace("\n", " ").split()).strip()
    return text[:max_len].rstrip() + " […]" if len(text) > max_len else text

def build_references_block(evidence: list[dict]) -> str:
    if not evidence: return "📚 References:\n- Không có book evidence phù hợp để trích dẫn."
    return "\n".join(["📚 References:"] + [f"[{i}] {e.get('source', 'Unknown source')}" + (f", page {e.get('page')}." if e.get('page') else ".") + f" “{clean_reference_quote(e.get('text', ''))}”" for i, e in enumerate(evidence[:MAX_FINAL_EVIDENCE], start=1)])

def build_source_intro(panels: list[str]) -> str:
    panels = set(panels or [])
    lines = ["📚 Nguồn tài liệu:"]
    if "CBC" in panels: lines.extend(["- Harrison's Principles of Internal Medicine: Giáo trình nội khoa kinh điển, được sử dụng rộng rãi trong đào tạo bác sĩ toàn cầu.", "- Clinical Hematology: Tài liệu chuyên sâu về huyết học lâm sàng, hỗ trợ diễn giải công thức máu và các rối loạn huyết học."])
    if "BIOCHEM" in panels: lines.extend(["- Henry’s Clinical Diagnosis and Management by Laboratory Methods: Tài liệu chuẩn về diễn giải xét nghiệm cận lâm sàng và y học xét nghiệm.", "- Tietz Fundamentals of Clinical Chemistry and Molecular Diagnostics: Tài liệu nền tảng về hóa sinh lâm sàng, xét nghiệm sinh hóa và marker bệnh lý."])
    lines.extend(["", "Lưu ý: Nội dung chỉ có mục đích hỗ trợ diễn giải xét nghiệm, không thay thế chẩn đoán hoặc chỉ định điều trị của bác sĩ."])
    return "\n".join(lines)

def build_user_visible_answer(answer: str, ctx: dict, evidence: list[dict]) -> str:
    return f"{mechanical_cleanup_answer(answer)}\n\n{build_references_block(evidence)}\n\n{build_source_intro(ctx.get('panels', []))}".strip()

def build_final_prompt(reasoning_context: dict, evidence: list[dict], reasoning_paths: list[dict] | None = None) -> str:
    abnormal_list = []
    for item in reasoning_context.get("abnormal_items", []):
        val = item.get('value') if item.get('value') is not None else item.get('raw_value', '')
        ref = f", reference = {item.get('reference_range')}" if item.get("reference_range") else ""
        abnormal_list.append(f"- {item.get('panel')} | {item.get('test')} ({item.get('test_label')}): {val} {item.get('unit', '')}, status = {item.get('status')}{ref}")
    abnormal_text = "\n".join(abnormal_list) if abnormal_list else "- Không phát hiện bất thường rõ ràng."

    pattern_list = []
    for p in reasoning_context.get("detected_patterns", [])[:6]:
        cond = f" Conditions: {', '.join(p.get('conditions'))}" if p.get("conditions") else ""
        pattern_list.append(f"- {p.get('pattern_name')} ({p.get('panel')}), confidence={p.get('confidence')}: {p.get('description')}{cond}")
    pattern_text = "\n".join(pattern_list) if pattern_list else "- Không phát hiện pattern phối hợp rõ ràng."

    evidence_list = []
    for i, e in enumerate(evidence[:MAX_FINAL_EVIDENCE], start=1):
        quote = clean_quote(e.get('text', ''))
        tests = ", ".join(e.get('tests', []))
        conds = ", ".join(e.get('conditions', []))
        score = e.get('final_score', e.get('score'))
        evidence_list.append(f"[{i}] Panel={e.get('panel', 'UNKNOWN')}; Tests={tests}; Conditions={conds}; Source={e.get('source', 'Unknown')}; Page={e.get('page', '')}; Score={score}; Quote=\"{quote}\"")
    evidence_text = "\n".join(evidence_list) if evidence_list else "- Không có evidence phù hợp."

    graph_text = format_reasoning_paths_for_prompt(reasoning_paths or [])

    return f"""
Bạn là chuyên gia diễn giải xét nghiệm cận lâm sàng.

Nhiệm vụ:
- Diễn giải các bất thường xét nghiệm CBC và/hoặc sinh hóa cho người dùng cuối bằng tiếng Việt.
- Dùng ABNORMAL FINDINGS để biết chính xác xét nghiệm nào bất thường.
- Dùng DETECTED PATTERNS và GRAPH REASONING PATHS như context hỗ trợ suy luận.
- Chỉ dùng EVIDENCE từ sách PDF để đặt citation [1], [2], [3].
- Không dùng static pattern/rule làm citation nếu static pattern/rule không nằm trong EVIDENCE.
- Không bịa xét nghiệm không có trong ABNORMAL FINDINGS.
- Không bịa nguồn, tên sách hoặc số trang.
- Không chẩn đoán chắc chắn; chỉ dùng các từ: "gợi ý", "phù hợp với", "có thể liên quan đến", "cần đối chiếu lâm sàng".
- Nếu evidence không đủ cho một nhận định, phải nói rõ là bằng chứng còn hạn chế.
- Không kê thuốc, không đưa liều điều trị.
- Không nhắc lại prompt.
- Không dùng dấu "...".
- Trả lời gọn, rõ, phù hợp để hiển thị trực tiếp cho user.

========================
ABNORMAL FINDINGS
{abnormal_text}

DETECTED PATTERNS
{pattern_text}

GRAPH REASONING PATHS
{graph_text}

EVIDENCE
{evidence_text}
========================

YÊU CẦU OUTPUT:

### 1. Tóm tắt bất thường
- Liệt kê ngắn các chỉ số bất thường chính.
- Ghi giá trị, đơn vị và khoảng tham chiếu nếu có.

### 2. Ý nghĩa lâm sàng
- Giải thích ý nghĩa của các bất thường theo cụm xét nghiệm.
- Chỉ giải thích các xét nghiệm có trong ABNORMAL FINDINGS.
- Ưu tiên kết nối các chỉ số liên quan với nhau thay vì diễn giải rời rạc.
- Mỗi nhận định y khoa quan trọng cần có citation dạng [1], [2], [3].

### 3. Pattern gợi ý
- Nêu tối đa 2 pattern quan trọng nhất nếu có.
- Chỉ nêu pattern thật sự phù hợp với các bất thường trong ABNORMAL FINDINGS.
- Không nêu pattern cần xét nghiệm không xuất hiện trong case.
- Dùng ngôn ngữ thận trọng: "gợi ý", "phù hợp với", "có thể liên quan đến".

### 4. Lưu ý an toàn
- Nếu có bất thường có thể nguy hiểm, nhắc người dùng nên đi khám sớm hoặc cấp cứu phù hợp.
- Nếu chưa thấy critical flag rõ ràng, nói rằng vẫn cần đối chiếu với triệu chứng, bệnh sử, thuốc đang dùng và bác sĩ.

### 5. Nên làm gì tiếp theo
- Gợi ý các bước đánh giá tiếp theo hợp lý, ví dụ: kiểm tra lại xét nghiệm, xét nghiệm bổ sung, trao đổi bác sĩ.
- Không kê thuốc.
- Không đưa phác đồ điều trị.

### 6. Hạn chế
- Nêu những thông tin còn thiếu làm hạn chế diễn giải.
- Ví dụ: thiếu triệu chứng, bệnh sử, thuốc đang dùng, xét nghiệm nước tiểu, eGFR, tuổi/giới, hoặc kết quả lặp lại.

Lưu ý citation:
- Chỉ dùng citation dạng [1], [2], [3], [4], [5], [6] tương ứng với thứ tự trong EVIDENCE.
- Không dùng citation ngoài danh sách EVIDENCE.
- Không trích nguyên văn quote dài trong phần trả lời chính.
""".strip()

# =========================================================
# CLEANUP LLM OUTPUT
# =========================================================

def has_bad_placeholders(text: str) -> bool:
    lower = str(text or "").lower()
    for pattern in [r"\(\.\.\.\)", r"\(source,\s*p\.[^)]+\)", r"\[citation needed\]"]:
        if re.search(pattern, lower, flags=re.IGNORECASE): return True
    return False

def mechanical_cleanup_answer(text: str) -> str:
    text = str(text or "").replace("(...)", "")
    text = re.sub(r"\(Source,\s*p\.[^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()

def extract_final_answer(raw_text: str) -> str:
    text, lower = str(raw_text or "").strip(), str(raw_text or "").strip().lower()
    for marker in ["\nassistant\n", "\nassistant:", "assistant\n", "assistant:"]:
        if (pos := lower.rfind(marker)) != -1:
            cleaned = text[pos + len(marker):].strip()
            if cleaned: return cleaned
    return text