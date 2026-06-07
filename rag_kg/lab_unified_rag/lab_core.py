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

# Neo4j GraphRAG retriever (thêm vào để hỗ trợ graph retrieval)
try:
    from neo4j_retriever import neo4j_retrieve, retrieve_reasoning_chain
    NEO4J_RETRIEVER_AVAILABLE = True
except ImportError:
    NEO4J_RETRIEVER_AVAILABLE = False
    print("[lab_core] WARNING: neo4j_retriever.py chưa có hoặc thiếu package neo4j.")

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

            if not line:
                continue

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
    if value is None:
        return default

    try:
        return float(value)
    except Exception:
        return default


def ensure_list(value: Any) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


# =========================================================
# TEXT CLEANING / PDF PROCESSING
# =========================================================

def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\x00", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def clean_text_keep_case(text: str) -> str:
    text = str(text or "")
    text = text.replace("\x00", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = str(text or "").strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def get_pdf_skip_pages(pdf_path: Path | str) -> int:
    pdf_path = Path(pdf_path)
    return int(PDF_SKIP_PAGES_BY_FILE.get(pdf_path.name, PDF_SKIP_FIRST_PAGES))


def read_pdf_pages(pdf_path: Path | str, skip_first_pages: int | None = None) -> list[dict]:
    """
    Returns:
    [
      {"page": 18, "text": "..."},
      ...
    ]

    page is 1-based actual PDF page number.
    """

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

        if page_no <= skip_first_pages:
            continue

        try:
            page = doc.load_page(page_index)
            text = page.get_text("text")
            text = clean_text(text)

            if text:
                pages.append({
                    "page": page_no,
                    "text": text,
                })

        except Exception as exc:
            print(f"Warning: cannot read page {page_no} from {pdf_path.name}: {exc}")

    doc.close()
    return pages


# =========================================================
# KB BUILD: DETECTION
# =========================================================

def get_panel_test_map(panel: str) -> dict[str, list[str]]:
    panel = str(panel).upper()

    if panel == "CBC":
        return CBC_TEST_MAP

    if panel == "BIOCHEM":
        return BIOCHEM_TEST_MAP

    raise ValueError(f"Unknown panel: {panel}")


def get_panel_topic_keywords(panel: str) -> dict[str, list[str]]:
    panel = str(panel).upper()

    if panel == "CBC":
        return CBC_TOPIC_KEYWORDS

    if panel == "BIOCHEM":
        return BIOCHEM_TOPIC_KEYWORDS

    raise ValueError(f"Unknown panel: {panel}")


def detect_tests_in_text(text: str, panel: str) -> list[str]:
    text_l = str(text or "").lower()
    found: list[str] = []

    for test, keywords in get_panel_test_map(panel).items():
        for kw in keywords:
            kw_l = str(kw).lower().strip()

            if not kw_l:
                continue

            # For very short tokens such as k, na, cl, avoid excessive false positives.
            if len(kw_l) <= 2:
                pattern = rf"(?<![a-z0-9]){re.escape(kw_l)}(?![a-z0-9])"
            else:
                pattern = rf"\b{re.escape(kw_l)}\b"

            if re.search(pattern, text_l):
                found.append(test)
                break

    return sorted(set(found))


def detect_topics_in_text(text: str, panel: str) -> list[str]:
    text_l = str(text or "").lower()
    topics: list[str] = []

    for topic, keywords in get_panel_topic_keywords(panel).items():
        for kw in keywords:
            kw_l = str(kw).lower().strip()

            if kw_l and kw_l in text_l:
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
        candidates = [
            "anemia", "iron", "deficiency", "infection", "bacterial", "viral",
            "inflammation", "bleeding", "bone marrow", "thalassemia",
            "polycythemia", "neutrophilia", "neutropenia", "lymphocytosis",
            "thrombocytopenia", "macrocytic", "microcytic", "hypochromic",
            "megaloblastic", "hemolytic", "sepsis", "coagulation", "left shift",
            "pancytopenia", "leukocytosis", "leukopenia", "eosinophilia",
            "basophilia", "erythrocytosis",
        ]
    else:
        candidates = [
            "hepatitis", "hepatocellular", "transaminase", "aminotransferase",
            "azotemia", "uremia", "acute kidney injury", "chronic kidney disease",
            "renal function", "electrolyte", "hyponatremia", "hypernatremia",
            "hypokalemia", "hyperkalemia", "diabetes", "hyperglycemia",
            "hypoglycemia", "glycemic control", "hba1c", "dyslipidemia",
            "atherosclerosis", "cardiovascular risk", "lipoprotein",
            "myocardial infarction", "acute coronary syndrome", "heart failure",
            "troponin", "natriuretic peptide", "ferritin", "iron deficiency",
            "albumin", "hypoalbuminemia", "malnutrition", "protein loss",
            "ionized calcium", "parathyroid hormone", "hypercalcemia",
            "hypocalcemia", "urate", "gout", "hyperuricemia",
        ]

    return sorted({kw for kw in candidates if kw in text_l})


def infer_conditions_from_text(text: str, panel: str, tests: list[str], topics: list[str], keywords: list[str]) -> list[str]:
    text_l = str(text or "").lower()
    topics_l = [str(x).lower() for x in topics]
    keywords_l = [str(x).lower() for x in keywords]

    conditions: set[str] = set()

    topic_map = {
        # CBC
        "anemia": ["anemia"],
        "infection": ["infection"],
        "inflammation": ["inflammation"],
        "bleeding": ["bleeding_risk"],
        "bone_marrow": ["bone_marrow_disorder"],
        "thalassemia": ["thalassemia"],
        "polycythemia": ["polycythemia"],
        "coagulation": ["coagulation_disorder"],

        # BIOCHEM
        "liver_injury": ["liver_injury", "hepatocellular_injury"],
        "renal_function": ["renal_impairment", "reduced_kidney_function"],
        "electrolyte_disorder": ["electrolyte_disorder"],
        "diabetes": ["hyperglycemia", "diabetes_risk"],
        "dyslipidemia": ["dyslipidemia", "cardiovascular_risk"],
        "cardiac_biomarker": ["myocardial_injury", "cardiac_stress"],
        "iron_metabolism": ["iron_store_abnormality"],
        "protein_nutrition": ["hypoalbuminemia"],
        "bone_mineral": ["bone_mineral_disorder"],
        "purine_metabolism": ["hyperuricemia", "gout_risk"],
    }

    for topic in topics_l:
        for cond in topic_map.get(topic, []):
            conditions.add(cond)

    trigger_map = [
        # CBC
        (r"iron deficiency|iron deficient", "iron_deficiency_anemia"),
        (r"microcytic", "microcytic_anemia"),
        (r"macrocytic|megaloblastic|vitamin b12|folate", "macrocytic_anemia"),
        (r"thalassemia|thalassaemia", "thalassemia"),
        (r"hemolytic|hemolysis", "hemolytic_anemia"),
        (r"neutrophilia|bacterial infection", "bacterial_infection"),
        (r"lymphocytosis|viral infection", "viral_infection"),
        (r"left shift|immature granulocyte|band cell", "left_shift_stress_response"),
        (r"thrombocytopenia", "thrombocytopenia"),
        (r"thrombocytosis", "thrombocytosis"),
        (r"polycythemia|erythrocytosis", "polycythemia"),
        (r"leukocytosis", "leukocytosis"),
        (r"leukopenia", "leukopenia"),
        (r"neutropenia", "neutropenia"),
        (r"eosinophilia", "eosinophilia"),
        (r"basophilia", "basophilia"),

        # BIOCHEM
        (r"hepatocellular|aminotransferase|transaminase|hepatitis", "hepatocellular_injury"),
        (r"renal|kidney|creatinine|urea|azotemia|uremia", "renal_impairment"),
        (r"hyperglycemia|diabetes|glycemic", "hyperglycemia"),
        (r"dyslipidemia|lipoprotein|cardiovascular risk|atherosclerosis", "dyslipidemia"),
        (r"hyperkalemia|hypokalemia|hyponatremia|hypernatremia", "electrolyte_disorder"),
        (r"hypoalbuminemia|albumin", "hypoalbuminemia"),
        (r"ferritin|iron stores|iron storage", "iron_store_abnormality"),
        (r"urate|uric acid|gout|hyperuricemia", "hyperuricemia"),
        (r"troponin|ck-mb|myocardial|acute coronary syndrome", "myocardial_injury"),
        (r"natriuretic peptide|heart failure|pro-bnp|bnp", "cardiac_stress"),
        (r"parathyroid|pth|calcium", "bone_mineral_disorder"),
    ]

    combined = " ".join([text_l] + topics_l + keywords_l)

    for pattern, condition in trigger_map:
        if re.search(pattern, combined):
            conditions.add(condition)

    # Test-based weak hints, useful for KG and rerank.
    if panel == "CBC":
        test_condition_hints = {
            "HGB": ["anemia", "polycythemia"],
            "RBC": ["anemia", "polycythemia"],
            "HCT": ["anemia", "polycythemia"],
            "MCV": ["microcytic_anemia", "macrocytic_anemia"],
            "MCH": ["microcytic_anemia"],
            "MCHC": ["red_cell_index_abnormality"],
            "RDW": ["anisocytosis", "iron_deficiency_anemia"],
            "WBC": ["infection", "leukocytosis", "leukopenia"],
            "NEUT": ["bacterial_infection", "neutrophilia", "neutropenia"],
            "LYMPH": ["viral_infection", "lymphocytosis", "lymphopenia"],
            "PLT": ["thrombocytopenia", "thrombocytosis"],
            "IG": ["left_shift_stress_response"],
        }
    else:
        test_condition_hints = {
            "AST": ["liver_injury"],
            "ALT": ["liver_injury"],
            "UREA": ["renal_impairment"],
            "CREATININE": ["renal_impairment"],
            "NA": ["electrolyte_disorder"],
            "K": ["electrolyte_disorder"],
            "CL": ["electrolyte_disorder"],
            "GLUCOSE": ["hyperglycemia", "diabetes_risk"],
            "HBA1C": ["diabetes_risk", "poor_glycemic_control"],
            "CHOLESTEROL": ["dyslipidemia", "cardiovascular_risk"],
            "TRIGLYCERIDE": ["dyslipidemia", "cardiovascular_risk"],
            "HDL_C": ["dyslipidemia", "cardiovascular_risk"],
            "LDL_C": ["dyslipidemia", "cardiovascular_risk"],
            "CK_MB": ["myocardial_injury"],
            "TROPONIN_T": ["myocardial_injury"],
            "PRO_BNP": ["cardiac_stress"],
            "FERRITIN": ["iron_store_abnormality"],
            "ALBUMIN": ["hypoalbuminemia"],
            "CALCIUM_ION": ["bone_mineral_disorder"],
            "PTH": ["bone_mineral_disorder"],
            "URIC_ACID": ["hyperuricemia", "gout_risk"],
        }

    for test in tests:
        for cond in test_condition_hints.get(test, []):
            if cond.replace("_", " ") in combined or cond in combined:
                conditions.add(cond)

    return sorted(conditions)


def generate_query_hints(panel: str, tests: list[str], topics: list[str], conditions: list[str]) -> list[str]:
    queries: list[str] = []

    for test in tests:
        queries.append(f"{test.lower()} abnormal interpretation")
        queries.append(f"{test.lower()} clinical significance")
        queries.append(f"{test.lower()} causes of increase or decrease")

        if panel == "CBC":
            queries.append(f"{test.lower()} complete blood count finding")
        else:
            queries.append(f"{test.lower()} clinical chemistry interpretation")

    for topic in topics:
        if panel == "CBC":
            queries.append(f"{topic} cbc finding")
        else:
            queries.append(f"{topic} biochemistry interpretation")

    for condition in conditions:
        queries.append(f"{condition} laboratory interpretation")

    seen = set()
    output = []

    for q in queries:
        q = str(q).strip()

        if q and q not in seen:
            seen.add(q)
            output.append(q)

    return output


def is_noise_text(text: str) -> bool:
    text_l = str(text or "").lower().strip()
    words = text_l.split()

    if len(words) < MIN_WORDS:
        return True

    hard_noise = [
        "table of contents",
        "contents",
        "copyright",
        "isbn",
        "bibliography",
        "references",
        "index",
        "chapter contents",
        "permission",
        "all rights reserved",
        "printed in",
        "editorial",
    ]

    if any(x in text_l for x in hard_noise):
        # Avoid removing clinically valuable sentences that merely cite references.
        if len(words) < 80:
            return True

    # Too many numeric-only tokens often means tables, references, or index lines.
    numeric_tokens = 0

    for w in words:
        cleaned = w.replace(".", "").replace(",", "").replace("-", "").replace("–", "")
        if cleaned.isdigit():
            numeric_tokens += 1

    digit_ratio = numeric_tokens / max(len(words), 1)

    if digit_ratio > 0.45:
        return True

    # Reference-heavy paragraph.
    if text_l.count(" et al") >= 3 or text_l.count(" doi") >= 1:
        return True

    return False


def score_chunk(text: str, ctype: str, tests: list[str], topics: list[str], conditions: list[str]) -> float:
    text_l = str(text or "").lower()
    words = len(text_l.split())
    points = 0.0

    if ctype == "interpretation":
        points += 3.0
    elif ctype == "cause":
        points += 2.0
    elif ctype == "definition":
        points += 1.0

    if len(tests) >= 3:
        points += 2.5
    elif len(tests) == 2:
        points += 1.8
    elif len(tests) == 1:
        points += 1.0

    if topics:
        points += min(1.5, len(topics) * 0.5)

    if conditions:
        points += min(2.0, len(conditions) * 0.6)

    clinical_markers = [
        "indicates", "suggests", "consistent with", "associated with",
        "reflects", "is seen in", "caused by", "due to", "leads to",
        "result of", "elevation of", "decrease in", "low levels", "high levels",
    ]

    if any(marker in text_l for marker in clinical_markers):
        points += 1.0

    if 25 <= words <= MAX_WORDS:
        points += 1.0
    elif words > MAX_WORDS:
        points += 0.4

    # Penalize weak text.
    if not tests and not topics:
        points -= 1.0

    if is_noise_text(text_l):
        points -= 2.0

    score = points / 10.0
    score = max(0.0, min(1.0, score))

    return round(score, 2)


def semantic_chunk_page_text(text: str) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if is_noise_text(sentence):
            continue

        current_words = len(current.split())
        sentence_words = len(sentence.split())

        should_start_new = (
            _BOUNDARY_REGEX.search(sentence) is not None
            and current_words >= 25
        )

        if should_start_new:
            if current.strip():
                chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip()

        if len(current.split()) >= MAX_WORDS:
            chunks.append(current.strip())
            current = ""

        # If single sentence is very long, cut it softly.
        if sentence_words >= MAX_WORDS and current:
            chunks.append(current.strip())
            current = ""

    if len(current.split()) >= MIN_WORDS:
        chunks.append(current.strip())

    cleaned_chunks = []

    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()

        if not chunk:
            continue

        if is_noise_text(chunk):
            continue

        if len(chunk.split()) < MIN_WORDS:
            continue

        cleaned_chunks.append(chunk)

    return cleaned_chunks


def build_kb_from_pdf_paths(pdf_paths: list[Path], panel: str) -> list[dict]:
    panel = str(panel).upper()
    kb: list[dict] = []

    print("=" * 80)
    print(f"BUILDING {panel} KB FROM PDF")
    print("=" * 80)

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        source = pdf_path.name
        trust = SOURCE_TRUST.get(source, 0.8)
        skip_pages = get_pdf_skip_pages(pdf_path)

        print(f"\nPDF: {pdf_path}")
        print(f"Skip first pages: {skip_pages}")

        if not pdf_path.exists():
            print(f"Warning: missing PDF, skip: {pdf_path}")
            continue

        pages = read_pdf_pages(pdf_path, skip_first_pages=skip_pages)
        print(f"Pages read: {len(pages)}")

        source_count = 0

        for page_obj in pages:
            page_no = page_obj["page"]
            page_text = page_obj["text"]
            chunks = semantic_chunk_page_text(page_text)

            for chunk in chunks:
                tests = detect_tests_in_text(chunk, panel)
                topics = detect_topics_in_text(chunk, panel)
                keywords = extract_keywords(chunk, panel)
                ctype = classify_chunk_type(chunk)
                conditions = infer_conditions_from_text(chunk, panel, tests, topics, keywords)
                score = score_chunk(chunk, ctype, tests, topics, conditions)

                if score < KB_QUALITY_THRESHOLD:
                    continue

                evidence_id = f"ev_{panel.lower()}_{len(kb):06d}_{stable_hash(source + str(page_no) + chunk)}"

                item = {
                    "evidence_id": evidence_id,
                    "panel": panel,
                    "text": chunk,
                    "bm25_text": chunk,
                    "tests": tests,
                    "topics": topics,
                    "keywords": keywords,
                    "conditions": conditions,
                    "query_hints": generate_query_hints(panel, tests, topics, conditions),
                    "type": ctype,
                    "score": score,
                    "trust": trust,
                    "source": source,
                    "page": page_no,
                    "source_type": "pdf_book",
                }

                item["embedding_text"] = build_embedding_text(item)

                kb.append(item)
                source_count += 1

                if MAX_CHUNKS_PER_PDF is not None and source_count >= MAX_CHUNKS_PER_PDF:
                    break

            if MAX_CHUNKS_PER_PDF is not None and source_count >= MAX_CHUNKS_PER_PDF:
                break

        print(f"Chunks kept from {source}: {source_count}")

    print(f"\nTotal {panel} KB chunks: {len(kb)}")
    return kb


# =========================================================
# NORMALIZATION FOR CASE INPUT
# =========================================================

def normalize_raw_test_name(test_name: str) -> str:
    raw = str(test_name or "").strip()
    raw = raw.replace("-", "_")
    raw = raw.replace(" ", "_")
    raw = raw.upper()
    return raw


def guess_panel(test_name: str) -> str:
    raw = normalize_raw_test_name(test_name)

    for panel, mapping in TEST_NORMALIZATION.items():
        if raw in mapping:
            return panel

    cbc_tests = {
        "RBC", "HGB", "HB", "HCT", "MCV", "MCH", "MCHC",
        "RDW", "RDW_SD", "RDW_CV", "WBC", "NEUT", "LYM",
        "LYMPH", "MONO", "EOS", "BASO", "PLT", "IG",
        "NEUT_PERCENT", "NEUT_ABS", "LYM_PERCENT", "LYM_ABS",
        "MONO_PERCENT", "MONO_ABS", "EOS_PERCENT", "EOS_ABS",
        "BASO_PERCENT", "BASO_ABS", "IG_PERCENT", "IG_ABS",
        "PCT", "MPV", "PDW",
    }

    if raw in cbc_tests:
        return "CBC"

    return "BIOCHEM"


def normalize_test_name(test_name: str, panel: str | None = None) -> str:
    raw = normalize_raw_test_name(test_name)

    if panel:
        panel = str(panel).upper()
        return TEST_NORMALIZATION.get(panel, {}).get(raw, raw)

    guessed_panel = guess_panel(raw)
    return TEST_NORMALIZATION.get(guessed_panel, {}).get(raw, raw)


def normalize_status(status: Any) -> str:
    raw = str(status or "").strip().lower()

    if not raw:
        return ""

    return STATUS_NORMALIZATION.get(raw, raw)


def normalize_panel(panel: Any, test_name: str = "") -> str:
    if panel:
        p = str(panel).strip().upper()

        if p in {"CBC", "HEMATOLOGY", "HAEMATOLOGY"}:
            return "CBC"

        if p in {"BIOCHEM", "BIOCHEMISTRY", "CHEMISTRY", "CLINICAL_CHEMISTRY"}:
            return "BIOCHEM"

    return guess_panel(test_name)


def get_case_id(case: dict, idx: int = 0) -> str:
    """
    Case identity rule:
    - case_id has highest priority.
    - otherwise id/image_id/file_name.
    - CBC id == BIOCHEM id means same patient/case.
    - Different ids are treated as different cases.
    """

    for key in ["case_id", "id", "image_id", "img_id", "file_name", "filename"]:
        if case.get(key):
            return str(case[key])

    return f"case_{idx:04d}"


def format_ref_range(ref_range: Any) -> str:
    if not ref_range:
        return ""

    if isinstance(ref_range, dict):
        ref_min = ref_range.get("ref_min")
        ref_max = ref_range.get("ref_max")

        if ref_min is not None and ref_max is not None:
            return f"{ref_min} - {ref_max}"

        return json.dumps(ref_range, ensure_ascii=False)

    return str(ref_range)


def extract_case_items(case: dict | list, default_panel: str | None = None) -> list[dict]:
    if isinstance(case, list):
        raw_items = case
    else:
        raw_items = (
            case.get("data")
            or case.get("results")
            or case.get("items")
            or case.get("lab_results")
            or []
        )

    output: list[dict] = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        raw_test = (
            item.get("test_name")
            or item.get("name")
            or item.get("test")
            or item.get("parameter")
            or item.get("analyte")
            or ""
        )

        if not raw_test:
            continue

        panel = normalize_panel(item.get("panel") or default_panel, raw_test)
        test = normalize_test_name(raw_test, panel)
        status = normalize_status(item.get("status", ""))

        ref_range = (
            item.get("reference_range")
            or item.get("ref_range")
            or item.get("ref")
            or item.get("range")
            or ""
        )

        output.append({
            "panel": panel,
            "raw_test": raw_test,
            "test": test,
            "test_label": TEST_LABELS.get(test, test),
            "value": item.get("value"),
            "raw_value": item.get("raw_value"),
            "unit": item.get("unit", ""),
            "status": status,
            "reference_range": format_ref_range(ref_range),
            "raw_text_line": item.get("raw_text_line", ""),
            "source_case_item": item,
        })

    return output


def normalize_single_case(case: dict, idx: int = 0, default_panel: str | None = None) -> dict:
    case_id = get_case_id(case, idx)
    items = extract_case_items(case, default_panel=default_panel)

    return {
        "case_id": case_id,
        "data": items,
        "raw_case": case,
    }


def merge_case_lists(cbc_cases: list[dict], biochem_cases: list[dict]) -> list[dict]:
    """
    Merge by case_id/id only.
    Same id = same patient.
    Different id = different case.
    """

    merged: dict[str, dict] = {}

    for idx, case in enumerate(cbc_cases):
        norm = normalize_single_case(case, idx, default_panel="CBC")
        case_id = norm["case_id"]

        merged.setdefault(case_id, {
            "case_id": case_id,
            "data": [],
            "raw_sources": {},
        })

        merged[case_id]["data"].extend(norm["data"])
        merged[case_id]["raw_sources"]["CBC"] = case

    for idx, case in enumerate(biochem_cases):
        norm = normalize_single_case(case, idx, default_panel="BIOCHEM")
        case_id = norm["case_id"]

        merged.setdefault(case_id, {
            "case_id": case_id,
            "data": [],
            "raw_sources": {},
        })

        merged[case_id]["data"].extend(norm["data"])
        merged[case_id]["raw_sources"]["BIOCHEM"] = case

    output = list(merged.values())
    output.sort(key=lambda x: str(x.get("case_id", "")))
    return output


def extract_lab_items(case: dict) -> list[dict]:
    raw_items = case.get("data", [])
    output: list[dict] = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        raw_test = item.get("raw_test") or item.get("test_name") or item.get("test") or ""
        test = item.get("test") or normalize_test_name(raw_test, item.get("panel"))
        panel = normalize_panel(item.get("panel"), raw_test or test)
        status = normalize_status(item.get("status", ""))

        ref_range = (
            item.get("reference_range")
            or item.get("ref_range")
            or item.get("ref")
            or ""
        )

        output.append({
            "panel": panel,
            "raw_test": raw_test or test,
            "test": test,
            "test_label": TEST_LABELS.get(test, test),
            "value": item.get("value"),
            "raw_value": item.get("raw_value"),
            "unit": item.get("unit", ""),
            "status": status,
            "reference_range": format_ref_range(ref_range),
            "raw_text_line": item.get("raw_text_line", ""),
            "source_case_item": item.get("source_case_item", item),
        })

    return output


def extract_abnormal_items(items: list[dict]) -> list[dict]:
    abnormal: list[dict] = []

    for item in items:
        status = normalize_status(item.get("status"))

        if status and status != "normal":
            x = dict(item)
            x["status"] = status
            abnormal.append(x)

    return abnormal


# =========================================================
# PATTERN DETECTION
# =========================================================

def has_finding(abnormal_items: list[dict], panel: str, test: str, status: str) -> bool:
    panel = str(panel).upper()
    test = normalize_test_name(test, panel)
    status = normalize_status(status)

    for item in abnormal_items:
        if (
            str(item.get("panel", "")).upper() == panel
            and item.get("test") == test
            and normalize_status(item.get("status")) == status
        ):
            return True

    return False


def detect_panel_patterns(abnormal_items: list[dict]) -> list[dict]:
    detected: list[dict] = []

    for panel, rules in PANEL_PATTERNS.items():
        panel_items = [x for x in abnormal_items if x.get("panel") == panel]

        if not panel_items:
            continue

        for rule in rules:
            requires = rule.get("requires", [])
            optional = rule.get("optional", [])

            required_hits = 0
            matched_required = []

            for test, status in requires:
                if has_finding(abnormal_items, panel, test, status):
                    required_hits += 1
                    matched_required.append(f"{test}_{status}")

            if requires and required_hits < len(requires):
                continue

            optional_hits = 0
            matched_optional = []

            for test, status in optional:
                if has_finding(abnormal_items, panel, test, status):
                    optional_hits += 1
                    matched_optional.append(f"{test}_{status}")

            denominator = len(requires) + max(len(optional), 1)
            confidence = (required_hits + 0.5 * optional_hits) / denominator
            confidence = min(0.95, max(0.6, confidence))

            detected.append({
                "pattern_id": rule["pattern_id"],
                "pattern_name": rule["name"],
                "panel": panel,
                "conditions": rule.get("conditions", []),
                "description": rule.get("description", ""),
                "confidence": round(confidence, 2),
                "matched_required": matched_required,
                "matched_optional": matched_optional,
                "source": "panel_rule",
            })

    detected.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return detected


def detect_cross_panel_patterns(abnormal_items: list[dict]) -> list[dict]:
    panels = {item.get("panel") for item in abnormal_items}

    # Do not apply cross-panel rules unless the same merged case contains both panels.
    if not {"CBC", "BIOCHEM"}.issubset(panels):
        return []

    detected: list[dict] = []

    for rule in CROSS_PANEL_PATTERNS:
        requires = rule.get("requires", [])
        optional = rule.get("optional", [])

        required_hits = 0
        matched_required = []

        for panel, test, status in requires:
            if has_finding(abnormal_items, panel, test, status):
                required_hits += 1
                matched_required.append(f"{panel}_{test}_{status}")

        if requires and required_hits < len(requires):
            continue

        optional_hits = 0
        matched_optional = []

        for panel, test, status in optional:
            if has_finding(abnormal_items, panel, test, status):
                optional_hits += 1
                matched_optional.append(f"{panel}_{test}_{status}")

        denominator = len(requires) + max(len(optional), 1)
        confidence = (required_hits + 0.5 * optional_hits) / denominator
        confidence = min(0.95, max(0.65, confidence))

        detected.append({
            "pattern_id": rule["pattern_id"],
            "pattern_name": rule["name"],
            "panel": "CROSS_PANEL",
            "conditions": rule.get("conditions", []),
            "description": rule.get("description", ""),
            "confidence": round(confidence, 2),
            "matched_required": matched_required,
            "matched_optional": matched_optional,
            "source": "cross_panel_rule",
        })

    detected.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return detected


def build_reasoning_context(case: dict, idx: int = 0) -> dict:
    case_id = get_case_id(case, idx)

    items = extract_lab_items(case)
    abnormal_items = extract_abnormal_items(items)

    panel_patterns = detect_panel_patterns(abnormal_items)
    cross_patterns = detect_cross_panel_patterns(abnormal_items)

    detected_patterns = sorted(
        panel_patterns + cross_patterns,
        key=lambda x: x.get("confidence", 0),
        reverse=True,
    )

    panels = sorted({x["panel"] for x in items if x.get("panel")})
    abnormal_tests = sorted({x["test"] for x in abnormal_items if x.get("test")})

    conditions = sorted({
        condition
        for pattern in detected_patterns
        for condition in pattern.get("conditions", [])
    })

    return {
        "case_id": case_id,
        "panels": panels,
        "items": items,
        "abnormal_items": abnormal_items,
        "abnormal_tests": abnormal_tests,
        "detected_patterns": detected_patterns,
        "conditions": conditions,
    }


# =========================================================
# KB NORMALIZATION / EMBEDDING TEXT
# =========================================================

def build_embedding_text(item: dict) -> str:
    parts: list[str] = []

    parts.append("Panel: " + str(item.get("panel", "")))
    parts.append("Clinical text: " + str(item.get("text", "")))

    if item.get("tests"):
        parts.append("Tests: " + ", ".join(item["tests"]))

    if item.get("topics"):
        parts.append("Topics: " + ", ".join(item["topics"]))

    if item.get("conditions"):
        parts.append("Conditions: " + ", ".join(item["conditions"]))

    if item.get("keywords"):
        parts.append("Keywords: " + ", ".join(item["keywords"]))

    if item.get("type"):
        parts.append("Type: " + str(item["type"]))

    if item.get("tests"):
        parts.append("Clinical interpretation of " + ", ".join(item["tests"]))

    return " | ".join(parts)


def normalize_kb_item(item: dict, panel: str, idx: int = 0) -> dict:
    x = dict(item)
    panel = str(panel).upper()
    source = str(x.get("source", "Unknown"))
    text = str(x.get("text", ""))

    x["panel"] = panel
    x["evidence_id"] = x.get("evidence_id") or f"ev_{panel.lower()}_{idx:06d}_{stable_hash(text)}"
    x["tests"] = sorted({normalize_test_name(t, panel) for t in x.get("tests", []) if t})
    x["topics"] = x.get("topics", [])
    x["keywords"] = x.get("keywords", [])

    if not x.get("conditions"):
        x["conditions"] = infer_conditions_from_text(
            text=text,
            panel=panel,
            tests=x["tests"],
            topics=x["topics"],
            keywords=x["keywords"],
        )

    x["type"] = x.get("type", "general")
    x["score"] = safe_float(x.get("score"), 0.3)
    x["trust"] = safe_float(x.get("trust"), SOURCE_TRUST.get(source, 0.8))
    x["source_type"] = x.get("source_type", "pdf_book")
    x["embedding_text"] = x.get("embedding_text") or build_embedding_text(x)

    return x


# =========================================================
# QUERY BUILDING
# =========================================================

def build_query_hints(reasoning_context: dict) -> list[str]:
    queries: list[str] = []

    for item in reasoning_context.get("abnormal_items", []):
        panel = item.get("panel")
        test = item.get("test")
        status = item.get("status")

        if not test or not status:
            continue

        queries.append(f"{panel} {test} {status} interpretation")
        queries.append(f"{test} {status} clinical significance")
        queries.append(f"{test} {status} causes")

        if panel == "CBC":
            queries.append(f"{test} {status} complete blood count interpretation")

        if panel == "BIOCHEM":
            queries.append(f"{test} {status} clinical chemistry interpretation")

    for pattern in reasoning_context.get("detected_patterns", []):
        if pattern.get("pattern_name"):
            queries.append(pattern["pattern_name"])

        if pattern.get("description"):
            queries.append(pattern["description"])

        for condition in pattern.get("conditions", []):
            queries.append(f"{condition} laboratory interpretation")

    abnormal_tests = reasoning_context.get("abnormal_tests", [])
    if abnormal_tests:
        queries.append(" ".join(abnormal_tests) + " laboratory abnormal pattern")

    queries.append("complete blood count interpretation")
    queries.append("clinical chemistry interpretation")
    queries.append("laboratory test abnormal pattern interpretation")

    seen = set()
    output = []

    for q in queries:
        q = str(q).strip()

        if q and q not in seen:
            seen.add(q)
            output.append(q)

    return output


# =========================================================
# QDRANT RETRIEVAL
# =========================================================

def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _EMBEDDING_MODEL


def get_qdrant_client() -> QdrantClient:
    global _QDRANT_CLIENT

    if _QDRANT_CLIENT is None:
        _QDRANT_CLIENT = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    return _QDRANT_CLIENT


def build_embedding_text_for_query(query: str) -> str:
    return f"Clinical laboratory interpretation query: {query}"


def qdrant_search(query: str, top_k: int = TOP_K_PER_QUERY) -> list[dict]:
    model = get_embedding_model()
    client = get_qdrant_client()

    query_text = build_embedding_text_for_query(query)
    vector = model.encode([query_text])[0].tolist()

    if hasattr(client, "search"):
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
    else:
        query_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        results = query_result.points

    evidence: list[dict] = []

    for hit in results:
        payload = hit.payload or {}
        hit_score = float(getattr(hit, "score", 0) or 0)

        evidence.append({
            "evidence_id": payload.get("evidence_id") or f"ev_{stable_hash(payload.get('text', ''))}",
            "panel": payload.get("panel", "UNKNOWN"),
            "text": payload.get("text", ""),
            "bm25_text": payload.get("bm25_text", ""),
            "tests": payload.get("tests", []),
            "topics": payload.get("topics", []),
            "conditions": payload.get("conditions", []),
            "keywords": payload.get("keywords", []),
            "type": payload.get("type", "general"),
            "score": hit_score,
            "kb_score": payload.get("score", 0),
            "trust": float(payload.get("trust", 0.8) or 0.8),
            "source": payload.get("source", "Unknown source"),
            "page": payload.get("page", ""),
            "source_type": payload.get("source_type", "pdf_book"),
            "is_static": False,
        })

    return evidence


def dedup_evidence(evidence: list[dict]) -> list[dict]:
    seen = set()
    output = []

    for e in evidence:
        text = re.sub(r"\s+", " ", str(e.get("text", "")).lower()).strip()
        fp = text[:260]

        if not fp:
            continue

        if fp in seen:
            continue

        seen.add(fp)
        output.append(e)

    return output


def rerank_evidence(evidence: list[dict], reasoning_context: dict) -> list[dict]:
    abnormal_tests = set(reasoning_context.get("abnormal_tests", []))
    panels = set(reasoning_context.get("panels", []))
    conditions = set(reasoning_context.get("conditions", []))

    ranked: list[dict] = []

    for e in evidence:
        score = safe_float(e.get("score"), 0.0)

        e_tests = set(e.get("tests", []))
        e_conditions = set(e.get("conditions", []))
        e_panel = e.get("panel")

        score += safe_float(e.get("trust"), 0.8) * 0.10
        score += safe_float(e.get("kb_score"), 0.0) * 0.12

        if e_tests & abnormal_tests:
            score += 0.30

        if e_conditions & conditions:
            score += 0.30

        if e_panel in panels:
            score += 0.18

        if e.get("type") == "interpretation":
            score += 0.12
        elif e.get("type") == "cause":
            score += 0.08
        elif e.get("type") == "definition":
            score += 0.03

        # Penalty Neo4j chunk không nhắc đến test nào trong phiếu
        if e.get("retrieval_path", "").startswith("neo4j"):
            test_keywords = {t.lower() for t in abnormal_tests}
            text_lower = str(e.get("text", "")).lower()
            keyword_hits = sum(1 for kw in test_keywords if kw in text_lower)
            if keyword_hits == 0:
                score -= 0.25

        x = dict(e)
        x["final_score"] = round(score, 4)
        ranked.append(x)

    ranked.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return ranked


def retrieve_evidence(reasoning_context: dict) -> list[dict]:
    """
    GraphRAG retrieval: kết hợp Qdrant (semantic) + Neo4j (graph relation).
    - Qdrant: tìm evidence gần nghĩa với query
    - Neo4j Path A: Evidence theo Test bất thường (MENTIONS_TEST)
    - Neo4j Path B: Evidence theo Condition từ Pattern (SUPPORTS)
    """
    all_evidence: list[dict] = []

    # --- Qdrant semantic retrieval ---
    queries = build_query_hints(reasoning_context)
    print(f"[retrieve] Qdrant queries: {queries[:3]}")

    for query in queries[:10]:
        try:
            hits = qdrant_search(query, top_k=TOP_K_PER_QUERY)
            all_evidence.extend(hits)
        except Exception as exc:
            print(f"[retrieve] Qdrant failed for query='{query}': {exc}")

    qdrant_count = len(all_evidence)
    print(f"[retrieve] Qdrant → {qdrant_count} evidence (before dedup)")

    # --- Neo4j graph retrieval ---
    if NEO4J_RETRIEVER_AVAILABLE:
        try:
            graph_evidence = neo4j_retrieve(reasoning_context)
            all_evidence.extend(graph_evidence)
            print(f"[retrieve] Neo4j → {len(graph_evidence)} evidence")
        except Exception as exc:
            print(f"[retrieve] Neo4j retrieval failed: {exc}")
    else:
        print("[retrieve] Neo4j không khả dụng, chỉ dùng Qdrant")

    # --- Dedup + rerank ---
    all_evidence = dedup_evidence(all_evidence)
    all_evidence = rerank_evidence(all_evidence, reasoning_context)

    print(f"[retrieve] Sau dedup+rerank: {len(all_evidence)} → trả về top {MAX_RAW_EVIDENCE}")
    return all_evidence[:MAX_RAW_EVIDENCE]


# =========================================================
# REASONING PATHS / GRAPH EXPLAINABILITY
# =========================================================

def finding_node_id(case_id: str, panel: str, test: str, status: str) -> str:
    return f"finding_{slugify(case_id)}_{panel}_{test}_{status}"


def build_reasoning_paths(reasoning_context: dict, evidence: list[dict]) -> list[dict]:
    """
    Xây dựng reasoning paths để đưa vào prompt LLM.
    Ưu tiên dùng Neo4j chain (graph thật), fallback về memory nếu không có.
    """
    # --- Thử Neo4j chain trước ---
    if NEO4J_RETRIEVER_AVAILABLE:
        case_id = reasoning_context.get("case_id", "")
        try:
            chain_rows = retrieve_reasoning_chain(case_id, limit=12)
            if chain_rows:
                # Group theo finding
                from collections import defaultdict
                grouped: dict = defaultdict(lambda: {"patterns": [], "evidence": []})
                for row in chain_rows:
                    key = f"{row.get('test_code')}_{row.get('direction')}"
                    node = grouped[key]
                    node["finding"] = {
                        "panel":           "CBC",
                        "test":            row.get("test_code", ""),
                        "test_label":      row.get("test_code", ""),
                        "status":          row.get("direction", ""),
                        "value":           row.get("value", ""),
                        "unit":            row.get("unit", ""),
                        "reference_range": "",
                    }
                    pat = {
                        "pattern_name": row.get("pattern_name", ""),
                        "conditions":   [row.get("condition_name", "")],
                        "confidence":   row.get("pattern_score", 0),
                    }
                    if pat not in node["patterns"]:
                        node["patterns"].append(pat)
                    ev = {
                        "evidence_id": row.get("ev_id", ""),
                        "source":      row.get("src_name", ""),
                        "page":        row.get("page", ""),
                        "score":       float(row.get("trust") or 0),
                    }
                    if ev not in node["evidence"]:
                        node["evidence"].append(ev)
                paths_neo4j = [
                    {
                        "finding":  v["finding"],
                        "patterns": v["patterns"][:3],
                        "evidence": v["evidence"][:3],
                    }
                    for v in grouped.values() if v.get("finding")
                ]
                if paths_neo4j:
                    return paths_neo4j
        except Exception as exc:
            print(f"[build_reasoning_paths] Neo4j chain failed: {exc}")

    # --- Fallback: build từ memory (logic cũ) ---
    paths: list[dict] = []

    abnormal_items = reasoning_context.get("abnormal_items", [])
    patterns = reasoning_context.get("detected_patterns", [])

    for item in abnormal_items:
        finding = {
            "panel": item.get("panel"),
            "test": item.get("test"),
            "test_label": item.get("test_label"),
            "status": item.get("status"),
            "value": item.get("value"),
            "raw_value": item.get("raw_value"),
            "unit": item.get("unit", ""),
            "reference_range": item.get("reference_range", ""),
        }

        related_patterns = []
        for pattern in patterns:
            pattern_panel = pattern.get("panel")

            if pattern_panel in [item.get("panel"), "CROSS_PANEL"]:
                related_patterns.append({
                    "pattern_id": pattern.get("pattern_id"),
                    "pattern_name": pattern.get("pattern_name"),
                    "panel": pattern.get("panel"),
                    "conditions": pattern.get("conditions", []),
                    "confidence": pattern.get("confidence"),
                    "description": pattern.get("description"),
                })

        related_evidence = []
        for e in evidence:
            if item.get("test") in e.get("tests", []):
                related_evidence.append({
                    "evidence_id": e.get("evidence_id"),
                    "panel": e.get("panel"),
                    "source": e.get("source"),
                    "page": e.get("page"),
                    "score": e.get("final_score", e.get("score")),
                })

        paths.append({
            "finding": finding,
            "patterns": related_patterns[:3],
            "evidence": related_evidence[:3],
        })

    return paths


def format_reasoning_paths_for_prompt(reasoning_paths: list[dict]) -> str:
    if not reasoning_paths:
        return "- Không có graph reasoning path rõ ràng."

    lines = []

    for idx, path in enumerate(reasoning_paths[:8], start=1):
        finding = path.get("finding", {})
        patterns = path.get("patterns", [])
        evidence = path.get("evidence", [])

        finding_text = (
            f"{finding.get('panel')} {finding.get('test')} "
            f"{finding.get('status')} ({finding.get('value')} {finding.get('unit', '')})"
        )

        if patterns:
            pattern_text = "; ".join([
                f"{p.get('pattern_name')} → {', '.join(p.get('conditions', []))}"
                for p in patterns[:2]
            ])
        else:
            pattern_text = "No matched pattern"

        if evidence:
            ev_text = "; ".join([
                f"{e.get('evidence_id')} ({e.get('source')}, p.{e.get('page')})"
                for e in evidence[:2]
            ])
        else:
            ev_text = "No direct evidence linked"

        lines.append(f"{idx}. Finding: {finding_text} → Pattern/Condition: {pattern_text} → Evidence: {ev_text}")

    return "\n".join(lines)


# =========================================================
# GRAPH BUILDER
# =========================================================

def build_graph_from_kb_and_cases(kb: list[dict], cases: list[dict]) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(node_id: str, node_type: str, **props):
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "node_type": node_type,
                **props,
            }

    def add_edge(source: str, relation: str, target: str, **props):
        edges.append({
            "source": source,
            "relation": relation,
            "target": target,
            **props,
        })

    for idx, item in enumerate(kb):
        panel = item.get("panel", "UNKNOWN")
        text = item.get("text", "")
        ev_id = item.get("evidence_id") or f"ev_{panel.lower()}_{idx:06d}_{stable_hash(text)}"

        source = item.get("source", "Unknown")
        source_id = f"source_{slugify(source)}"
        panel_id = f"panel_{panel}"

        add_node(panel_id, "Panel", name=panel)

        add_node(
            ev_id,
            "Evidence",
            panel=panel,
            text=text,
            source=source,
            page=item.get("page", ""),
            trust=item.get("trust", 0.8),
            score=item.get("score", 0),
            evidence_type=item.get("type", "general"),
        )

        add_node(source_id, "Source", name=source)

        add_edge(ev_id, "FROM_SOURCE", source_id, confidence=item.get("trust", 0.8))
        add_edge(ev_id, "BELONGS_TO_PANEL", panel_id)

        for test in item.get("tests", []):
            test_id = f"test_{test}"
            add_node(test_id, "Test", name=test, display_name=TEST_LABELS.get(test, test))
            add_edge(ev_id, "MENTIONS_TEST", test_id)

        for condition in item.get("conditions", []):
            condition_id = f"condition_{slugify(condition)}"
            add_node(condition_id, "Condition", name=condition)
            add_edge(ev_id, "SUPPORTS", condition_id)

    for idx, case in enumerate(cases):
        ctx = build_reasoning_context(case, idx)
        case_id = ctx["case_id"]
        case_node_id = f"case_{slugify(case_id)}"

        add_node(case_node_id, "Case", case_id=case_id)

        for item in ctx.get("abnormal_items", []):
            panel = item.get("panel")
            test = item.get("test")
            status = item.get("status")

            finding_id = finding_node_id(case_id, panel, test, status)
            test_id = f"test_{test}"
            status_id = f"status_{status}"
            panel_id = f"panel_{panel}"

            add_node(panel_id, "Panel", name=panel)
            add_node(test_id, "Test", name=test, display_name=TEST_LABELS.get(test, test))
            add_node(status_id, "Status", name=status)

            add_node(
                finding_id,
                "Finding",
                panel=panel,
                test=test,
                status=status,
                value=item.get("value"),
                raw_value=item.get("raw_value"),
                unit=item.get("unit", ""),
                reference_range=item.get("reference_range", ""),
            )

            add_edge(case_node_id, "HAS_FINDING", finding_id)
            add_edge(finding_id, "OF_TEST", test_id)
            add_edge(finding_id, "HAS_STATUS", status_id)
            add_edge(test_id, "BELONGS_TO_PANEL", panel_id)

        for pattern in ctx.get("detected_patterns", []):
            pattern_id = f"pattern_{slugify(pattern.get('pattern_id'))}"

            add_node(
                pattern_id,
                "Pattern",
                name=pattern.get("pattern_name"),
                panel=pattern.get("panel"),
                confidence=pattern.get("confidence"),
                description=pattern.get("description"),
            )

            add_edge(
                case_node_id,
                "MATCHES_PATTERN",
                pattern_id,
                confidence=pattern.get("confidence"),
            )

            for condition in pattern.get("conditions", []):
                condition_id = f"condition_{slugify(condition)}"
                add_node(condition_id, "Condition", name=condition)

                add_edge(
                    pattern_id,
                    "SUGGESTS",
                    condition_id,
                    confidence=pattern.get("confidence"),
                )

    seen = set()
    deduped_edges = []

    for edge in edges:
        key = (edge["source"], edge["relation"], edge["target"])

        if key not in seen:
            seen.add(key)
            deduped_edges.append(edge)

    node_type_counts: dict[str, int] = {}
    edge_type_counts: dict[str, int] = {}

    for node in nodes.values():
        node_type = node.get("node_type", "Unknown")
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

    for edge in deduped_edges:
        relation = edge.get("relation", "Unknown")
        edge_type_counts[relation] = edge_type_counts.get(relation, 0) + 1

    return {
        "nodes": list(nodes.values()),
        "edges": deduped_edges,
        "stats": {
            "num_nodes": len(nodes),
            "num_edges": len(deduped_edges),
            "node_type_counts": node_type_counts,
            "edge_type_counts": edge_type_counts,
        },
    }


def export_graph_csv(graph: dict, output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = output_dir / "nodes.csv"
    edges_path = output_dir / "relationships.csv"

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    node_keys = sorted({key for node in nodes for key in node.keys()})
    edge_keys = sorted({key for edge in edges for key in edge.keys()})

    with open(nodes_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=node_keys)
        writer.writeheader()

        for node in nodes:
            writer.writerow(node)

    with open(edges_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=edge_keys)
        writer.writeheader()

        for edge in edges:
            writer.writerow(edge)


# =========================================================
# PROMPT FORMAT
# =========================================================

def clean_quote(text: str, max_len: int = 350) -> str:
    text = str(text or "").replace("\n", " ")
    text = " ".join(text.split()).strip()

    if len(text) > max_len:
        return text[:max_len].rstrip() + "..."

    return text


def format_abnormal_findings(reasoning_context: dict) -> str:
    items = reasoning_context.get("abnormal_items", [])

    if not items:
        return "- Không phát hiện bất thường rõ ràng."

    lines = []

    for item in items:
        value = item.get("value")

        if value is None:
            value = item.get("raw_value", "")

        line = (
            f"- {item.get('panel')} | {item.get('test')} "
            f"({item.get('test_label')}): {value} {item.get('unit', '')}, "
            f"status = {item.get('status')}"
        )

        if item.get("reference_range"):
            line += f", reference = {item.get('reference_range')}"

        lines.append(line)

    return "\n".join(lines)


def format_patterns(reasoning_context: dict) -> str:
    patterns = reasoning_context.get("detected_patterns", [])

    if not patterns:
        return "- Không phát hiện pattern phối hợp rõ ràng."

    lines = []

    for pattern in patterns[:6]:
        line = (
            f"- {pattern.get('pattern_name')} ({pattern.get('panel')}), "
            f"confidence={pattern.get('confidence')}: {pattern.get('description')}"
        )

        if pattern.get("conditions"):
            line += f" Conditions: {', '.join(pattern.get('conditions'))}"

        lines.append(line)

    return "\n".join(lines)


def format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "- Không có evidence phù hợp."

    lines = []

    for i, e in enumerate(evidence[:MAX_FINAL_EVIDENCE], start=1):
        text = clean_quote(e.get("text", ""))
        source = e.get("source", "Unknown")
        page = e.get("page", "")
        panel = e.get("panel", "UNKNOWN")
        tests = ", ".join(e.get("tests", []))
        conditions = ", ".join(e.get("conditions", []))
        score = e.get("final_score", e.get("score"))

        lines.append(
            f"[{i}] Panel={panel}; Tests={tests}; Conditions={conditions}; "
            f"Source={source}; Page={page}; Score={score}; Quote=\"{text}\""
        )

    return "\n".join(lines)


def build_final_prompt(
    reasoning_context: dict,
    evidence: list[dict],
    reasoning_paths: list[dict] | None = None,
) -> str:
    abnormal_text = format_abnormal_findings(reasoning_context)
    pattern_text = format_patterns(reasoning_context)
    evidence_text = format_evidence(evidence)
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
# LLM
# =========================================================
def call_local_ollama_llm(prompt: str) -> str:
    load_dotenv()

    local_url = os.getenv("LOCAL_LLM_URL", "").strip()
    model_name = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b").strip()

    if not local_url:
        raise ValueError("Missing LOCAL_LLM_URL in .env")

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": COLAB_TEMPERATURE,
            "num_predict": COLAB_MAX_NEW_TOKENS,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }

    print(f"   🤖 Calling Local Ollama: {model_name}...")

    start = time.time()

    response = requests.post(
        local_url,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    elapsed = time.time() - start
    print(f"   ⏱️ Local Ollama returned in {elapsed:.1f}s")

    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict):
        text = (
            data.get("response")
            or data.get("text")
            or data.get("output")
            or data.get("generated_text")
            or data.get("answer")
            or ""
        )
    else:
        text = str(data)

    text = extract_final_answer(str(text))

    if not text:
        raise ValueError(f"Local Ollama returned empty response: {data}")

    return text

def timed_llm_call(provider_name: str, func, prompt: str) -> tuple[str, dict]:
    start = time.time()
    text = func(prompt)
    elapsed = time.time() - start

    meta = {
        "model_used": provider_name,
        "elapsed_seconds": round(elapsed, 2),
    }

    return text, meta


def normalize_colab_url(url: str) -> str:
    url = str(url or "").strip().rstrip("/")

    if not url:
        return ""

    if not url.endswith("/generate"):
        url += "/generate"

    return url


def call_colab_llm(prompt: str) -> str:
    load_dotenv()

    colab_url = normalize_colab_url(os.getenv("COLAB_LLM_URL", ""))

    if not colab_url:
        raise ValueError("Missing COLAB_LLM_URL in .env")

    payload = {
        "prompt": prompt,
        "max_new_tokens": COLAB_MAX_NEW_TOKENS,
        "temperature": COLAB_TEMPERATURE,
    }

    print("   🤖 Calling Colab LLM...")

    start = time.time()

    response = requests.post(
        colab_url,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    elapsed = time.time() - start
    print(f"   ⏱️ Colab LLM returned in {elapsed:.1f}s")

    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict):
        text = (
            data.get("response")
            or data.get("text")
            or data.get("output")
            or data.get("generated_text")
            or data.get("answer")
            or ""
        )
    else:
        text = str(data)

    text = extract_final_answer(str(text))

    if not text:
        raise ValueError("Colab LLM returned empty response")

    return text


def call_gemini_llm(prompt: str) -> str:
    load_dotenv()

    if genai is None:
        raise RuntimeError("google-genai is not installed")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    print(f"   🤖 Calling Gemini: {model_name}...")

    start = time.time()

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    elapsed = time.time() - start
    print(f"   ⏱️ Gemini returned in {elapsed:.1f}s")

    text = str(getattr(response, "text", "") or "").strip()
    text = extract_final_answer(text)

    if not text:
        raise ValueError("Gemini returned empty response")

    return text


def call_deepseek_llm(prompt: str) -> str:
    """
    DeepSeek fallback thông qua OpenRouter.
    .env cần:
    OPENROUTER_API_KEY=...
    OPENROUTER_MODEL=deepseek/deepseek-chat
    """

    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY in .env")

    model_name = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

    print(f"   🤖 Calling OpenRouter/DeepSeek: {model_name}...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": COLAB_TEMPERATURE,
        "max_tokens": COLAB_MAX_NEW_TOKENS,
    }

    start = time.time()

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    elapsed = time.time() - start
    print(f"   ⏱️ OpenRouter/DeepSeek returned in {elapsed:.1f}s")

    response.raise_for_status()
    data = response.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise ValueError(f"Unexpected OpenRouter response format: {data}") from exc

    text = extract_final_answer(str(text))

    if not text:
        raise ValueError("OpenRouter/DeepSeek returned empty response")

    return text

def call_llm_with_meta(prompt: str) -> tuple[str, dict]:
    """
    Provider order:
    1. Local Ollama via LOCAL_LLM_URL
    2. Gemini
    3. OpenRouter/DeepSeek
    4. Colab LLM
    """

    load_dotenv()

    errors = []

    # 1. Local Ollama first
    try:
        model_name = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b").strip()
        return timed_llm_call(
            f"local_ollama/{model_name}",
            call_local_ollama_llm,
            prompt,
        )
    except Exception as exc:
        err = f"Local Ollama failed: {exc}"
        print(f"⚠️ {err}")
        errors.append(err)

    # 2. Gemini fallback
    try:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return timed_llm_call(model_name, call_gemini_llm, prompt)
    except Exception as exc:
        err = f"Gemini failed: {exc}"
        print(f"⚠️ {err}")
        errors.append(err)

    # 3. OpenRouter/DeepSeek fallback
    try:
        model_name = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
        return timed_llm_call(f"openrouter/{model_name}", call_deepseek_llm, prompt)
    except Exception as exc:
        err = f"OpenRouter/DeepSeek failed: {exc}"
        print(f"⚠️ {err}")
        errors.append(err)

    # 4. Colab fallback
    try:
        return timed_llm_call("colab_llm", call_colab_llm, prompt)
    except Exception as exc:
        err = f"Colab LLM failed: {exc}"
        print(f"⚠️ {err}")
        errors.append(err)

    raise RuntimeError("All LLM providers failed:\n" + "\n".join(errors))


def call_llm(prompt: str) -> str:
    """
    Wrapper giữ compatibility với run_final.py hiện tại.
    run_final.py đang gọi call_llm(prompt) và expect string.
    """

    text, _meta = call_llm_with_meta(prompt)
    return text


def extract_final_answer(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = str(raw_text).strip()
    lower = text.lower()

    markers = [
        "\nassistant\n",
        "\nassistant:",
        "assistant\n",
        "assistant:",
    ]

    found_pos = -1
    found_marker = ""

    for marker in markers:
        pos = lower.rfind(marker)

        if pos > found_pos:
            found_pos = pos
            found_marker = marker

    if found_pos != -1:
        cleaned = text[found_pos + len(found_marker):].strip()

        if cleaned:
            return cleaned

    return text.strip()


def has_bad_placeholders(text: str) -> bool:
    if not text:
        return True

    bad_patterns = [
        r"\(\.\.\.\)",
        r"\.\.\.",
        r"\(source,\s*p\.[^)]+\)",
        r"\[citation needed\]",
    ]

    lower = text.lower()

    for pattern in bad_patterns:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            return True

    return False


def build_repair_prompt(bad_answer: str) -> str:
    return f"""
Bạn hãy sửa lại câu trả lời sau cho đúng format citation.

YÊU CẦU:
- Giữ nguyên nội dung y khoa nếu hợp lý.
- Xóa toàn bộ dấu "...".
- Xóa toàn bộ "(...)".
- Xóa toàn bộ "(Source, p.xxx)".
- Nếu cần citation, chỉ dùng dạng [1], [2], [3], [4], [5], [6].
- Không thêm thông tin mới.
- Không nhắc lại prompt.
- Chỉ trả về bản đã sửa bằng tiếng Việt.

NỘI DUNG CẦN SỬA:
{bad_answer}
""".strip()


def generate_clean_answer(prompt: str) -> tuple[str, dict]:
    answer, meta = call_llm_with_meta(prompt)

    if not has_bad_placeholders(answer):
        return answer.strip(), meta

    print("⚠️ Detected placeholder citation. Repairing...")

    repair_prompt = build_repair_prompt(answer)

    try:
        repaired, repair_meta = call_llm_with_meta(repair_prompt)

        if repaired and not has_bad_placeholders(repaired):
            meta["repaired"] = True
            meta["repair_model_used"] = repair_meta.get("model_used")
            meta["repair_elapsed_seconds"] = repair_meta.get("elapsed_seconds")
            meta["total_elapsed_seconds"] = round(
                float(meta.get("elapsed_seconds", 0))
                + float(repair_meta.get("elapsed_seconds", 0)),
                2,
            )
            return repaired.strip(), meta

    except Exception as exc:
        print(f"⚠️ Repair failed: {exc}")

    meta["repaired"] = False
    return mechanical_cleanup_answer(answer), meta


def mechanical_cleanup_answer(text: str) -> str:
    if not text:
        return ""

    text = text.replace("(...)", "")
    text = text.replace("...", "")
    text = re.sub(r"\(Source,\s*p\.[^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def build_references(evidence: list[dict]) -> list[dict]:
    refs = []

    for i, e in enumerate(evidence[:MAX_FINAL_EVIDENCE], start=1):
        refs.append({
            "id": i,
            "panel": e.get("panel"),
            "source": e.get("source"),
            "page": e.get("page"),
            "tests": e.get("tests", []),
            "conditions": e.get("conditions", []),
            "quote": clean_quote(e.get("text", ""), max_len=260),
            "score": e.get("final_score", e.get("score")),
        })

    return refs


def extract_final_answer(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = str(raw_text).strip()
    lower = text.lower()

    markers = [
        "\nassistant\n",
        "\nassistant:",
        "assistant\n",
        "assistant:",
    ]

    found_pos = -1
    found_marker = ""

    for marker in markers:
        pos = lower.rfind(marker)

        if pos > found_pos:
            found_pos = pos
            found_marker = marker

    if found_pos != -1:
        cleaned = text[found_pos + len(found_marker):].strip()

        if cleaned:
            return cleaned

    return text.strip()


def has_bad_placeholders(text: str) -> bool:
    if not text:
        return True

    bad_patterns = [
        r"\(\.\.\.\)",
        r"\(source,\s*p\.[^)]+\)",
        r"\[citation needed\]",
    ]

    lower = text.lower()

    for pattern in bad_patterns:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            return True

    return False


def mechanical_cleanup_answer(text: str) -> str:
    if not text:
        return ""

    text = text.replace("(...)", "")
    text = re.sub(r"\(Source,\s*p\.[^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def build_references(evidence: list[dict]) -> list[dict]:
    refs = []

    for i, e in enumerate(evidence[:MAX_FINAL_EVIDENCE], start=1):
        refs.append({
            "id": i,
            "panel": e.get("panel"),
            "source": e.get("source"),
            "page": e.get("page"),
            "tests": e.get("tests", []),
            "conditions": e.get("conditions", []),
            "quote": clean_quote(e.get("text", ""), max_len=260),
            "score": e.get("final_score", e.get("score")),
        })

    return refs