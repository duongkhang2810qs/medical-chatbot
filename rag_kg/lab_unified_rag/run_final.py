# run_final.py
from __future__ import annotations

import json
from typing import Any

from config import (
    CBC_CASE_PATH,
    BIOCHEM_CASE_PATH,
    LAB_CASE_PATH,
    OUTPUT_PATH,
    FAILED_PATH,
    MAX_FINAL_EVIDENCE,
    CBC_DEMO_PATTERN_PATH,
    BIOCHEM_PATTERN_PATH,
)
from lab_core import (
    load_json,
    load_jsonl,
    append_jsonl,
    merge_case_lists,
    build_reasoning_context,
    retrieve_evidence,
    build_reasoning_paths,
    build_final_prompt,
    generate_clean_answer,
    mechanical_cleanup_answer,
    normalize_test_name,
    normalize_status,
    safe_float,
)

# =========================================================
# LOAD CASES
# =========================================================

def load_processed_case_ids(path) -> set[str]:
    processed = set()

    for item in load_jsonl(path):
        case_id = item.get("case_id")
        if case_id:
            processed.add(str(case_id))

    return processed


def load_cases_for_final() -> list[dict]:
    """
    Ưu tiên:
    1. Nếu có all_results_lab.jsonl thì đọc file merged.
    2. Nếu chưa có thì đọc 2 file riêng:
       - all_results.jsonl
       - all_results_biochemistry.jsonl
       rồi merge theo case_id/id.
    """

    if LAB_CASE_PATH.exists():
        print(f"Loading merged case file: {LAB_CASE_PATH}")
        return load_jsonl(LAB_CASE_PATH)

    print("Merged case file chưa có, đọc 2 file riêng CBC + BIOCHEM...")

    cbc_cases = load_jsonl(CBC_CASE_PATH)
    biochem_cases = load_jsonl(BIOCHEM_CASE_PATH)

    print(f"CBC cases: {len(cbc_cases)}")
    print(f"BIOCHEM cases: {len(biochem_cases)}")

    merged_cases = merge_case_lists(cbc_cases, biochem_cases)

    print(f"Merged cases by same id/case_id: {len(merged_cases)}")

    return merged_cases


# =========================================================
# STATIC PATTERN LOADERS
# =========================================================

def load_cbc_demo_patterns() -> list[dict]:
    if not CBC_DEMO_PATTERN_PATH.exists():
        print(f"Warning: missing CBC demo pattern file: {CBC_DEMO_PATTERN_PATH}")
        return []

    return load_jsonl(CBC_DEMO_PATTERN_PATH)


def load_biochem_patterns() -> dict:
    if not BIOCHEM_PATTERN_PATH.exists():
        print(f"Warning: missing BIOCHEM pattern file: {BIOCHEM_PATTERN_PATH}")
        return {}

    data = load_json(BIOCHEM_PATTERN_PATH)

    if not isinstance(data, dict):
        raise ValueError("biochem_patterns.json must be a JSON object")

    return data


# =========================================================
# STATIC PATTERN MATCHING
# =========================================================

def slug_condition(text: str) -> str:
    text = str(text or "").lower().strip()
    out = []
    prev_is_sep = False

    for ch in text:
        if ch.isalnum():
            out.append(ch)
            prev_is_sep = False
        else:
            if not prev_is_sep:
                out.append("_")
                prev_is_sep = True

    return "".join(out).strip("_")


def build_static_evidence_from_context(ctx: dict, max_items: int = 3) -> list[dict]:
    evidence = []

    for idx, pattern in enumerate(ctx.get("static_context", [])[:max_items], start=1):
        text_parts = []

        if pattern.get("pattern_name"):
            text_parts.append(str(pattern["pattern_name"]))

        if pattern.get("description"):
            text_parts.append(str(pattern["description"]))

        extra = pattern.get("extra", {}) or {}

        if extra.get("clinical_flags"):
            text_parts.append("Lưu ý lâm sàng: " + "; ".join(extra["clinical_flags"][:2]))

        if extra.get("next_steps"):
            text_parts.append("Gợi ý đánh giá tiếp: " + "; ".join(extra["next_steps"][:2]))

        text = ". ".join([x for x in text_parts if x]).strip()

        if not text:
            continue

        evidence.append({
            "evidence_id": f"static_pattern_{idx}",
            "panel": pattern.get("panel", "STATIC"),
            "text": text,
            "tests": [],
            "conditions": pattern.get("conditions", []),
            "topics": [],
            "keywords": [],
            "type": "static_pattern_context",
            "score": 1.0,
            "final_score": 2.0,
            "trust": 0.95,
            "source": pattern.get("source", "static_pattern_file"),
            "page": "static",
            "source_type": "static_pattern_context",
            "is_static": True,
        })

    return evidence

def has_case_finding(ctx: dict, panel: str, test: str, status: str) -> bool:
    panel = str(panel).upper()
    test = normalize_test_name(test, panel)
    status = normalize_status(status)

    for item in ctx.get("abnormal_items", []):
        if (
            item.get("panel") == panel
            and item.get("test") == test
            and normalize_status(item.get("status")) == status
        ):
            return True

    return False


def case_simple_tags(ctx: dict) -> set[str]:
    tags = set()

    for item in ctx.get("abnormal_items", []):
        test = item.get("test")
        status = normalize_status(item.get("status"))

        if test and status:
            tags.add(f"{test}_{status.capitalize()}")

    return tags


def match_cbc_demo_patterns(ctx: dict, demo_patterns: list[dict]) -> list[dict]:
    """
    Hỗ trợ file demo_cases_all_clean.jsonl.
    Kỳ vọng mỗi row có thể có:
    - input: {"WBC": "high", "NEUT": "high"}
    - patterns: [...]
    - combined_interpretation
    """

    matches = []

    for row in demo_patterns:
        raw_input = row.get("input", {}) or {}

        if not isinstance(raw_input, dict):
            continue

        required = []

        for raw_test, raw_status in raw_input.items():
            test = normalize_test_name(raw_test, "CBC")
            status = normalize_status(raw_status)
            required.append((test, status))

        if not required:
            continue

        hit = 0

        for test, status in required:
            if has_case_finding(ctx, "CBC", test, status):
                hit += 1

        ratio = hit / max(len(required), 1)

        if hit == 0:
            continue

        if len(required) >= 2 and ratio < 0.6:
            continue

        if len(required) == 1 and ratio < 1.0:
            continue

        patterns = row.get("patterns", []) or []

        if not patterns:
            combined = row.get("combined_interpretation", "")
            patterns = [{
                "pattern_name": row.get("case_id", "CBC demo pattern"),
                "interpretation": combined,
                "confidence": row.get("confidence", "medium"),
                "match_score": ratio,
                "matched_conditions": [],
            }]

        for pattern in patterns:
            pattern_name = pattern.get("pattern_name", "CBC demo pattern")
            interpretation = pattern.get("interpretation") or row.get("combined_interpretation", "")

            matches.append({
                "pattern_id": f"cbc_static_{slug_condition(pattern_name)}",
                "pattern_name": pattern_name,
                "panel": "CBC",
                "conditions": [slug_condition(pattern_name)],
                "description": interpretation,
                "confidence": round(safe_float(pattern.get("match_score"), ratio), 2),
                "matched_required": [f"{t}_{s}" for t, s in required],
                "matched_optional": [],
                "source": "cbc_demo_cases",
                "static_rule": True,
            })

    return matches


def match_biochem_static_patterns(ctx: dict, biochem_patterns: dict) -> list[dict]:
    matches = []
    tags = case_simple_tags(ctx)

    single = biochem_patterns.get("single_test_patterns", {}) or {}

    # Single-test interpretation
    for item in ctx.get("abnormal_items", []):
        if item.get("panel") != "BIOCHEM":
            continue

        test = item.get("test")
        status = normalize_status(item.get("status"))

        if not test or not status:
            continue

        test_rule = single.get(test, {}) or {}
        status_rule = test_rule.get(status, {}) or {}

        if not status_rule:
            continue

        label = status_rule.get("label") or f"{test} {status}"
        clinical_meaning = status_rule.get("clinical_meaning") or status_rule.get("note") or ""

        matches.append({
            "pattern_id": f"biochem_single_{test}_{status}",
            "pattern_name": label,
            "panel": "BIOCHEM",
            "conditions": [slug_condition(label)],
            "description": clinical_meaning,
            "confidence": 0.85,
            "matched_required": [f"{test}_{status}"],
            "matched_optional": [],
            "source": "biochem_patterns_single",
            "static_rule": True,
            "extra": {
                "associated_tests": status_rule.get("associated_tests", []),
                "clinical_flags": status_rule.get("clinical_flags", []),
                "causes": status_rule.get("causes", []),
            },
        })

    # Combination patterns
    combos = biochem_patterns.get("pattern_combinations", []) or []

    for combo in combos:
        required_tags = combo.get("required_tags", []) or []
        optional_tags = combo.get("optional_tags", []) or []
        confidence_required = int(combo.get("confidence_required", 1) or 1)

        required_hits = [tag for tag in required_tags if tag in tags]
        optional_hits = [tag for tag in optional_tags if tag in tags]

        if required_tags:
            if len(required_hits) < len(required_tags):
                continue
        else:
            if len(optional_hits) < confidence_required:
                continue

        total_possible = len(required_tags) + len(optional_tags)
        total_hits = len(required_hits) + len(optional_hits)
        match_score = total_hits / max(total_possible, 1)

        matches.append({
            "pattern_id": combo.get("pattern_id", "biochem_combo_pattern"),
            "pattern_name": combo.get("name", "BIOCHEM combination pattern"),
            "panel": "BIOCHEM",
            "conditions": [slug_condition(combo.get("pattern_id", combo.get("name", "biochem_pattern")))],
            "description": combo.get("interpretation", ""),
            "confidence": round(match_score, 2),
            "matched_required": [x.lower() for x in required_hits],
            "matched_optional": [x.lower() for x in optional_hits],
            "source": "biochem_patterns_combo",
            "static_rule": True,
            "extra": {
                "next_steps": combo.get("next_steps", []),
                "sources": combo.get("sources", []),
                "severity_escalators": combo.get("severity_escalators", {}),
            },
        })

    return matches


def build_safety_warnings(ctx: dict, biochem_patterns: dict) -> list[dict]:
    """
    Rule-based safety flags from biochem_patterns.json.
    Không thay thế bác sĩ, chỉ cảnh báo nguy cơ khi giá trị rất bất thường.
    """

    warnings = []

    for item in ctx.get("abnormal_items", []):
        if item.get("panel") != "BIOCHEM":
            continue

        test = item.get("test")
        value = safe_float(item.get("value"), None)

        if value is None:
            continue

        # Hard-coded from biochem_patterns.json critical_values
        if test == "K" and value >= 6.5:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Kali ≥ 6.5 mmol/L: nguy cơ rối loạn nhịp tim đe dọa tính mạng, cần ECG và xử trí cấp cứu.",
            })

        if test == "K" and value <= 2.5:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Kali ≤ 2.5 mmol/L: nguy cơ loạn nhịp và yếu/liệt cơ, cần xử trí y tế.",
            })

        if test == "NA" and value <= 120:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Natri ≤ 120 mmol/L: hạ natri nặng, nguy cơ phù não/co giật.",
            })

        if test == "NA" and value >= 160:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Natri ≥ 160 mmol/L: tăng natri nặng, nguy cơ rối loạn tri giác/hôn mê.",
            })

        if test == "GLUCOSE" and value < 3.0:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Glucose < 3.0 mmol/L: hạ đường huyết nặng, cần xử trí ngay.",
            })

        if test == "GLUCOSE" and value >= 20:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Glucose ≥ 20 mmol/L: tăng đường huyết nặng, cần đánh giá DKA/HHS.",
            })

        if test == "CALCIUM_ION" and value >= 1.75:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Canxi ion hóa ≥ 1.75 mmol/L: tăng canxi nặng, có thể là cấp cứu.",
            })

        if test == "CALCIUM_ION" and value <= 0.80:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Canxi ion hóa ≤ 0.80 mmol/L: giảm canxi nặng, nguy cơ co giật/loạn nhịp.",
            })

        if test == "TRIGLYCERIDE" and value >= 11.3:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Triglyceride ≥ 11.3 mmol/L: nguy cơ viêm tụy cấp rất cao.",
            })

        if test == "CREATININE" and value >= 354:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Creatinine ≥ 354 µmol/L: gợi ý suy thận nặng hoặc AKI nặng, cần đánh giá thận học.",
            })

        if test == "UREA" and value >= 25:
            warnings.append({
                "test": test,
                "value": value,
                "unit": item.get("unit", ""),
                "level": "critical",
                "message": "Urê ≥ 25 mmol/L: tăng urê rất cao, cần đánh giá kèm triệu chứng và điện giải.",
            })

    return warnings


def augment_reasoning_context_with_static_patterns(
    ctx: dict,
    cbc_demo_patterns: list[dict],
    biochem_patterns: dict,
) -> dict:
    ctx = dict(ctx)

    static_context = []
    static_context.extend(match_cbc_demo_patterns(ctx, cbc_demo_patterns))
    static_context.extend(match_biochem_static_patterns(ctx, biochem_patterns))

    # Không merge static_context vào detected_patterns để tránh output rác.
    # detected_patterns chỉ giữ rule-based patterns thật sự từ case.
    detected_patterns = ctx.get("detected_patterns", [])

    conditions = sorted({
        condition
        for pattern in detected_patterns
        for condition in pattern.get("conditions", [])
    })

    safety_warnings = build_safety_warnings(ctx, biochem_patterns)

    ctx["detected_patterns"] = detected_patterns
    ctx["conditions"] = conditions
    ctx["static_context"] = static_context
    ctx["safety_warnings"] = safety_warnings

    return ctx


# =========================================================
# GRAPH REASONING PATHS
# =========================================================

def enrich_reasoning_paths(ctx: dict, evidence: list[dict]) -> list[dict]:
    """
    Build path:
    Case → Finding → Pattern → Condition → Evidence → Source
    dùng object trong memory, không cần query Neo4j.
    """

    base_paths = build_reasoning_paths(ctx, evidence)
    enriched = []

    for path in base_paths:
        finding = path.get("finding", {})
        patterns = path.get("patterns", [])

        candidate_conditions = set()
        for pattern in patterns:
            for condition in pattern.get("conditions", []):
                candidate_conditions.add(condition)

        related_evidence = []

        for e in evidence:
            e_tests = set(e.get("tests", []))
            e_conditions = set(e.get("conditions", []))

            same_test = finding.get("test") in e_tests
            same_condition = bool(candidate_conditions & e_conditions)

            if same_test or same_condition:
                related_evidence.append({
                    "evidence_id": e.get("evidence_id"),
                    "panel": e.get("panel"),
                    "source": e.get("source"),
                    "page": e.get("page"),
                    "tests": e.get("tests", []),
                    "conditions": e.get("conditions", []),
                    "score": e.get("final_score", e.get("score")),
                    "link_reason": "same_test" if same_test else "same_condition",
                })

        enriched.append({
            "case_id": ctx.get("case_id"),
            "finding": finding,
            "patterns": patterns[:3],
            "conditions": sorted(candidate_conditions),
            "evidence": related_evidence[:4],
            "path_text": build_path_text(ctx.get("case_id"), finding, patterns, related_evidence),
        })

    return enriched


def build_path_text(case_id: str, finding: dict, patterns: list[dict], evidence: list[dict]) -> str:
    finding_text = (
        f"Case {case_id} → Finding {finding.get('panel')} {finding.get('test')} "
        f"{finding.get('status')}"
    )

    if patterns:
        pattern_text = " → ".join([
            f"Pattern {p.get('pattern_name')}"
            for p in patterns[:2]
        ])
    else:
        pattern_text = "No matched pattern"

    conditions = sorted({
        c
        for p in patterns
        for c in p.get("conditions", [])
    })

    if conditions:
        condition_text = "Condition " + ", ".join(conditions[:3])
    else:
        condition_text = "No condition"

    if evidence:
        e = evidence[0]
        evidence_text = f"Evidence {e.get('evidence_id')} ({e.get('source')}, p.{e.get('page')})"
    else:
        evidence_text = "No direct evidence"

    return f"{finding_text} → {pattern_text} → {condition_text} → {evidence_text}"


# =========================================================
# RUN ONE CASE
# =========================================================
def clean_reference_quote(text: str, max_len: int = 360) -> str:
    text = str(text or "").replace("\n", " ")
    text = " ".join(text.split()).strip()

    if len(text) > max_len:
        return text[:max_len].rstrip() + " […]"

    return text


def build_references_block(evidence: list[dict]) -> str:
    if not evidence:
        return "📚 References:\n- Không có book evidence phù hợp để trích dẫn."

    lines = ["📚 References:"]

    for i, e in enumerate(evidence[:MAX_FINAL_EVIDENCE], start=1):
        source = e.get("source", "Unknown source")
        page = e.get("page", "")
        quote = clean_reference_quote(e.get("text", ""))

        if page:
            lines.append(f'[{i}] {source}, page {page}. “{quote}”')
        else:
            lines.append(f'[{i}] {source}. “{quote}”')

    return "\n".join(lines)


def build_source_intro(panels: list[str]) -> str:
    panels = set(panels or [])

    lines = ["📚 Nguồn tài liệu:"]

    if "CBC" in panels:
        lines.extend([
            "- Harrison's Principles of Internal Medicine: Giáo trình nội khoa kinh điển, được sử dụng rộng rãi trong đào tạo bác sĩ toàn cầu.",
            "- Clinical Hematology: Tài liệu chuyên sâu về huyết học lâm sàng, hỗ trợ diễn giải công thức máu và các rối loạn huyết học.",
        ])

    if "BIOCHEM" in panels:
        lines.extend([
            "- Henry’s Clinical Diagnosis and Management by Laboratory Methods: Tài liệu chuẩn về diễn giải xét nghiệm cận lâm sàng và y học xét nghiệm.",
            "- Tietz Fundamentals of Clinical Chemistry and Molecular Diagnostics: Tài liệu nền tảng về hóa sinh lâm sàng, xét nghiệm sinh hóa và marker bệnh lý.",
        ])

    lines.append("")
    lines.append("Lưu ý: Nội dung chỉ có mục đích hỗ trợ diễn giải xét nghiệm, không thay thế chẩn đoán hoặc chỉ định điều trị của bác sĩ.")

    return "\n".join(lines)


def build_user_visible_answer(answer: str, ctx: dict, evidence: list[dict]) -> str:
    answer = mechanical_cleanup_answer(answer)

    references_block = build_references_block(evidence)
    source_intro = build_source_intro(ctx.get("panels", []))

    return f"{answer}\n\n{references_block}\n\n{source_intro}".strip()

def process_one_case(
    case: dict,
    idx: int,
    cbc_demo_patterns: list[dict],
    biochem_patterns: dict,
) -> dict:
    ctx = build_reasoning_context(case, idx)
    ctx = augment_reasoning_context_with_static_patterns(
        ctx=ctx,
        cbc_demo_patterns=cbc_demo_patterns,
        biochem_patterns=biochem_patterns,
    )

    case_id = ctx["case_id"]

    print("\n" + "=" * 80)
    print(f"Processing case: {case_id}")
    print(f"Panels: {ctx.get('panels')}")
    print(f"Abnormal tests: {ctx.get('abnormal_tests')}")

    patterns = ctx.get("detected_patterns", [])
    if patterns:
        print("Detected patterns:")
        for p in patterns[:8]:
            src = p.get("source", "rule")
            print(
                f"  - {p.get('pattern_name')} | {p.get('panel')} | "
                f"confidence={p.get('confidence')} | source={src}"
            )
    else:
        print("Detected patterns: none")

    if ctx.get("safety_warnings"):
        print("Safety warnings:")
        for w in ctx["safety_warnings"]:
            print(f"  - {w.get('message')}")

    book_evidence = retrieve_evidence(ctx)
    static_evidence = build_static_evidence_from_context(ctx, max_items=3)

    # Static evidence chỉ dùng như context nội bộ, không dùng làm citation.
    # Citation [1], [2], [3] phải ưu tiên từ sách PDF để user kiểm tra lại được.
    final_evidence = book_evidence[:MAX_FINAL_EVIDENCE]

    print(f"Book evidence retrieved: {len(book_evidence)}")
    print(f"Static context available: {len(static_evidence)}")
    print(f"Book evidence used for citation: {len(final_evidence)}")

    graph_reasoning_paths = enrich_reasoning_paths(ctx, final_evidence)

    prompt = build_final_prompt(
        reasoning_context=ctx,
        evidence=final_evidence,
        reasoning_paths=graph_reasoning_paths,
    )

    answer, llm_meta = generate_clean_answer(prompt)
    answer = build_user_visible_answer(
        answer=answer,
        ctx=ctx,
        evidence=final_evidence,
    )

    elapsed = llm_meta.get(
        "total_elapsed_seconds",
        llm_meta.get("elapsed_seconds", 0.0),
    )

    return {
        "case_id": case_id,
        "model_used": llm_meta.get("model_used", "unknown"),
        "elapsed_seconds": elapsed,
        "answer": answer,
    }


# =========================================================
# MAIN
# =========================================================


def main():
    print("=" * 80)
    print("RUN UNIFIED LAB FINAL PIPELINE")
    print("=" * 80)

    cases = load_cases_for_final()

    # Resume từ final_output.jsonl hiện có
    # Không xóa file cũ nữa.
    processed = load_processed_case_ids(OUTPUT_PATH)

    cbc_demo_patterns = load_cbc_demo_patterns()
    biochem_patterns = load_biochem_patterns()

    print(f"Total cases: {len(cases)}")
    print(f"Already processed in final_output: {len(processed)}")
    print(f"CBC static pattern rows: {len(cbc_demo_patterns)}")
    print(f"BIOCHEM pattern sections: {list(biochem_patterns.keys())[:5]}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Failed path: {FAILED_PATH}")

    for idx, case in enumerate(cases):
        preview_ctx = build_reasoning_context(case, idx)
        case_id = preview_ctx["case_id"]

        if case_id in processed:
            print(f"Skip processed case: {case_id}")
            continue

        try:
            result = process_one_case(
                case=case,
                idx=idx,
                cbc_demo_patterns=cbc_demo_patterns,
                biochem_patterns=biochem_patterns,
            )

            append_jsonl(OUTPUT_PATH, result)
            processed.add(case_id)
            print(f"Saved result: {case_id}")

        except Exception as exc:
            append_jsonl(FAILED_PATH, {
                "case_id": case_id,
                "error": str(exc),
            })
            print(f"Failed case {case_id}: {exc}")

    print("\nDONE")
    print(f"Final output: {OUTPUT_PATH}")
    print(f"Failed cases: {FAILED_PATH}")
    print(f"Total successful cases now: {len(load_processed_case_ids(OUTPUT_PATH))}")


if __name__ == "__main__":
    main()