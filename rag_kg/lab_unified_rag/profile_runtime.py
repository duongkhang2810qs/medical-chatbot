from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from config import OUTPUT_DIR, MAX_FINAL_EVIDENCE
from lab_core import (
    build_reasoning_context,
    retrieve_evidence,
    build_final_prompt,
    generate_clean_answer,
)
from run_final import (
    load_cases_for_final,
    load_cbc_demo_patterns,
    load_biochem_patterns,
    augment_reasoning_context_with_static_patterns,
    enrich_reasoning_paths,
)


OUT_DIR = OUTPUT_DIR / "runtime_profile"
DETAILS_CSV = OUT_DIR / "runtime_profile_details.csv"
SUMMARY_CSV = OUT_DIR / "runtime_profile_summary.csv"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def avg(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows]
    return sum(values) / len(values) if values else 0.0


def profile_one_case(
    case: dict,
    idx: int,
    cbc_demo_patterns: list[dict],
    biochem_patterns: dict,
) -> dict:
    total_start = time.perf_counter()

    # =========================
    # 1. Build reasoning context
    # =========================
    t0 = time.perf_counter()

    ctx = build_reasoning_context(case, idx)
    ctx = augment_reasoning_context_with_static_patterns(
        ctx=ctx,
        cbc_demo_patterns=cbc_demo_patterns,
        biochem_patterns=biochem_patterns,
    )

    context_seconds = time.perf_counter() - t0

    case_id = ctx.get("case_id")
    panel = ",".join(ctx.get("panels", []))

    print("\n" + "=" * 80)
    print(f"[{idx + 1}] Profiling case: {case_id}")
    print(f"Panel: {panel}")
    print(f"Abnormal tests: {ctx.get('abnormal_tests')}")

    # =========================
    # 2. RAG retrieval
    # =========================
    t0 = time.perf_counter()

    book_evidence = retrieve_evidence(ctx)
    final_evidence = book_evidence[:MAX_FINAL_EVIDENCE]

    retrieval_seconds = time.perf_counter() - t0

    print(f"Retrieved evidence: {len(book_evidence)}")
    print(f"Used evidence: {len(final_evidence)}")
    print(f"RAG retrieval: {retrieval_seconds:.2f}s")

    # =========================
    # 3. KG / reasoning paths
    # =========================
    t0 = time.perf_counter()

    graph_reasoning_paths = enrich_reasoning_paths(ctx, final_evidence)

    kg_seconds = time.perf_counter() - t0

    print(f"Reasoning paths: {len(graph_reasoning_paths)}")
    print(f"KG/context reasoning: {kg_seconds:.2f}s")

    # =========================
    # 4. Prompt build
    # =========================
    t0 = time.perf_counter()

    prompt = build_final_prompt(
        reasoning_context=ctx,
        evidence=final_evidence,
        reasoning_paths=graph_reasoning_paths,
    )

    prompt_build_seconds = time.perf_counter() - t0

    # =========================
    # 5. LLM generation
    # =========================
    t0 = time.perf_counter()

    answer, llm_meta = generate_clean_answer(prompt)

    llm_generation_seconds = time.perf_counter() - t0

    total_seconds = time.perf_counter() - total_start

    model_used = llm_meta.get("model_used", "unknown")

    print(f"LLM model: {model_used}")
    print(f"LLM generation: {llm_generation_seconds:.2f}s")
    print(f"Total: {total_seconds:.2f}s")

    return {
        "case_id": case_id,
        "panel": panel,
        "model_used": model_used,
        "context_seconds": round(context_seconds, 4),
        "rag_retrieval_seconds": round(retrieval_seconds, 4),
        "kg_reasoning_seconds": round(kg_seconds, 4),
        "prompt_build_seconds": round(prompt_build_seconds, 4),
        "llm_generation_seconds": round(llm_generation_seconds, 4),
        "total_seconds": round(total_seconds, 4),
        "num_evidence_retrieved": len(book_evidence),
        "num_evidence_used": len(final_evidence),
        "num_reasoning_paths": len(graph_reasoning_paths),
        "answer_chars": len(answer or ""),
    }


def main(limit: int = 5) -> None:
    print("=" * 80)
    print("PROFILE RUNTIME: RAG RETRIEVAL vs LLM GENERATION")
    print("=" * 80)

    cases = load_cases_for_final()[:limit]

    cbc_demo_patterns = load_cbc_demo_patterns()
    biochem_patterns = load_biochem_patterns()

    print(f"Số case đo thời gian: {len(cases)}")

    rows = []

    for idx, case in enumerate(cases):
        try:
            row = profile_one_case(
                case=case,
                idx=idx,
                cbc_demo_patterns=cbc_demo_patterns,
                biochem_patterns=biochem_patterns,
            )
            rows.append(row)

        except Exception as exc:
            print(f"Failed profiling case index={idx}: {exc}")

    if not rows:
        print("Không có case nào đo thành công.")
        return

    summary_rows = [
        {
            "Module": "Build context + pattern context",
            "Thời gian TB (giây/ca)": round(avg(rows, "context_seconds"), 4),
        },
        {
            "Module": "RAG retrieval",
            "Thời gian TB (giây/ca)": round(avg(rows, "rag_retrieval_seconds"), 4),
        },
        {
            "Module": "KG/context reasoning",
            "Thời gian TB (giây/ca)": round(avg(rows, "kg_reasoning_seconds"), 4),
        },
        {
            "Module": "Prompt building",
            "Thời gian TB (giây/ca)": round(avg(rows, "prompt_build_seconds"), 4),
        },
        {
            "Module": "LLM generation",
            "Thời gian TB (giây/ca)": round(avg(rows, "llm_generation_seconds"), 4),
        },
        {
            "Module": "Tổng",
            "Thời gian TB (giây/ca)": round(avg(rows, "total_seconds"), 4),
        },
    ]

    write_csv(DETAILS_CSV, rows)
    write_csv(SUMMARY_CSV, summary_rows)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for row in summary_rows:
        print(row)

    print("\nSaved:")
    print(f"- {DETAILS_CSV}")
    print(f"- {SUMMARY_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Số case dùng để đo thời gian trung bình.",
    )

    args = parser.parse_args()

    main(limit=args.limit)