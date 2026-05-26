from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from config import (
    CBC_CASE_PATH,
    BIOCHEM_CASE_PATH,
    OUTPUT_DIR,
)
from lab_core import (
    load_jsonl,
    save_json,
    normalize_single_case,
    build_reasoning_context,
    retrieve_evidence,
    call_llm_with_meta,
)


DEFAULT_OUT_DIR = OUTPUT_DIR / "retrieval_eval"
DETAILS_CSV = DEFAULT_OUT_DIR / "retrieval_eval_details.csv"
DETAILS_JSONL = DEFAULT_OUT_DIR / "retrieval_eval_details.jsonl"
SUMMARY_JSON = DEFAULT_OUT_DIR / "retrieval_eval_summary.json"
SUMMARY_CSV = DEFAULT_OUT_DIR / "retrieval_eval_summary.csv"


def safe_text(value: Any, limit: int | None = None) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_cases(path: Path, default_panel: str, limit: int | None = None) -> list[dict]:
    raw_cases = load_jsonl(path)

    if limit is not None:
        raw_cases = raw_cases[:limit]

    cases = []
    for idx, case in enumerate(raw_cases):
        cases.append(
            normalize_single_case(
                case,
                idx=idx,
                default_panel=default_panel,
            )
        )

    return cases


def format_case_for_prompt(ctx: dict) -> str:
    rows = []

    for item in ctx.get("abnormal_items", []):
        rows.append(
            {
                "panel": item.get("panel"),
                "test": item.get("test"),
                "test_label": item.get("test_label"),
                "value": item.get("value"),
                "unit": item.get("unit"),
                "status": item.get("status"),
                "reference_range": item.get("reference_range"),
            }
        )

    return json.dumps(rows, ensure_ascii=False, indent=2)


def build_eval_prompt(ctx: dict, chunk: dict, rank: int) -> str:
    return f"""
Bạn là người chấm retrieval cho hệ RAG diễn giải xét nghiệm.

CASE BẤT THƯỜNG:
{format_case_for_prompt(ctx)}

CHUNK TRUY XUẤT rank={rank}:
- panel: {chunk.get('panel')}
- tests metadata: {chunk.get('tests', [])}
- conditions metadata: {chunk.get('conditions', [])}
- source: {chunk.get('source')}, page: {chunk.get('page')}
- text: {safe_text(chunk.get('text'), 1800)}

Hãy chấm theo thang:
0 = Không liên quan đến bất thường xét nghiệm của case.
1 = Liên quan một phần: cùng panel/test/chủ đề nhưng chưa giải thích trực tiếp ý nghĩa tăng/giảm hoặc pattern của case.
2 = Liên quan trực tiếp: giải thích rõ ý nghĩa, nguyên nhân, pattern hoặc diễn giải lâm sàng cho ít nhất một bất thường chính trong case.

Đánh giá cite_ok:
- true nếu chunk đủ dùng làm nguồn trích dẫn trong bài diễn giải: phải liên quan trực tiếp, có nội dung giải thích/ý nghĩa/nguyên nhân/pattern, không chỉ nhắc tên xét nghiệm.
- false nếu không đủ dùng làm citation.

Chỉ trả về JSON hợp lệ, không markdown:
{{
  "score": 0,
  "cite_ok": false,
  "reason": "một câu ngắn bằng tiếng Việt"
}}
""".strip()


def extract_json_object(text: str) -> dict:
    text = str(text or "").strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Không tìm thấy JSON object trong LLM output: {text[:300]}")

    return json.loads(match.group(0))


def judge_chunk_with_llm(
    ctx: dict,
    chunk: dict,
    rank: int,
    retries: int = 2,
) -> tuple[dict, dict]:
    prompt = build_eval_prompt(ctx, chunk, rank)
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            raw, meta = call_llm_with_meta(prompt)
            obj = extract_json_object(raw)

            score = int(obj.get("score", 0))
            score = max(0, min(2, score))

            cite_ok = bool(obj.get("cite_ok", False))
            reason = safe_text(obj.get("reason", ""), 500)

            # Cite@5 chỉ tính nếu chunk đạt liên quan trực tiếp.
            if score < 2:
                cite_ok = False

            return {
                "score": score,
                "cite_ok": cite_ok,
                "reason": reason,
                "raw_eval": obj,
            }, meta

        except Exception as exc:
            last_error = exc
            print(f"      ⚠️ LLM judge failed attempt {attempt}/{retries}: {exc}")
            time.sleep(1)

    return {
        "score": 0,
        "cite_ok": False,
        "reason": f"LLM judge failed: {last_error}",
        "raw_eval": {},
    }, {"model_used": "ERROR", "elapsed_seconds": 0}


def compute_case_metrics(
    chunk_scores: list[int],
    cite_flags: list[bool],
    k: int = 5,
) -> dict:
    padded_scores = (chunk_scores + [0] * k)[:k]
    padded_cites = (cite_flags + [False] * k)[:k]

    rel_at_5 = sum(padded_scores) / k
    hit_at_5 = 1 if any(score == 2 for score in padded_scores) else 0
    cite_at_5 = 1 if any(padded_cites) else 0

    return {
        "Rel@5": round(rel_at_5, 4),
        "Hit@5": hit_at_5,
        "Cite@5": cite_at_5,
    }


def summarize(case_rows: list[dict]) -> list[dict]:
    panels = ["CBC", "BIOCHEM"]
    summary = []

    for panel in panels:
        rows = [r for r in case_rows if r["eval_panel"] == panel]

        if not rows:
            continue

        summary.append(
            {
                "Panel": "Sinh hóa" if panel == "BIOCHEM" else "CBC",
                "panel_key": panel,
                "Số case": len(rows),
                "Rel@5": round(sum(r["Rel@5"] for r in rows) / len(rows), 4),
                "Hit@5": round(sum(r["Hit@5"] for r in rows) / len(rows), 4),
                "Cite@5": round(sum(r["Cite@5"] for r in rows) / len(rows), 4),
            }
        )

    if case_rows:
        summary.append(
            {
                "Panel": "Trung bình",
                "panel_key": "ALL",
                "Số case": len(case_rows),
                "Rel@5": round(sum(r["Rel@5"] for r in case_rows) / len(case_rows), 4),
                "Hit@5": round(sum(r["Hit@5"] for r in case_rows) / len(case_rows), 4),
                "Cite@5": round(sum(r["Cite@5"] for r in case_rows) / len(case_rows), 4),
            }
        )

    return summary


def write_summary_csv(path: Path, summary_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Panel", "Số case", "Rel@5", "Hit@5", "Cite@5"],
        )
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(
                {
                    "Panel": row.get("Panel"),
                    "Số case": row.get("Số case"),
                    "Rel@5": row.get("Rel@5"),
                    "Hit@5": row.get("Hit@5"),
                    "Cite@5": row.get("Cite@5"),
                }
            )


def write_details_csv(path: Path, detail_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "eval_panel",
        "rank",
        "evidence_id",
        "chunk_panel",
        "source",
        "page",
        "retrieval_score",
        "final_score",
        "tests",
        "conditions",
        "llm_score",
        "cite_ok",
        "judge_reason",
        "judge_model",
        "text",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in detail_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def evaluate_panel(
    cases: list[dict],
    eval_panel: str,
    k: int,
    judge: bool,
) -> tuple[list[dict], list[dict]]:
    case_metric_rows = []
    detail_rows = []

    for idx, case in enumerate(cases, start=1):
        ctx = build_reasoning_context(case, idx=idx)
        case_id = ctx.get("case_id", f"case_{idx:04d}")

        print("=" * 80)
        print(f"[{eval_panel}] Case {idx}/{len(cases)}: {case_id}")

        evidence = retrieve_evidence(ctx)[:k]
        print(f"   Retrieved chunks: {len(evidence)}")

        scores: list[int] = []
        cites: list[bool] = []

        for rank in range(1, k + 1):
            chunk = evidence[rank - 1] if rank <= len(evidence) else {}

            if chunk and judge:
                judged, meta = judge_chunk_with_llm(ctx, chunk, rank)
            elif chunk:
                judged, meta = {
                    "score": 0,
                    "cite_ok": False,
                    "reason": "not judged",
                    "raw_eval": {},
                }, {}
            else:
                judged, meta = {
                    "score": 0,
                    "cite_ok": False,
                    "reason": "missing chunk",
                    "raw_eval": {},
                }, {}

            scores.append(int(judged["score"]))
            cites.append(bool(judged["cite_ok"]))

            row = {
                "case_id": case_id,
                "eval_panel": eval_panel,
                "rank": rank,
                "evidence_id": chunk.get("evidence_id", ""),
                "chunk_panel": chunk.get("panel", ""),
                "source": chunk.get("source", ""),
                "page": chunk.get("page", ""),
                "retrieval_score": chunk.get("score", ""),
                "final_score": chunk.get("final_score", ""),
                "tests": json.dumps(chunk.get("tests", []), ensure_ascii=False),
                "conditions": json.dumps(chunk.get("conditions", []), ensure_ascii=False),
                "llm_score": judged["score"],
                "cite_ok": judged["cite_ok"],
                "judge_reason": judged["reason"],
                "judge_model": meta.get("model_used", ""),
                "text": safe_text(chunk.get("text", ""), 3000),
            }

            detail_rows.append(row)
            append_jsonl(
                DETAILS_JSONL,
                {
                    **row,
                    "raw_eval": judged.get("raw_eval", {}),
                },
            )

            print(
                f"      rank {rank}: "
                f"score={judged['score']} "
                f"cite_ok={judged['cite_ok']} - "
                f"{judged['reason']}"
            )

        metrics = compute_case_metrics(scores, cites, k=k)

        case_metric_rows.append(
            {
                "case_id": case_id,
                "eval_panel": eval_panel,
                **metrics,
            }
        )

        print(
            f"   Case metrics: "
            f"Rel@5={metrics['Rel@5']} "
            f"Hit@5={metrics['Hit@5']} "
            f"Cite@5={metrics['Cite@5']}"
        )

    return case_metric_rows, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval@5 with automatic LLM judging."
    )

    parser.add_argument("--cbc-path", type=Path, default=CBC_CASE_PATH)
    parser.add_argument("--bio-path", type=Path, default=BIOCHEM_CASE_PATH)
    parser.add_argument("--cbc-limit", type=int, default=22)
    parser.add_argument("--bio-limit", type=int, default=7)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Chỉ retrieve, không gọi LLM chấm điểm.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xóa output cũ trước khi chạy.",
    )

    args = parser.parse_args()

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset:
        for p in [DETAILS_CSV, DETAILS_JSONL, SUMMARY_JSON, SUMMARY_CSV]:
            if p.exists():
                p.unlink()

    print("=" * 80)
    print("EVALUATE RETRIEVAL")
    print("=" * 80)
    print(f"CBC path: {args.cbc_path}")
    print(f"BIOCHEM path: {args.bio_path}")
    print(f"K: {args.k}")
    print("Lưu ý: script KHÔNG build lại embedding/Qdrant; chỉ query collection đã có.")

    cbc_cases = load_cases(
        args.cbc_path,
        default_panel="CBC",
        limit=args.cbc_limit,
    )

    bio_cases = load_cases(
        args.bio_path,
        default_panel="BIOCHEM",
        limit=args.bio_limit,
    )

    print(f"CBC cases loaded: {len(cbc_cases)}")
    print(f"BIOCHEM cases loaded: {len(bio_cases)}")

    all_case_metrics = []
    all_details = []

    cbc_metrics, cbc_details = evaluate_panel(
        cbc_cases,
        "CBC",
        k=args.k,
        judge=not args.no_judge,
    )
    all_case_metrics.extend(cbc_metrics)
    all_details.extend(cbc_details)

    bio_metrics, bio_details = evaluate_panel(
        bio_cases,
        "BIOCHEM",
        k=args.k,
        judge=not args.no_judge,
    )
    all_case_metrics.extend(bio_metrics)
    all_details.extend(bio_details)

    summary_rows = summarize(all_case_metrics)

    write_details_csv(DETAILS_CSV, all_details)
    write_summary_csv(SUMMARY_CSV, summary_rows)

    save_json(
        SUMMARY_JSON,
        {
            "metric_definitions": {
                "Rel@5": "Trung bình điểm liên quan của 5 chunk được truy xuất: (s1+s2+s3+s4+s5)/5",
                "Hit@5": "1 nếu tồn tại ít nhất một chunk có điểm 2, ngược lại 0; báo cáo là trung bình theo case",
                "Cite@5": "1 nếu tồn tại ít nhất một chunk đủ dùng làm nguồn trích dẫn, ngược lại 0; báo cáo là trung bình theo case",
            },
            "case_metrics": all_case_metrics,
            "summary": summary_rows,
            "outputs": {
                "details_csv": str(DETAILS_CSV),
                "details_jsonl": str(DETAILS_JSONL),
                "summary_csv": str(SUMMARY_CSV),
                "summary_json": str(SUMMARY_JSON),
            },
        },
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for row in summary_rows:
        print(
            f"{row['Panel']}: "
            f"cases={row['Số case']} "
            f"Rel@5={row['Rel@5']} "
            f"Hit@5={row['Hit@5']} "
            f"Cite@5={row['Cite@5']}"
        )

    print("\nSaved:")
    print(f"- {DETAILS_CSV}")
    print(f"- {DETAILS_JSONL}")
    print(f"- {SUMMARY_CSV}")
    print(f"- {SUMMARY_JSON}")


if __name__ == "__main__":
    main()