from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests

try:
    from google import genai
except Exception:
    genai = None


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


EVALUATE_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVALUATE_DIR.parent
LAB_UNIFIED_DIR = ROOT_DIR / "rag_kg" / "lab_unified_rag"

load_dotenv(ROOT_DIR / ".env")
load_dotenv(LAB_UNIFIED_DIR / ".env")

DEFAULT_ANSWERS_JSONL = EVALUATE_DIR / "qa_100_eval_results.jsonl"
DEFAULT_CONTEXT_JSONL = EVALUATE_DIR / "context" / "question_context_100.jsonl"
DEFAULT_OUTPUT_JSONL = EVALUATE_DIR / "kg_llm_judge_results.jsonl"
DEFAULT_OUTPUT_CSV = EVALUATE_DIR / "kg_llm_judge_results.csv"
DEFAULT_FAILED_CSV = EVALUATE_DIR / "kg_llm_judge_failed.csv"

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 3
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
COLAB_MAX_NEW_TOKENS = int(os.getenv("JUDGE_MAX_NEW_TOKENS", "900"))
COLAB_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))


FIELDNAMES = [
    "id",
    "sample_type",
    "panel",
    "indicator",
    "type",
    "question",
    "answer",
    "model_answer",
    "judge_model",
    "judge_response_time_seconds",
    "node_score",
    "node_reason",
    "path_score",
    "path_reason",
    "evidence_score",
    "evidence_reason",
    "total_kg_3",
    "judge_summary",
    "status",
    "error",
    "attempts",
    "created_at",
    "human_node_score",
    "human_path_score",
    "human_evidence_score",
    "human_total_kg_3",
    "human_note",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_latest_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        qid = str(row.get("id") or row.get("question_id") or row.get("case_id") or "").strip()
        if qid:
            rows[qid] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def evidence_sources(evidence: list[dict[str, Any]]) -> list[str]:
    sources = []
    for item in evidence:
        source = str(item.get("source") or "").strip()
        page = item.get("page")
        if source and page not in [None, ""]:
            sources.append(f"{source}:page {page}")
        elif source:
            sources.append(source)
    return sources


def context_for_id(context_by_id: dict[str, dict[str, Any]], qid: str) -> dict[str, Any]:
    ctx = context_by_id.get(qid, {})
    return {
        "id": qid,
        "panel": ctx.get("panel") or "",
        "indicator": ctx.get("indicator") or "",
        "type": ctx.get("category") or ctx.get("type") or "",
        "question": ctx.get("question") or "",
        "tests": ctx.get("tests") or [],
        "book_evidence": ctx.get("book_evidence") or [],
        "evidence_ids": ctx.get("evidence_ids") or [],
        "expected_answer_vi": ctx.get("expected_answer_vi") or "",
        "answer_constraints_vi": ctx.get("answer_constraints_vi") or [],
    }


def format_evidence_block(evidence: list[dict[str, Any]], max_items: int) -> str:
    blocks = []
    for idx, item in enumerate(evidence[:max_items], start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] evidence_id: {item.get('evidence_id', '')}",
                    f"source: {item.get('source', '')}, page: {item.get('page', '')}",
                    f"tests: {', '.join(item.get('tests') or [])}",
                    f"conditions/topics: {', '.join(item.get('conditions') or item.get('topics') or [])}",
                    f"quote: {str(item.get('quote') or '').strip()}",
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "Không có evidence."


def build_judge_prompt(answer_row: dict[str, Any], ctx: dict[str, Any], max_evidence: int) -> str:
    evidence = ctx.get("book_evidence", [])
    expected_nodes = [f"Test:{test}" for test in ctx.get("tests", [])]
    expected_nodes += [f"Evidence:{item.get('evidence_id')}" for item in evidence[:max_evidence] if item.get("evidence_id")]

    expected_path = (
        f"Question:{ctx['id']} -> Test:{', '.join(ctx.get('tests', [])) or ctx.get('indicator', '')} "
        f"-> Evidence:{' -> '.join(ctx.get('evidence_ids', [])[:max_evidence])} -> Answer"
    )

    return f"""
Bạn là giám khảo đánh giá KG/RAG cho câu trả lời y khoa.

Chỉ chấm 3 tiêu chí KG, KHÔNG chấm văn phong chung.
Thang điểm mỗi tiêu chí chỉ được là 0, 0.5 hoặc 1.

Rubric:
1) node_score - Node đúng trọng tâm
- 0: Sai hoặc thiếu test/finding/question chính.
- 0.5: Đúng một phần.
- 1: Đúng test/finding/question trọng tâm.

2) path_score - Path reasoning hợp lý
- 0: Không nối được hoặc sai logic.
- 0.5: Có path nhưng thiếu/rời rạc.
- 1: Path đủ và logic.

3) evidence_score - Truy vết evidence/source
- 0: Không có evidence/source hoặc citation sai.
- 0.5: Có evidence nhưng yếu hoặc thiếu source/page rõ.
- 1: Có evidence + source/page rõ, dùng được.

Quy tắc bắt buộc:
- Trả về JSON object hợp lệ, không markdown.
- Mỗi reason phải nói rõ vì sao chấm 0.5 hoặc 0 nếu điểm không đạt 1.
- Nếu điểm là 1, reason vẫn phải ngắn gọn giải thích vì sao đạt.
- total_kg_3 = node_score + path_score + evidence_score.

Input:
- ID: {ctx['id']}
- Panel: {ctx['panel']}
- Indicator: {ctx['indicator']}
- Type: {ctx['type']}
- Question: {answer_row.get('question') or ctx.get('question')}

Expected KG nodes:
{json.dumps(expected_nodes, ensure_ascii=False)}

Expected reasoning path:
{expected_path}

Book evidence/source/page:
{format_evidence_block(evidence, max_evidence)}

Model answer to judge:
{answer_row.get('answer', '')}

Return JSON schema:
{{
  "node_score": 0 | 0.5 | 1,
  "node_reason": "...",
  "path_score": 0 | 0.5 | 1,
  "path_reason": "...",
  "evidence_score": 0 | 0.5 | 1,
  "evidence_reason": "...",
  "total_kg_3": 0,
  "judge_summary": "..."
}}
""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    raise ValueError(f"Cannot parse JSON object from judge output: {text[:500]}")


def normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        raise ValueError(f"Invalid score: {value}")

    allowed = {0.0, 0.5, 1.0}
    if score not in allowed:
        raise ValueError(f"Score must be one of 0, 0.5, 1. Got: {value}")

    return score


def normalize_judge_obj(obj: dict[str, Any]) -> dict[str, Any]:
    node_score = normalize_score(obj.get("node_score"))
    path_score = normalize_score(obj.get("path_score"))
    evidence_score = normalize_score(obj.get("evidence_score"))
    total = round(node_score + path_score + evidence_score, 2)

    return {
        "node_score": node_score,
        "node_reason": str(obj.get("node_reason") or "").strip(),
        "path_score": path_score,
        "path_reason": str(obj.get("path_reason") or "").strip(),
        "evidence_score": evidence_score,
        "evidence_reason": str(obj.get("evidence_reason") or "").strip(),
        "total_kg_3": total,
        "judge_summary": str(obj.get("judge_summary") or "").strip(),
    }


def extract_final_answer(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    for marker in ["FINAL ANSWER:", "Final answer:", "ANSWER:", "Answer:"]:
        if marker in text:
            text = text.split(marker, 1)[1].strip()
    return text


def call_judge_llm(prompt: str) -> tuple[str, str, float]:
    errors = []
    providers = [
        (f"local_ollama/{os.getenv('LOCAL_LLM_MODEL', 'qwen2.5:3b').strip()}", call_local_ollama),
        (os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(), call_gemini),
        (f"openrouter/{os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-chat').strip()}", call_openrouter),
        ("colab_llm", call_colab),
    ]

    for model, func in providers:
        try:
            start = time.perf_counter()
            text = func(prompt)
            elapsed = time.perf_counter() - start
            return text, model, round(elapsed, 3)
        except Exception as exc:
            errors.append(f"{model} failed: {exc}")

    raise RuntimeError("All judge LLM providers failed:\n" + "\n".join(errors))


def call_local_ollama(prompt: str) -> str:
    local_url = os.getenv("LOCAL_LLM_URL", "").strip()
    model_name = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b").strip()
    if not local_url:
        raise ValueError("Missing LOCAL_LLM_URL")

    response = requests.post(
        local_url,
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": COLAB_TEMPERATURE,
                "num_predict": COLAB_MAX_NEW_TOKENS,
            },
        },
        headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict):
        return str(
            data.get("response")
            or data.get("text")
            or data.get("output")
            or data.get("generated_text")
            or data.get("answer")
            or ""
        ).strip()
    return str(data).strip()


def call_gemini(prompt: str) -> str:
    if genai is None:
        raise RuntimeError("google-genai is not installed")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(response, "text", "") or ""
    if not text.strip():
        raise ValueError("Gemini returned empty response")
    return text.strip()


def call_openrouter(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model_name = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat").strip()
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": COLAB_TEMPERATURE,
            "max_tokens": COLAB_MAX_NEW_TOKENS,
            "response_format": {"type": "json_object"},
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise ValueError(f"Unexpected OpenRouter response: {data}") from exc

    if not text.strip():
        raise ValueError("OpenRouter returned empty response")
    return text.strip()


def normalize_colab_url(url: str) -> str:
    url = str(url or "").strip().rstrip("/")
    if url and not url.endswith("/generate"):
        url += "/generate"
    return url


def call_colab(prompt: str) -> str:
    colab_url = normalize_colab_url(os.getenv("COLAB_LLM_URL", ""))
    if not colab_url:
        raise ValueError("Missing COLAB_LLM_URL")

    response = requests.post(
        colab_url,
        json={
            "prompt": prompt,
            "max_new_tokens": COLAB_MAX_NEW_TOKENS,
            "temperature": COLAB_TEMPERATURE,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict):
        return str(
            data.get("response")
            or data.get("text")
            or data.get("output")
            or data.get("generated_text")
            or data.get("answer")
            or ""
        ).strip()
    return str(data).strip()


def judge_with_retry(prompt: str) -> dict[str, Any]:
    last_error = ""
    total_elapsed = 0.0
    judge_model = "unknown"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text, judge_model, elapsed = call_judge_llm(prompt)
            total_elapsed += elapsed
            obj = normalize_judge_obj(extract_json_object(text))

            return {
                **obj,
                "judge_model": judge_model,
                "judge_response_time_seconds": round(total_elapsed, 3),
                "status": "OK",
                "error": "",
                "attempts": attempt,
            }
        except Exception as exc:
            last_error = str(exc)
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {last_error}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    return {
        "node_score": "",
        "node_reason": "",
        "path_score": "",
        "path_reason": "",
        "evidence_score": "",
        "evidence_reason": "",
        "total_kg_3": "",
        "judge_summary": "",
        "judge_model": judge_model,
        "judge_response_time_seconds": round(total_elapsed, 3),
        "status": "ERROR",
        "error": last_error,
        "attempts": MAX_RETRIES,
    }


def should_run(qid: str, old_results: dict[str, dict[str, Any]], mode: str) -> bool:
    old = old_results.get(qid)
    if mode == "all":
        return True
    if mode == "resume":
        return old is None
    if mode == "failed":
        if old is None:
            return True
        return old.get("status") != "OK" or bool(old.get("error"))
    return False


def make_row(answer_row: dict[str, Any], ctx: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": answer_row.get("id") or ctx.get("id"),
        "sample_type": "qa_eval",
        "panel": answer_row.get("panel") or ctx.get("panel"),
        "indicator": answer_row.get("indicator") or ctx.get("indicator"),
        "type": answer_row.get("type") or ctx.get("type"),
        "question": answer_row.get("question") or ctx.get("question"),
        "answer": answer_row.get("answer", ""),
        "model_answer": answer_row.get("model", ""),
        "judge_model": judge.get("judge_model", ""),
        "judge_response_time_seconds": judge.get("judge_response_time_seconds", ""),
        "node_score": judge.get("node_score", ""),
        "node_reason": judge.get("node_reason", ""),
        "path_score": judge.get("path_score", ""),
        "path_reason": judge.get("path_reason", ""),
        "evidence_score": judge.get("evidence_score", ""),
        "evidence_reason": judge.get("evidence_reason", ""),
        "total_kg_3": judge.get("total_kg_3", ""),
        "judge_summary": judge.get("judge_summary", ""),
        "status": judge.get("status", "ERROR"),
        "error": judge.get("error", ""),
        "attempts": judge.get("attempts", 0),
        "created_at": now_iso(),
        "human_node_score": "",
        "human_path_score": "",
        "human_evidence_score": "",
        "human_total_kg_3": "",
        "human_note": "",
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row.get("id", ""))

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in FIELDNAMES})


def export_csv(jsonl_path: Path, csv_path: Path, failed_csv_path: Path) -> None:
    rows = list(load_latest_by_id(jsonl_path).values())
    write_csv(rows, csv_path)
    write_csv([row for row in rows if row.get("status") != "OK" or row.get("error")], failed_csv_path)
    print(f"Exported CSV: {csv_path}")
    print(f"Exported failed CSV: {failed_csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge QA answers with KG rubric 0/0.5/1.")
    parser.add_argument(
        "--mode",
        choices=["resume", "failed", "all", "export"],
        default="resume",
        help="resume: skip judged rows; failed: rerun failed judge rows; all: append fresh judge rows; export: CSV only",
    )
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS_JSONL)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT_JSONL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--failed-csv", type=Path, default=DEFAULT_FAILED_CSV)
    parser.add_argument("--panel", choices=["all", "cbc", "biochem"], default="all")
    parser.add_argument("--max-evidence", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "export":
        export_csv(args.output, args.csv, args.failed_csv)
        return

    answers = list(load_latest_by_id(args.answers).values())
    context_by_id = load_latest_by_id(args.context)
    old_results = load_latest_by_id(args.output)

    answers = [row for row in answers if row.get("status") == "OK" and row.get("answer")]

    if args.panel != "all":
        panel = "CBC" if args.panel == "cbc" else "BIOCHEM"
        answers = [row for row in answers if str(row.get("panel", "")).upper() == panel]

    print(f"Answers to judge: {len(answers)}")
    print(f"Loaded previous judge rows: {len(old_results)}")
    print(f"Mode: {args.mode}")
    print(f"Answers JSONL: {args.answers}")
    print(f"Context JSONL: {args.context}")
    print(f"Output JSONL: {args.output}")
    print("-" * 80)

    for answer_row in answers:
        qid = str(answer_row.get("id") or "").strip()
        if not qid:
            continue

        if not should_run(qid, old_results, args.mode):
            print(f"Skip {qid}: already judged")
            continue

        ctx = context_for_id(context_by_id, qid)
        print(f"Judge {qid}: {answer_row.get('question') or ctx.get('question')}")

        prompt = build_judge_prompt(answer_row, ctx, max_evidence=args.max_evidence)
        judge = judge_with_retry(prompt)
        row = make_row(answer_row, ctx, judge)

        append_jsonl(args.output, row)
        old_results[qid] = row

        if row["status"] == "OK":
            print(
                f"  OK | total={row['total_kg_3']}/3 | "
                f"node={row['node_score']} path={row['path_score']} evidence={row['evidence_score']}"
            )
        else:
            print(f"  ERROR | {row['error']}")

        time.sleep(args.sleep)

    export_csv(args.output, args.csv, args.failed_csv)
    print("Done.")


if __name__ == "__main__":
    main()
