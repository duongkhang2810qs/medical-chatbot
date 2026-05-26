from __future__ import annotations

import argparse
import csv
import json
import os
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

DEFAULT_CONTEXT_PATH = EVALUATE_DIR / "context" / "question_context_100.jsonl"
DEFAULT_OUTPUT_JSONL = EVALUATE_DIR / "qa_100_eval_results.jsonl"
DEFAULT_OUTPUT_CSV = EVALUATE_DIR / "qa_100_eval_results.csv"
DEFAULT_FAILED_CSV = EVALUATE_DIR / "qa_100_eval_failed.csv"
DEFAULT_REVIEW_CSV = EVALUATE_DIR / "kg_human_review_template.csv"
DEFAULT_PUBLIC_JSONL = EVALUATE_DIR / "qa_100_eval_results_public.jsonl"

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 3
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
COLAB_MAX_NEW_TOKENS = int(os.getenv("COLAB_MAX_NEW_TOKENS", "970"))
COLAB_TEMPERATURE = float(os.getenv("COLAB_TEMPERATURE", "0.2"))
PROVIDER_ORDER = [
    item.strip().lower()
    for item in os.getenv("EVAL_PROVIDER_ORDER", "local,gemini,openrouter,colab").split(",")
    if item.strip()
]


FIELDNAMES = [
    "id",
    "panel",
    "indicator",
    "type",
    "question",
    "answer",
    "model",
    "response_time_seconds",
    "status",
    "error",
    "attempts",
    "created_at",
]


REVIEW_FIELDNAMES = [
    "ID",
    "Loại mẫu",
    "Input / Câu hỏi",
    "Model answer",
    "Evidence source/page",
    "Evidence text",
    "Node đúng trong tâm",
    "Path reasoning hợp lý",
    "Truy vết evidence/source",
    "Tổng KG /3",
    "Ghi chú",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    rows: dict[str, dict[str, Any]] = {}

    for row in load_jsonl(path):
        qid = str(row.get("id") or "").strip()
        if qid:
            rows[qid] = row

    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_context_row(row: dict[str, Any]) -> dict[str, Any]:
    qid = str(row.get("question_id") or row.get("case_id") or row.get("id") or "")
    category = str(row.get("category") or row.get("type") or "")

    return {
        "id": qid,
        "panel": str(row.get("panel") or ""),
        "indicator": str(row.get("indicator") or ""),
        "type": category,
        "question": str(row.get("question") or ""),
        "record_type": str(row.get("record_type") or "qa_eval"),
        "tests": row.get("tests") or [],
        "expected_answer_vi": row.get("expected_answer_vi") or "",
        "book_evidence": row.get("book_evidence") or [],
        "evidence_ids": row.get("evidence_ids") or [],
        "answer_constraints_vi": row.get("answer_constraints_vi") or [],
        "citation_instruction_vi": row.get("citation_instruction_vi") or "",
        "context_version": row.get("context_version") or "",
        "raw_context": row,
    }


def load_context_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing context file: {path}. Copy/build question_context_100.jsonl first."
        )

    rows = [normalize_context_row(row) for row in load_jsonl(path)]
    rows = [row for row in rows if row["id"] and row["question"]]

    if not rows:
        raise RuntimeError(f"No valid context rows found in {path}")

    return rows


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


def evidence_texts(evidence: list[dict[str, Any]], max_items: int) -> list[str]:
    texts = []

    for idx, item in enumerate(evidence[:max_items], start=1):
        evidence_id = str(item.get("evidence_id") or "").strip()
        quote = " ".join(str(item.get("quote") or "").split())

        if quote:
            texts.append(f"[{idx}] {evidence_id}: {quote}")

    return texts


def kg_nodes(row: dict[str, Any]) -> list[str]:
    nodes = [f"Test:{test}" for test in row.get("tests", [])]

    for item in row.get("book_evidence", []):
        evidence_id = item.get("evidence_id")
        if evidence_id:
            nodes.append(f"Evidence:{evidence_id}")

    return nodes


def reasoning_path(row: dict[str, Any]) -> str:
    tests = ", ".join(row.get("tests", [])) or row.get("indicator", "")
    evidence = " -> ".join(row.get("evidence_ids", []))

    if evidence:
        return f"Question:{row['id']} -> Test:{tests} -> Evidence:{evidence} -> Answer"

    return f"Question:{row['id']} -> Test:{tests} -> Answer"


def format_evidence_block(evidence: list[dict[str, Any]], max_items: int) -> str:
    blocks = []

    for idx, item in enumerate(evidence[:max_items], start=1):
        quote = str(item.get("quote") or "").strip()
        tests = ", ".join(item.get("tests") or [])
        conditions = ", ".join(item.get("conditions") or [])
        source = item.get("source") or ""
        page = item.get("page") or ""
        evidence_id = item.get("evidence_id") or ""

        blocks.append(
            "\n".join(
                [
                    f"[{idx}] evidence_id: {evidence_id}",
                    f"source: {source}, page: {page}",
                    f"tests: {tests}",
                    f"conditions/topics: {conditions}",
                    f"quote: {quote}",
                ]
            )
        )

    return "\n\n".join(blocks) if blocks else "Không có evidence sách phù hợp."


def short_quote(text: str, max_len: int = 360) -> str:
    quote = " ".join(str(text or "").split())

    if len(quote) <= max_len:
        return quote

    return quote[:max_len].rstrip() + " [...]"


def build_references_block(evidence: list[dict[str, Any]], max_items: int = 3) -> str:
    lines = []

    for idx, item in enumerate(evidence[:max_items], start=1):
        source = str(item.get("source") or "").strip()
        page = item.get("page")
        quote = short_quote(str(item.get("quote") or ""))

        if not source and not quote:
            continue

        source_part = source
        if page not in [None, ""]:
            source_part = f"{source}, page {page}"

        if quote:
            lines.append(f"[{idx}] {source_part}. \"{quote}\"")
        else:
            lines.append(f"[{idx}] {source_part}.")

    if not lines:
        return ""

    return "References:\n" + "\n".join(lines)


def build_prompt(row: dict[str, Any], max_evidence: int) -> str:
    evidence_block = format_evidence_block(row["book_evidence"], max_evidence)
    constraints = "\n".join(f"- {item}" for item in row.get("answer_constraints_vi", []))
    if not constraints:
        constraints = (
            "- Trả lời tiếng Việt, ngắn gọn, dễ hiểu.\n"
            "- Không chẩn đoán chắc chắn.\n"
            "- Không kê thuốc."
        )

    return f"""
Bạn là trợ lý y khoa đang sinh câu trả lời cho bộ đánh giá QA.

Nhiệm vụ:
- Chỉ tạo ANSWER cho người dùng, không tự chấm điểm.
- Phải dựa trên KG/book context bên dưới.
- Bắt buộc dùng ít nhất một citation [1], [2] hoặc [3] vì EVIDENCE đã được cung cấp.
- Dùng citation [1], [2], [3] đúng theo thứ tự EVIDENCE ngay sau mệnh đề được evidence hỗ trợ.
- Không bịa source, không bịa số trang, không dùng citation ngoài EVIDENCE.
- Nếu context không đủ trực tiếp, nói rõ "Bằng chứng truy xuất còn hạn chế" nhưng vẫn trả lời an toàn ở mức tổng quát.
- Không tự viết phần "References"; hệ thống sẽ tự gắn References từ evidence sau khi model trả lời.

Metadata:
- ID: {row['id']}
- Loại mẫu: {row['record_type']}
- Panel: {row['panel']}
- Indicator: {row['indicator']}
- Type: {row['type']}
- KG nodes đúng trọng tâm dự kiến: {", ".join(kg_nodes(row))}
- Reasoning path dự kiến: {reasoning_path(row)}

Câu hỏi:
{row['question']}

KG / BOOK EVIDENCE:
{evidence_block}

Ràng buộc trả lời:
{constraints}

Chỉ trả về nội dung câu trả lời cuối cùng bằng tiếng Việt.
""".strip()


def build_compact_prompt(row: dict[str, Any], max_evidence: int) -> str:
    evidence_block = format_evidence_block(row["book_evidence"], max_evidence)

    return f"""
Trả lời ngắn gọn bằng tiếng Việt cho câu hỏi xét nghiệm máu.
Dựa vào EVIDENCE, không bịa thông tin, không chẩn đoán chắc chắn, không kê thuốc.
Bắt buộc dùng citation [1] nếu dùng evidence.
Không tự viết References.

ID: {row['id']}
Panel: {row['panel']}
Chỉ số: {row['indicator']}
Câu hỏi: {row['question']}

EVIDENCE:
{evidence_block}

Answer:
""".strip()


def build_minimal_prompt(row: dict[str, Any]) -> str:
    evidence = row.get("book_evidence", [])
    first = evidence[0] if evidence else {}
    source = str(first.get("source") or "").strip()
    page = first.get("page") or ""
    quote = short_quote(str(first.get("quote") or ""), max_len=420)

    return f"""
Trả lời ngắn gọn bằng tiếng Việt.
Không chẩn đoán chắc chắn, không kê thuốc.
Dựa vào evidence và bắt buộc cite [1].

Câu hỏi: {row['question']}

[1] {source}, page {page}: "{quote}"

Answer:
""".strip()


def has_citation(text: str) -> bool:
    return any(f"[{idx}]" in str(text or "") for idx in range(1, 10))


def cited_indexes(text: str) -> list[int]:
    found = []
    text = str(text or "")

    for idx in range(1, 10):
        if f"[{idx}]" in text:
            found.append(idx)

    return found


def remap_citations_contiguous(answer: str) -> tuple[str, dict[int, int]]:
    answer = str(answer or "")
    old_indexes = cited_indexes(answer)
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(old_indexes, start=1)}

    for old_idx, new_idx in sorted(mapping.items(), reverse=True):
        answer = answer.replace(f"[{old_idx}]", f"[{new_idx}]")

    return answer, mapping


def ensure_answer_citation(answer: str, context_row: dict[str, Any]) -> str:
    answer = str(answer or "").strip()

    if not answer or has_citation(answer):
        return answer

    if context_row.get("book_evidence"):
        return answer.rstrip(".。 ") + " [1]."

    return answer


def select_cited_evidence(
    evidence: list[dict[str, Any]],
    citation_mapping: dict[int, int],
) -> list[tuple[int, dict[str, Any]]]:
    selected = []

    for old_idx, new_idx in citation_mapping.items():
        evidence_pos = old_idx - 1

        if 0 <= evidence_pos < len(evidence):
            selected.append((new_idx, evidence[evidence_pos]))

    return selected


def build_references_block_for_citations(
    evidence: list[dict[str, Any]],
    citation_mapping: dict[int, int],
) -> str:
    lines = []

    for citation_idx, item in select_cited_evidence(evidence, citation_mapping):
        source = str(item.get("source") or "").strip()
        page = item.get("page")
        quote = short_quote(str(item.get("quote") or ""))

        if not source and not quote:
            continue

        source_part = source
        if page not in [None, ""]:
            source_part = f"{source}, page {page}"

        if quote:
            lines.append(f"[{citation_idx}] {source_part}. \"{quote}\"")
        else:
            lines.append(f"[{citation_idx}] {source_part}.")

    if not lines:
        return ""

    return "References:\n" + "\n".join(lines)


def strip_references_block(answer: str) -> str:
    answer = str(answer or "").strip()
    marker = "\n\nReferences:"

    if marker in answer:
        return answer.split(marker, 1)[0].strip()

    marker = "\nReferences:"
    if marker in answer:
        return answer.split(marker, 1)[0].strip()

    return answer


def attach_references(answer: str, context_row: dict[str, Any], max_evidence: int = 3) -> str:
    del max_evidence

    answer = strip_references_block(answer)
    answer = ensure_answer_citation(answer, context_row)
    answer, citation_mapping = remap_citations_contiguous(answer)
    references = build_references_block_for_citations(
        context_row.get("book_evidence", []),
        citation_mapping,
    )

    if not references:
        return answer

    return f"{answer.strip()}\n\n{references}"


def extract_final_answer(raw_text: str) -> str:
    text = str(raw_text or "").strip()

    for marker in ["FINAL ANSWER:", "Final answer:", "ANSWER:", "Answer:"]:
        if marker in text:
            text = text.split(marker, 1)[1].strip()

    return text


def call_context_llm(prompt: str) -> tuple[str, str, float]:
    errors = []
    providers = {
        "local": (f"local_ollama/{os.getenv('LOCAL_LLM_MODEL', 'qwen2.5:3b').strip()}", call_local_ollama),
        "gemini": (os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(), call_gemini),
        "openrouter": (f"openrouter/{os.getenv('OPENROUTER_MODEL', 'deepseek-chat').strip()}", call_openrouter),
        "colab": ("colab_llm", call_colab),
    }

    for provider_name in PROVIDER_ORDER:
        if provider_name not in providers:
            errors.append(f"Unknown provider skipped: {provider_name}")
            continue

        model, func = providers[provider_name]

        try:
            start = time.perf_counter()
            answer = func(prompt)
            elapsed = time.perf_counter() - start

            return answer, model, round(elapsed, 3)

        except Exception as exc:
            errors.append(f"{model} failed: {exc}")

    raise RuntimeError("All LLM providers failed:\n" + "\n".join(errors))


def call_local_ollama(prompt: str) -> str:
    local_url = normalize_local_ollama_url(os.getenv("LOCAL_LLM_URL", ""))
    model_name = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:3b").strip()

    if not local_url:
        raise ValueError("Missing LOCAL_LLM_URL")

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": COLAB_TEMPERATURE,
            "num_predict": COLAB_MAX_NEW_TOKENS,
        },
    }

    response = requests.post(
        local_url,
        json=payload,
        headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"},
        timeout=REQUEST_TIMEOUT,
    )
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

    text = extract_final_answer(text)
    if not text:
        raise ValueError("Local Ollama returned empty response")

    return text


def normalize_local_ollama_url(url: str) -> str:
    url = str(url or "").strip().replace(" ", "").rstrip("/")

    if not url:
        return ""

    while url.endswith("/api/generate/api/generate"):
        url = url[: -len("/api/generate")]

    if not url.endswith("/api/generate"):
        url += "/api/generate"

    return url


def call_gemini(prompt: str) -> str:
    if genai is None:
        raise RuntimeError("google-genai is not installed")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = extract_final_answer(getattr(response, "text", "") or "")

    if not text:
        raise ValueError("Gemini returned empty response")

    return text


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
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise ValueError(f"Unexpected OpenRouter response: {data}") from exc

    text = extract_final_answer(text)
    if not text:
        raise ValueError("OpenRouter returned empty response")

    return text


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

    text = extract_final_answer(text)
    if not text:
        raise ValueError("Colab returned empty response")

    return text


def call_context_llm_with_retry(prompt: str) -> dict[str, Any]:
    last_error = ""
    total_elapsed = 0.0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            answer, model, elapsed = call_context_llm(prompt)
            total_elapsed += elapsed

            if not answer:
                raise RuntimeError("Empty answer from LLM")

            return {
                "status": "OK",
                "answer": answer,
                "model": model,
                "response_time_seconds": round(total_elapsed, 3),
                "error": "",
                "attempts": attempt,
            }

        except Exception as exc:
            last_error = str(exc)
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {last_error}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    return {
        "status": "ERROR",
        "answer": "",
        "model": "unknown",
        "response_time_seconds": round(total_elapsed, 3),
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


def make_row(context_row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = result.get("answer", "")
    if result.get("status") == "OK":
        answer = attach_references(answer, context_row)

    return {
        "id": context_row["id"],
        "panel": context_row["panel"],
        "indicator": context_row["indicator"],
        "type": context_row["type"],
        "question": context_row["question"],
        "answer": answer,
        "model": result.get("model", ""),
        "response_time_seconds": result.get("response_time_seconds", 0),
        "status": result.get("status", "ERROR"),
        "error": result.get("error", ""),
        "attempts": result.get("attempts", 0),
        "created_at": now_iso(),
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row.get("id", ""))

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in FIELDNAMES})


def write_public_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row.get("id", ""))

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            public_row = {key: row.get(key, "") for key in FIELDNAMES}
            f.write(json.dumps(public_row, ensure_ascii=False) + "\n")


def write_checkpoint_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row.get("id", ""))

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_context_by_id(context_path: Path) -> dict[str, dict[str, Any]]:
    if not context_path.exists():
        return {}

    return {row["id"]: row for row in load_context_rows(context_path)}


def write_review_template(
    rows: list[dict[str, Any]],
    path: Path,
    context_by_id: dict[str, dict[str, Any]],
    max_evidence: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row.get("id", ""))

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()

        for row in rows:
            ctx = context_by_id.get(str(row.get("id") or ""), {})
            evidence = ctx.get("book_evidence", [])

            writer.writerow(
                {
                    "ID": row.get("id", ""),
                    "Loại mẫu": "qa_eval",
                    "Input / Câu hỏi": row.get("question", ""),
                    "Model answer": row.get("answer", ""),
                    "Evidence source/page": " | ".join(evidence_sources(evidence[:max_evidence])),
                    "Evidence text": " || ".join(evidence_texts(evidence, max_evidence)),
                    "Node đúng trong tâm": "",
                    "Path reasoning hợp lý": "",
                    "Truy vết evidence/source": "",
                    "Tổng KG /3": "",
                    "Ghi chú": "",
                }
            )


def enrich_rows_for_public_output(
    rows: list[dict[str, Any]],
    context_by_id: dict[str, dict[str, Any]],
    max_evidence: int,
) -> list[dict[str, Any]]:
    enriched = []

    for row in rows:
        out = dict(row)
        ctx = context_by_id.get(str(row.get("id") or ""), {})

        if out.get("status") == "OK" and out.get("answer") and ctx:
            out["answer"] = attach_references(out["answer"], ctx, max_evidence)

        enriched.append(out)

    return enriched


def export_files(
    jsonl_path: Path,
    csv_path: Path,
    failed_csv_path: Path,
    review_csv_path: Path,
    public_jsonl_path: Path,
    context_path: Path,
    max_evidence: int,
) -> None:
    checkpoint = load_checkpoint(jsonl_path)
    rows = list(checkpoint.values())
    context_by_id = build_context_by_id(context_path)
    public_rows = enrich_rows_for_public_output(rows, context_by_id, max_evidence)

    write_checkpoint_jsonl(public_rows, jsonl_path)
    write_public_jsonl(public_rows, public_jsonl_path)
    write_csv(public_rows, csv_path)
    write_csv(
        [row for row in public_rows if row.get("status") != "OK" or row.get("error")],
        failed_csv_path,
    )
    write_review_template(public_rows, review_csv_path, context_by_id, max_evidence)

    print(f"Exported CSV: {csv_path}")
    print(f"Exported failed CSV: {failed_csv_path}")
    print(f"Exported public JSONL: {public_jsonl_path}")
    print(f"Exported KG review template: {review_csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate QA answers from lab_unified KG/book context."
    )
    parser.add_argument(
        "--mode",
        choices=["resume", "failed", "all", "export"],
        default="resume",
        help="resume: skip done rows; failed: rerun failed rows; all: append a fresh run; export: CSV only",
    )
    parser.add_argument(
        "--panel",
        choices=["all", "cbc", "biochem"],
        default="all",
        help="Context subset to run.",
    )
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--failed-csv", type=Path, default=DEFAULT_FAILED_CSV)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--public-jsonl", type=Path, default=DEFAULT_PUBLIC_JSONL)
    parser.add_argument("--max-evidence", type=int, default=3)
    parser.add_argument(
        "--compact-prompt",
        action="store_true",
        help="Use a shorter prompt for unstable self-hosted models.",
    )
    parser.add_argument(
        "--minimal-prompt",
        action="store_true",
        help="Use the shortest prompt: question + first evidence quote only.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=[],
        help="Run only specific IDs, for example: --ids CBC001 BIO002",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N rows after filtering. Useful for smoke test.",
    )
    parser.add_argument("--sleep", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "export":
        export_files(
            args.output,
            args.csv,
            args.failed_csv,
            args.review_csv,
            args.public_jsonl,
            args.context,
            args.max_evidence,
        )
        return

    context_rows = load_context_rows(args.context)

    if args.panel != "all":
        panel = "CBC" if args.panel == "cbc" else "BIOCHEM"
        context_rows = [row for row in context_rows if row["panel"].upper() == panel]

    if args.ids:
        wanted_ids = {item.strip().upper() for item in args.ids}
        context_rows = [row for row in context_rows if row["id"].upper() in wanted_ids]

    if args.limit > 0:
        context_rows = context_rows[: args.limit]

    old_results = load_checkpoint(args.output)

    print(f"Context rows: {len(context_rows)}")
    print(f"Loaded checkpoint rows: {len(old_results)}")
    print(f"Mode: {args.mode}")
    print(f"Context: {args.context}")
    print(f"Output JSONL: {args.output}")
    print("-" * 80)

    for context_row in context_rows:
        qid = context_row["id"]

        if not should_run(qid, old_results, args.mode):
            print(f"Skip {qid}: already done")
            continue

        print(f"Run {qid} [{context_row['panel']}]: {context_row['question']}")

        if args.minimal_prompt:
            prompt = build_minimal_prompt(context_row)
        elif args.compact_prompt:
            prompt = build_compact_prompt(context_row, max_evidence=args.max_evidence)
        else:
            prompt = build_prompt(context_row, max_evidence=args.max_evidence)
        result = call_context_llm_with_retry(prompt)
        row = make_row(context_row, result)

        append_jsonl(args.output, row)
        old_results[qid] = row

        if row["status"] == "OK":
            print(
                f"  OK | model={row['model']} | "
                f"time={row['response_time_seconds']}s | "
                f"evidence={len(context_row['book_evidence'])}"
            )
        else:
            print(f"  ERROR | {row['error']}")

        time.sleep(args.sleep)

    export_files(
        args.output,
        args.csv,
        args.failed_csv,
        args.review_csv,
        args.public_jsonl,
        args.context,
        args.max_evidence,
    )
    print("Done.")


if __name__ == "__main__":
    main()
# run all: python evaluate/run_qa_eval.py --mode all
# run case failed: python evaluate/run_qa_eval.py --mode failed
