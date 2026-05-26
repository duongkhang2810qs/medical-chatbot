from __future__ import annotations

from pathlib import Path

from config import OUTPUT_DIR
from lab_core import (
    append_jsonl,
    build_reasoning_context,
    generate_clean_answer,
)
from run_final import load_cases_for_final


OUT_DIR = OUTPUT_DIR / "ablation"
OUTPUT_PATH = OUT_DIR / "final_output_llm_only.jsonl"
FAILED_PATH = OUT_DIR / "failed_cases_llm_only.jsonl"


def load_processed_case_ids(path: Path) -> set[str]:
    processed = set()

    if not path.exists():
        return processed

    from lab_core import load_jsonl

    for item in load_jsonl(path):
        case_id = item.get("case_id")
        if case_id:
            processed.add(str(case_id))

    return processed


def build_llm_only_prompt(ctx: dict) -> str:
    abnormal_items = ctx.get("abnormal_items", [])

    lines = []

    for item in abnormal_items:
        lines.append(
            f"- {item.get('panel')} | {item.get('test_label') or item.get('test')}: "
            f"{item.get('value')} {item.get('unit', '')} | "
            f"{item.get('status')} | "
            f"tham chiếu: {item.get('reference_range', '')}"
        )

    abnormal_text = "\n".join(lines) if lines else "Không có chỉ số bất thường rõ ràng."

    return f"""
Bạn là trợ lý hỗ trợ diễn giải kết quả xét nghiệm cho người dùng phổ thông.

DỮ LIỆU XÉT NGHIỆM BẤT THƯỜNG:
{abnormal_text}

Yêu cầu:
- Diễn giải ý nghĩa các chỉ số bất thường.
- Nêu các khả năng liên quan nhưng không chẩn đoán chắc chắn.
- Nếu có chỉ số nguy hiểm, cần cảnh báo phù hợp.
- Khuyến nghị người dùng trao đổi với bác sĩ khi cần.
- Viết bằng tiếng Việt, rõ ràng, dễ hiểu.
- Không tạo citation hoặc references vì không có tài liệu truy xuất.

Format trả lời:
### 1. Tóm tắt bất thường
### 2. Ý nghĩa lâm sàng
### 3. Lưu ý an toàn
### 4. Nên làm gì tiếp theo
### 5. Hạn chế

Lưu ý: Nội dung chỉ nhằm hỗ trợ tham khảo, không thay thế chẩn đoán hoặc điều trị của bác sĩ.
""".strip()


def process_one_case(case: dict, idx: int) -> dict:
    ctx = build_reasoning_context(case, idx)
    case_id = ctx["case_id"]

    print("\n" + "=" * 80)
    print(f"LLM ONLY - Processing case: {case_id}")
    print(f"Panels: {ctx.get('panels')}")
    print(f"Abnormal tests: {ctx.get('abnormal_tests')}")

    prompt = build_llm_only_prompt(ctx)
    answer, meta = generate_clean_answer(prompt)

    elapsed = meta.get(
        "total_elapsed_seconds",
        meta.get("elapsed_seconds", 0.0),
    )

    return {
        "case_id": case_id,
        "mode": "llm_only",
        "panel": ctx.get("panels", [""])[0] if ctx.get("panels") else "",
        "model_used": meta.get("model_used", "unknown"),
        "elapsed_seconds": elapsed,
        "answer": answer,
    }


def main(reset: bool = False):
    print("=" * 80)
    print("RUN LLM ONLY BASELINE")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if reset:
        for path in [OUTPUT_PATH, FAILED_PATH]:
            if path.exists():
                path.unlink()
                print(f"Removed old file: {path}")

    cases = load_cases_for_final()
    processed = load_processed_case_ids(OUTPUT_PATH)

    print(f"Total cases: {len(cases)}")
    print(f"Already processed: {len(processed)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Failed: {FAILED_PATH}")

    for idx, case in enumerate(cases):
        ctx = build_reasoning_context(case, idx)
        case_id = ctx["case_id"]

        if case_id in processed:
            print(f"Skip processed case: {case_id}")
            continue

        try:
            result = process_one_case(case, idx)
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
    print(f"Output: {OUTPUT_PATH}")
    print(f"Failed: {FAILED_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xóa output LLM only cũ rồi chạy lại từ đầu.",
    )

    args = parser.parse_args()

    main(reset=args.reset)