import requests
import csv
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

API_URL = "http://127.0.0.1:8002/api/v1/chat"
SESSION_ID = "eval_cbc_001"

CHECKPOINT_FILE = Path("cbc_50_eval_checkpoint.jsonl")
CSV_FILE = Path("cbc_50_eval_results.csv")
FAILED_CSV_FILE = Path("cbc_50_eval_failed.csv")

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 3
REQUEST_TIMEOUT = 120


QUESTIONS = [
    ("CBC001", "RBC", "Khái niệm", "RBC là gì trong xét nghiệm công thức máu?"),
    ("CBC002", "HGB", "Khái niệm", "HGB là gì và dùng để đánh giá điều gì?"),
    ("CBC003", "HCT", "Khái niệm", "HCT là gì trong xét nghiệm máu?"),
    ("CBC004", "MCV", "Khái niệm", "MCV là gì và phản ánh đặc điểm nào của hồng cầu?"),
    ("CBC005", "MCH", "Khái niệm", "MCH là gì trong công thức máu?"),
    ("CBC006", "MCHC", "Khái niệm", "MCHC là gì và có ý nghĩa gì?"),
    ("CBC007", "RDW", "Khái niệm", "RDW là gì trong xét nghiệm máu?"),
    ("CBC008", "WBC", "Khái niệm", "WBC là gì trong công thức máu?"),
    ("CBC009", "NEUT", "Khái niệm", "NEUT là gì trong xét nghiệm máu?"),
    ("CBC010", "LYMPH", "Khái niệm", "LYMPH/LYM là gì trong công thức máu?"),
    ("CBC011", "MONO", "Khái niệm", "MONO là gì trong xét nghiệm máu?"),
    ("CBC012", "EOS", "Khái niệm", "EOS là gì trong công thức máu?"),
    ("CBC013", "BASO", "Khái niệm", "BASO là gì trong xét nghiệm máu?"),
    ("CBC014", "PLT", "Khái niệm", "PLT là gì trong xét nghiệm máu?"),
    ("CBC015", "IG", "Khái niệm", "IG là gì trong công thức máu?"),

    ("CBC016", "RBC", "Tăng/giảm", "RBC thấp có thể gợi ý vấn đề gì?"),
    ("CBC017", "RBC", "Tăng/giảm", "RBC cao có ý nghĩa gì?"),
    ("CBC018", "HGB", "Tăng/giảm", "HGB thấp có phải là thiếu máu không?"),
    ("CBC019", "HGB", "Tăng/giảm", "HGB cao có thể liên quan đến tình trạng nào?"),
    ("CBC020", "HCT", "Tăng/giảm", "HCT thấp thường liên quan đến vấn đề gì?"),
    ("CBC021", "HCT", "Tăng/giảm", "HCT cao có thể do nguyên nhân nào?"),
    ("CBC022", "MCV", "Tăng/giảm", "MCV thấp có ý nghĩa gì trong đánh giá thiếu máu?"),
    ("CBC023", "MCV", "Tăng/giảm", "MCV cao có thể liên quan đến vấn đề gì?"),
    ("CBC024", "MCH", "Tăng/giảm", "MCH thấp có liên quan đến thiếu máu không?"),
    ("CBC025", "MCHC", "Tăng/giảm", "MCHC thấp có ý nghĩa gì?"),
    ("CBC026", "RDW", "Tăng/giảm", "RDW cao có ý nghĩa gì trong đánh giá thiếu máu?"),
    ("CBC027", "WBC", "Tăng/giảm", "WBC cao có ý nghĩa gì?"),
    ("CBC028", "WBC", "Tăng/giảm", "WBC thấp có nguy hiểm không?"),
    ("CBC029", "WBC", "Tăng/giảm", "WBC cao có chắc chắn là nhiễm trùng không?"),
    ("CBC030", "WBC", "Tăng/giảm", "WBC cao có phải ung thư máu không?"),
    ("CBC031", "NEUT", "Tăng/giảm", "NEUT cao thường liên quan đến vấn đề gì?"),
    ("CBC032", "NEUT", "Tăng/giảm", "NEUT thấp có ảnh hưởng gì đến sức khỏe?"),
    ("CBC033", "WBC + NEUT", "Tăng/giảm", "WBC cao kèm NEUT cao nói lên điều gì?"),
    ("CBC034", "WBC + NEUT", "Tăng/giảm", "WBC và NEUT cao có cần dùng kháng sinh không?"),
    ("CBC035", "LYMPH", "Tăng/giảm", "LYMPH cao có ý nghĩa gì?"),
    ("CBC036", "LYMPH", "Tăng/giảm", "LYMPH thấp có đáng lo không?"),
    ("CBC037", "NEUT + LYMPH", "Tăng/giảm", "NEUT cao nhưng LYMPH thấp có thể gợi ý điều gì?"),
    ("CBC038", "MONO", "Tăng/giảm", "MONO cao thường liên quan đến tình trạng nào?"),
    ("CBC039", "EOS", "Tăng/giảm", "EOS cao có thể liên quan đến dị ứng hoặc ký sinh trùng không?"),
    ("CBC040", "BASO", "Tăng/giảm", "BASO cao có ý nghĩa gì?"),
    ("CBC041", "PLT", "Tăng/giảm", "PLT thấp có nguy hiểm không?"),
    ("CBC042", "PLT", "Tăng/giảm", "PLT cao có ý nghĩa gì?"),
    ("CBC043", "PLT", "Tăng/giảm", "Tiểu cầu thấp khi nào cần đi khám sớm?"),
    ("CBC044", "IG", "Tăng/giảm", "IG tăng có thể gợi ý điều gì trong công thức máu?"),
    ("CBC045", "HGB + HCT + RBC", "Tăng/giảm", "HGB, HCT và RBC cùng thấp thì có thể gợi ý điều gì?"),
    ("CBC046", "HGB + MCV", "Tăng/giảm", "HGB thấp kèm MCV thấp thường gợi ý điều gì?"),
    ("CBC047", "HGB + MCV", "Tăng/giảm", "HGB thấp kèm MCV cao có thể liên quan đến vấn đề gì?"),
    ("CBC048", "CBC tổng quát", "Tăng/giảm", "Một chỉ số trong công thức máu hơi cao thì có nguy hiểm không?"),
    ("CBC049", "CBC tổng quát", "Tăng/giảm", "Kết quả công thức máu bất thường có chắc chắn là bệnh không?"),
    ("CBC050", "CBC tổng quát", "Tăng/giảm", "Khi nào cần gặp bác sĩ sau khi có kết quả công thức máu bất thường?"),
]


FIELDNAMES = [
    "id",
    "indicator",
    "type",
    "question",
    "status",
    "intent",
    "has_answer",
    "reference_count",
    "biochem_leak",
    "has_error",
    "error",
    "attempts",
    "created_at",
    "answer",

    # Cột cho human đánh giá sau
    "human_correctness_0_2",
    "human_medical_safety_0_2",
    "human_citation_relevance_0_2",
    "human_no_noise_0_2",
    "human_clarity_0_2",
    "human_total_0_10",
    "human_note",
]


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def load_checkpoint():
    """
    Đọc checkpoint JSONL.
    Nếu cùng ID xuất hiện nhiều lần, lấy bản cuối cùng.
    """
    results = {}

    if not CHECKPOINT_FILE.exists():
        return results

    with CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                qid = row.get("id")
                if qid:
                    results[qid] = row
            except Exception:
                continue

    return results


def append_checkpoint(row):
    """
    Ghi từng câu ngay sau khi chạy xong.
    Nếu script chết giữa chừng thì không mất kết quả cũ.
    """
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_chat_once(question: str, session_id: str):
    payload = {
        "text": question,
        "session_id": session_id
    }

    r = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict):
        if "answer" in data:
            return data.get("answer", ""), data.get("intent", "")

        if "data" in data and isinstance(data["data"], dict):
            return data["data"].get("answer", ""), data["data"].get("intent", "")

    return str(data), ""


def call_chat_with_retry(question: str, session_id: str):
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            answer, intent = call_chat_once(question, session_id)

            if not answer or not answer.strip():
                raise RuntimeError("Empty answer from chatbot API")

            return {
                "status": "OK",
                "answer": answer,
                "intent": intent,
                "error": "",
                "attempts": attempt,
            }

        except Exception as e:
            last_error = str(e)
            print(f"  ⚠️ Attempt {attempt}/{MAX_RETRIES} failed: {last_error}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    return {
        "status": "ERROR",
        "answer": "",
        "intent": "",
        "error": last_error,
        "attempts": MAX_RETRIES,
    }


def has_biochem_leak(answer: str) -> bool:
    leak_words = [
        "tietz",
        "henry",
        "creatinine",
        "glucose",
        "triglyceride",
        "cholesterol",
        "bilirubin",
        "ast",
        "alt",
        "urea",
        "sinh hóa",
        "men gan",
        "mỡ máu",
    ]

    a = (answer or "").lower()
    return any(w in a for w in leak_words)


def count_references(answer: str) -> int:
    a = answer or ""
    count = 0

    for i in range(1, 10):
        if f"[{i}]" in a:
            count += 1

    return count


def make_row(qid, indicator, qtype, question, result):
    answer = result.get("answer", "")

    return {
        "id": qid,
        "indicator": indicator,
        "type": qtype,
        "question": question,
        "status": result.get("status", "ERROR"),
        "intent": result.get("intent", ""),
        "has_answer": bool(answer.strip()),
        "reference_count": count_references(answer),
        "biochem_leak": has_biochem_leak(answer),
        "has_error": bool(result.get("error")),
        "error": result.get("error", ""),
        "attempts": result.get("attempts", 0),
        "created_at": now_iso(),
        "answer": answer,

        # Để trống cho human chấm sau
        "human_correctness_0_2": "",
        "human_medical_safety_0_2": "",
        "human_citation_relevance_0_2": "",
        "human_no_noise_0_2": "",
        "human_clarity_0_2": "",
        "human_total_0_10": "",
        "human_note": "",
    }


def write_csv(rows, path):
    rows = sorted(rows, key=lambda x: x["id"])

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in rows:
            clean_row = {k: row.get(k, "") for k in FIELDNAMES}
            writer.writerow(clean_row)


def export_all_csv():
    checkpoint = load_checkpoint()
    rows = list(checkpoint.values())
    write_csv(rows, CSV_FILE)

    failed_rows = [
        r for r in rows
        if r.get("status") != "OK" or r.get("has_error") is True
    ]
    write_csv(failed_rows, FAILED_CSV_FILE)

    print(f"✅ Exported: {CSV_FILE}")
    print(f"✅ Exported failed only: {FAILED_CSV_FILE}")


def should_run_question(qid, old_results, mode):
    old = old_results.get(qid)

    if mode == "all":
        return True

    if mode == "resume":
        # Chỉ chạy câu chưa có kết quả
        return old is None

    if mode == "failed":
        # Chỉ chạy lại câu lỗi
        if old is None:
            return True
        return old.get("status") != "OK" or bool(old.get("has_error"))

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["resume", "failed", "all", "export"],
        default="resume",
        help=(
            "resume: chỉ chạy câu chưa có checkpoint; "
            "failed: chỉ chạy lại câu lỗi; "
            "all: chạy lại tất cả; "
            "export: chỉ xuất CSV từ checkpoint"
        )
    )
    parser.add_argument(
        "--session-id",
        default=SESSION_ID,
        help="Session ID gửi tới chatbot API"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Thời gian nghỉ giữa mỗi câu"
    )
    args = parser.parse_args()

    if args.mode == "export":
        export_all_csv()
        return

    old_results = load_checkpoint()

    print(f"Loaded checkpoint: {len(old_results)} existing rows")
    print(f"Mode: {args.mode}")
    print(f"Session ID: {args.session_id}")
    print("-" * 80)

    for qid, indicator, qtype, question in QUESTIONS:
        if not should_run_question(qid, old_results, args.mode):
            print(f"⏭️ Skip {qid}: already done")
            continue

        print(f"▶️ Testing {qid}: {question}")

        result = call_chat_with_retry(question, args.session_id)
        row = make_row(qid, indicator, qtype, question, result)

        append_checkpoint(row)
        old_results[qid] = row

        if row["status"] == "OK":
            print(
                f"  ✅ OK | refs={row['reference_count']} | "
                f"biochem_leak={row['biochem_leak']}"
            )
        else:
            print(f"  ❌ ERROR | {row['error']}")

        time.sleep(args.sleep)

    export_all_csv()
    print("Done.")


if __name__ == "__main__":
    main()

# python test_50_cbc.py --mode resume