import requests
import csv
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

API_URL = "http://127.0.0.1:8002/api/v1/chat"
SESSION_ID = "eval_biochem_001"

CHECKPOINT_FILE = Path("biochem_50_eval_checkpoint.jsonl")
CSV_FILE = Path("biochem_50_eval_results.csv")
FAILED_CSV_FILE = Path("biochem_50_eval_failed.csv")

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 3
REQUEST_TIMEOUT = 120


QUESTIONS = [
    ("BIO001", "AST", "Khái niệm", "AST là gì trong xét nghiệm sinh hóa máu?"),
    ("BIO002", "ALT", "Khái niệm", "ALT là gì trong xét nghiệm sinh hóa máu?"),
    ("BIO003", "UREA", "Khái niệm", "Urea/BUN là gì trong xét nghiệm máu?"),
    ("BIO004", "CREATININE", "Khái niệm", "Creatinine là gì trong xét nghiệm máu?"),
    ("BIO005", "NA", "Khái niệm", "Natri máu là gì?"),
    ("BIO006", "K", "Khái niệm", "Kali máu là gì?"),
    ("BIO007", "CL", "Khái niệm", "Clo máu là gì?"),
    ("BIO008", "GLUCOSE", "Khái niệm", "Glucose máu là gì?"),
    ("BIO009", "HBA1C", "Khái niệm", "HbA1c là gì và khác gì với glucose máu?"),
    ("BIO010", "CHOLESTEROL", "Khái niệm", "Cholesterol toàn phần là gì?"),
    ("BIO011", "TRIGLYCERIDE", "Khái niệm", "Triglyceride là gì?"),
    ("BIO012", "HDL_C", "Khái niệm", "HDL-C là gì?"),
    ("BIO013", "LDL_C", "Khái niệm", "LDL-C là gì?"),
    ("BIO014", "CK_MB", "Khái niệm", "CK-MB là gì trong xét nghiệm máu?"),
    ("BIO015", "TROPONIN_T", "Khái niệm", "Troponin T là gì?"),
    ("BIO016", "PRO_BNP", "Khái niệm", "Pro-BNP là gì và thường dùng để đánh giá gì?"),
    ("BIO017", "FERRITIN", "Khái niệm", "Ferritin là gì trong xét nghiệm máu?"),
    ("BIO018", "ALBUMIN", "Khái niệm", "Albumin là gì trong xét nghiệm máu?"),
    ("BIO019", "CALCIUM_ION", "Khái niệm", "Canxi ion hóa là gì?"),
    ("BIO020", "PTH", "Khái niệm", "PTH là gì?"),
    ("BIO021", "URIC_ACID", "Khái niệm", "Acid uric là gì?"),

    ("BIO022", "AST/ALT", "Tăng/giảm", "AST và ALT cao có ý nghĩa gì?"),
    ("BIO023", "AST/ALT", "Tăng/giảm", "Men gan cao có chắc chắn là viêm gan không?"),
    ("BIO024", "AST/ALT", "Tăng/giảm", "Men gan cao khi nào cần đi khám?"),
    ("BIO025", "UREA", "Tăng/giảm", "Urea máu cao có thể liên quan đến vấn đề gì?"),
    ("BIO026", "UREA", "Tăng/giảm", "Urea thấp có ý nghĩa gì không?"),
    ("BIO027", "CREATININE", "Tăng/giảm", "Creatinine cao có ý nghĩa gì?"),
    ("BIO028", "CREATININE", "Tăng/giảm", "Creatinine cao có chắc chắn là suy thận không?"),
    ("BIO029", "UREA + CREATININE", "Tăng/giảm", "Urea và creatinine cùng cao có thể gợi ý điều gì?"),
    ("BIO030", "NA", "Tăng/giảm", "Natri máu thấp có nguy hiểm không?"),
    ("BIO031", "NA", "Tăng/giảm", "Natri máu cao có ý nghĩa gì?"),
    ("BIO032", "K", "Tăng/giảm", "Kali máu cao có nguy hiểm không?"),
    ("BIO033", "K", "Tăng/giảm", "Kali máu thấp có thể gây vấn đề gì?"),
    ("BIO034", "CL", "Tăng/giảm", "Clo máu bất thường có thể liên quan đến tình trạng nào?"),
    ("BIO035", "GLUCOSE", "Tăng/giảm", "Glucose máu cao có ý nghĩa gì?"),
    ("BIO036", "GLUCOSE", "Tăng/giảm", "Glucose máu thấp có nguy hiểm không?"),
    ("BIO037", "GLUCOSE", "Tăng/giảm", "Glucose cao có chắc chắn là tiểu đường không?"),
    ("BIO038", "HBA1C", "Tăng/giảm", "HbA1c cao có ý nghĩa gì?"),
    ("BIO039", "CHOLESTEROL", "Tăng/giảm", "Cholesterol toàn phần cao có nguy hiểm không?"),
    ("BIO040", "LDL_C", "Tăng/giảm", "LDL-C cao có ý nghĩa gì đối với nguy cơ tim mạch?"),
    ("BIO041", "HDL_C", "Tăng/giảm", "HDL-C thấp có ý nghĩa gì?"),
    ("BIO042", "TRIGLYCERIDE", "Tăng/giảm", "Triglyceride cao có nguy hiểm không?"),
    ("BIO043", "Lipid profile", "Tăng/giảm", "LDL cao, HDL thấp và triglyceride cao nói lên điều gì?"),
    ("BIO044", "CK_MB/TROPONIN_T", "Tăng/giảm", "CK-MB hoặc Troponin T cao có thể liên quan đến vấn đề gì?"),
    ("BIO045", "PRO_BNP", "Tăng/giảm", "Pro-BNP cao có ý nghĩa gì?"),
    ("BIO046", "FERRITIN", "Tăng/giảm", "Ferritin thấp hoặc cao có thể liên quan đến tình trạng nào?"),
    ("BIO047", "ALBUMIN", "Tăng/giảm", "Albumin thấp có thể liên quan đến vấn đề gì?"),
    ("BIO048", "CALCIUM_ION/PTH", "Tăng/giảm", "Canxi ion hóa hoặc PTH bất thường có thể gợi ý điều gì?"),
    ("BIO049", "URIC_ACID", "Tăng/giảm", "Acid uric cao có phải là gout không?"),
    ("BIO050", "Sinh hóa tổng quát", "Tăng/giảm", "Một chỉ số sinh hóa máu bất thường có chắc chắn là bệnh không?"),
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
    "cbc_leak",
    "has_error",
    "error",
    "attempts",
    "created_at",
    "answer",

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


def has_cbc_leak(answer: str) -> bool:
    """
    Với bộ sinh hóa, flag nếu câu trả lời bị lôi sang CBC/huyết học quá rõ.
    Một số từ như 'máu' không tính là leak vì sinh hóa cũng là xét nghiệm máu.
    """
    leak_words = [
        "clinical_hematology",
        "hematology",
        "huyết học",
        "công thức máu",
        "wbc",
        "rbc",
        "hgb",
        "hct",
        "mcv",
        "mch",
        "mchc",
        "rdw",
        "plt",
        "bạch cầu",
        "hồng cầu",
        "tiểu cầu",
        "neut",
        "lymph",
        "lym",
        "mono",
        "eos",
        "baso",
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
        "cbc_leak": has_cbc_leak(answer),
        "has_error": bool(result.get("error")),
        "error": result.get("error", ""),
        "attempts": result.get("attempts", 0),
        "created_at": now_iso(),
        "answer": answer,

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
        return old is None

    if mode == "failed":
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
                f"cbc_leak={row['cbc_leak']}"
            )
        else:
            print(f"  ❌ ERROR | {row['error']}")

        time.sleep(args.sleep)

    export_all_csv()
    print("Done.")


if __name__ == "__main__":
    main()
    
    
# python test_50_biochem.py --mode resume