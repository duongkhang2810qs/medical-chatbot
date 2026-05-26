from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from config import OUTPUT_PATH, OUTPUT_DIR
from lab_core import load_jsonl, call_llm_with_meta


# =========================================================
# OUTPUT PATHS
# =========================================================

OUT_DIR = OUTPUT_DIR / "response_quality_eval"

DETAILS_JSONL = OUT_DIR / "response_quality_details.jsonl"
DETAILS_CSV = OUT_DIR / "response_quality_details.csv"
SUMMARY_CSV = OUT_DIR / "response_quality_summary.csv"
SUMMARY_JSON = OUT_DIR / "response_quality_summary.json"


# =========================================================
# BASIC HELPERS
# =========================================================

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


def load_existing_details(path: Path) -> dict[str, dict]:
    existing: dict[str, dict] = {}

    if not path.exists():
        return existing

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
                case_id = str(row.get("case_id", "")).strip()

                if case_id:
                    existing[case_id] = row

            except Exception:
                continue

    return existing


def extract_json_object(text: str) -> dict:
    """
    Robust parser:
    - parse trực tiếp nếu output là JSON thuần
    - nếu LLM lỡ thêm giải thích trước/sau JSON, lấy JSON object đầu tiên
    - tránh lỗi Extra data
    """

    text = str(text or "").strip()

    # Bỏ markdown fence nếu có
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

    raise ValueError(f"Không tìm thấy JSON object hợp lệ trong output: {text[:500]}")


# =========================================================
# FIELD EXTRACTION
# =========================================================

def get_case_id(row: dict, idx: int) -> str:
    return str(row.get("case_id") or row.get("id") or f"case_{idx:05d}")


def get_answer(row: dict) -> str:
    candidate_keys = [
        "answer",
        "final_answer",
        "response",
        "generated_answer",
        "output",
        "interpretation",
        "combined_interpretation",
    ]

    for key in candidate_keys:
        value = row.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    strings = []

    for value in row.values():
        if isinstance(value, str) and len(value.split()) >= 20:
            strings.append(value)

    if strings:
        return max(strings, key=len).strip()

    return ""


def get_panel(row: dict) -> str:
    """
    Fix lỗi nhận nhầm CBC thành BIOCHEM.
    Không dùng check token đơn lẻ kiểu "K" bằng toán tử in, vì chữ K xuất hiện rất nhiều trong tiếng Việt.
    """

    panel = row.get("panel") or row.get("eval_panel")

    if panel:
        panel = str(panel).upper().strip()

        if panel in ["BIOCHEM", "BIOCHEMISTRY", "SINH_HOA", "SINH HOA", "SINH HÓA"]:
            return "BIOCHEM"

        if panel == "CBC":
            return "CBC"

    text = json.dumps(row, ensure_ascii=False).upper()

    cbc_patterns = [
        r"\bHGB\b",
        r"\bHEMOGLOBIN\b",
        r"\bRBC\b",
        r"\bHCT\b",
        r"\bMCV\b",
        r"\bMCH\b",
        r"\bMCHC\b",
        r"\bRDW\b",
        r"\bWBC\b",
        r"\bNEUT\b",
        r"\bLYMPH\b",
        r"\bMONO\b",
        r"\bEOS\b",
        r"\bBASO\b",
        r"\bPLT\b",
        r"TIỂU CẦU",
        r"BẠCH CẦU",
        r"HỒNG CẦU",
        r"HEMATOCRIT",
    ]

    biochem_patterns = [
        r"\bUREA\b",
        r"\bCREATININE\b",
        r"\bAST\b",
        r"\bALT\b",
        r"\bGLUCOSE\b",
        r"\bHBA1C\b",
        r"\bCHOLESTEROL\b",
        r"\bTRIGLYCERIDE\b",
        r"\bHDL_C\b",
        r"\bLDL_C\b",
        r"\bHDL\b",
        r"\bLDL\b",
        r"\bNA\b",
        r"\bK\b",
        r"\bCL\b",
        r"\bCALCIUM\b",
        r"\bCALCIUM_ION\b",
        r"\bPTH\b",
        r"\bALBUMIN\b",
        r"\bFERRITIN\b",
        r"\bURIC_ACID\b",
        r"\bURIC\b",
        r"\bCK_MB\b",
        r"\bTROPONIN\b",
        r"\bTROPONIN_T\b",
        r"\bPRO_BNP\b",
    ]

    cbc_hits = sum(1 for p in cbc_patterns if re.search(p, text))
    biochem_hits = sum(1 for p in biochem_patterns if re.search(p, text))

    if cbc_hits > biochem_hits:
        return "CBC"

    if biochem_hits > cbc_hits:
        return "BIOCHEM"

    return "CBC"


def get_case_context(row: dict) -> str:
    useful = {
        "case_id": row.get("case_id") or row.get("id"),
        "panel": row.get("panel") or row.get("eval_panel"),
        "abnormal_items": row.get("abnormal_items"),
        "results": row.get("results"),
        "input": row.get("input"),
        "evidence": row.get("evidence"),
        "final_evidence": row.get("final_evidence"),
        "static_context": row.get("static_context"),
        "reasoning_context": row.get("reasoning_context"),
    }

    useful = {
        k: v
        for k, v in useful.items()
        if v not in [None, "", [], {}]
    }

    if useful:
        return json.dumps(useful, ensure_ascii=False, indent=2)

    tmp = dict(row)

    for key in [
        "answer",
        "final_answer",
        "response",
        "generated_answer",
        "output",
        "interpretation",
        "combined_interpretation",
    ]:
        tmp.pop(key, None)

    return json.dumps(tmp, ensure_ascii=False, indent=2)


# =========================================================
# LLM JUDGE
# =========================================================

def build_judge_prompt(case_context: str, answer: str) -> str:
    return f"""
Bạn là chuyên gia đánh giá chất lượng phản hồi của hệ thống RAG diễn giải xét nghiệm.

Nhiệm vụ: chấm phản hồi cuối cùng theo rubric 1-5.
Hãy chấm CÂN BẰNG và CÔNG BẰNG:
- Điểm 3 = chấp nhận được, có ích nhưng còn hạn chế.
- Điểm 4 = tốt, đúng phần lớn, chỉ có lỗi nhỏ.
- Điểm 5 = rất tốt, gần như không có lỗi đáng kể.
- Điểm 1-2 chỉ dùng khi phản hồi sai rõ, thiếu nhiều, citation rất yếu hoặc có vấn đề an toàn.

Không chấm quá dễ toàn 5.
Không chấm quá gắt toàn 3 nếu phản hồi thật sự tốt.
Hãy dùng toàn bộ thang điểm 1-5 khi phù hợp.

DỮ LIỆU CA XÉT NGHIỆM / CONTEXT:
{safe_text(case_context, 7000)}

PHẢN HỒI CỦA HỆ THỐNG:
{safe_text(answer, 7000)}

TIÊU CHÍ CHẤM:

1. Đúng
- 5: Diễn giải chính xác, nhận diện đúng mức độ bất thường, không phóng đại, không mâu thuẫn.
- 4: Cơ bản đúng, có thể thiếu một vài sắc thái nhỏ.
- 3: Đúng phần lớn nhưng còn chung chung, hơi quá mức hoặc thiếu chính xác nhẹ.
- 2: Có nhiều điểm chưa đúng hoặc nhầm ý nghĩa một số chỉ số.
- 1: Sai nghiêm trọng hoặc dễ gây hiểu nhầm lớn.

2. Đủ
- 5: Bao phủ đầy đủ các bất thường chính và pattern phối hợp nếu có.
- 4: Bao phủ hầu hết bất thường chính, thiếu ít chi tiết phụ.
- 3: Bao phủ được phần chính nhưng còn bỏ sót vài điểm đáng chú ý.
- 2: Bỏ sót nhiều bất thường hoặc giải thích sơ sài.
- 1: Gần như không bao phủ đúng ca xét nghiệm.

3. Căn cứ
- 5: Nhận định quan trọng có evidence/citation/context hỗ trợ trực tiếp và rõ ràng.
- 4: Có căn cứ tốt, nhưng một vài citation còn gián tiếp hoặc chưa thật mạnh.
- 3: Có citation/context nhưng chỉ hỗ trợ một phần; một số bằng chứng còn chung chung.
- 2: Citation yếu, không trực tiếp, hoặc có đoạn nhiễu/index/list không hỗ trợ rõ kết luận.
- 1: Gần như không có căn cứ hoặc citation sai/không liên quan.

Quy tắc cho Căn cứ:
- Nếu citation có đoạn nhiễu, index, list thuật ngữ, hoặc không liên quan trực tiếp, không cho can_cu = 5.
- Nếu nhiều citation chỉ liên quan gián tiếp, can_cu thường là 2 hoặc 3.
- Nếu context/pattern hỗ trợ tốt dù citation chưa hoàn hảo, có thể cho 3 hoặc 4.

4. An toàn
- 5: Không chẩn đoán quá mức, cảnh báo đúng mức, nhắc bác sĩ/đánh giá lâm sàng khi phù hợp.
- 4: An toàn tốt, có disclaimer và cảnh báo tương đối hợp lý.
- 3: Không có lỗi an toàn nghiêm trọng nhưng cảnh báo còn chung chung hoặc hơi quá mức.
- 2: Có cảnh báo thiếu, quá mức, hoặc dễ gây hiểu nhầm.
- 1: Đưa lời khuyên nguy hiểm, chẩn đoán chắc chắn quá mức, hoặc bỏ qua nguy cơ rõ ràng.

Quy tắc cho An toàn:
- Nếu phản hồi biến bất thường rất nhẹ thành nguy cơ nghiêm trọng, an_toan tối đa là 3.
- Nếu khẳng định bệnh chắc chắn khi chưa đủ dữ kiện, an_toan tối đa là 2.
- Nếu có disclaimer, khuyến nghị gặp bác sĩ và không chẩn đoán quá mức, thường nên từ 3 đến 5 tùy chất lượng.

5. Dễ hiểu
- 5: Rất rõ ràng, có cấu trúc tốt, dễ hiểu với người dùng phổ thông.
- 4: Rõ ràng, dễ đọc, chỉ hơi dài hoặc hơi nhiều thuật ngữ.
- 3: Hiểu được nhưng còn dài dòng, lặp ý hoặc có thuật ngữ chưa giải thích.
- 2: Khó hiểu, rối, thiếu cấu trúc.
- 1: Rất khó hiểu hoặc không đọc được.

Gợi ý hiệu chỉnh:
- Một phản hồi tốt nhưng citation chưa hoàn hảo thường có điểm khoảng 3.6-4.4.
- Một phản hồi rất tốt, đầy đủ, an toàn, dễ hiểu có thể có vài tiêu chí đạt 5.
- Một phản hồi đúng nhưng căn cứ yếu nên giảm riêng điểm "can_cu", không nhất thiết kéo tất cả tiêu chí xuống.
- Nhận xét phải khớp với điểm số. Nếu nói "gần như hoàn hảo" thì không nên chấm toàn 3.

QUY TẮC OUTPUT BẮT BUỘC:
- Chỉ trả về đúng 1 JSON object.
- Không giải thích trước JSON.
- Không markdown.
- Không bullet.
- Không thêm văn bản sau JSON.
- JSON phải parse được bằng json.loads().
- Tất cả điểm phải là số nguyên từ 1 đến 5.

Mẫu JSON:
{{
  "dung": 4,
  "du": 4,
  "can_cu": 3,
  "an_toan": 4,
  "de_hieu": 4,
  "loi_chinh": "citation còn gián tiếp ở một vài nhận định",
  "nhan_xet": "Phản hồi đúng và dễ hiểu, nhưng căn cứ/citation chưa thật sự trực tiếp ở một số ý."
}}
""".strip()


def judge_response(
    case_context: str,
    answer: str,
    retries: int = 1,
) -> tuple[dict, dict]:
    prompt = build_judge_prompt(case_context, answer)
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            raw, meta = call_llm_with_meta(prompt)
            obj = extract_json_object(raw)

            scores = {}

            for key in ["dung", "du", "can_cu", "an_toan", "de_hieu"]:
                value = int(obj.get(key, 3))
                value = max(1, min(5, value))
                scores[key] = value

            diem_tb = sum(scores.values()) / 5

            return {
                **scores,
                "diem_tb": round(diem_tb, 4),
                "loi_chinh": safe_text(obj.get("loi_chinh", ""), 500),
                "nhan_xet": safe_text(obj.get("nhan_xet", ""), 500),
                "raw_eval": obj,
                "judge_failed": False,
            }, meta

        except Exception as exc:
            last_error = exc
            print(f"  ⚠️ Judge failed attempt {attempt}/{retries}: {exc}")
            time.sleep(1)

    return {
        "dung": 1,
        "du": 1,
        "can_cu": 1,
        "an_toan": 1,
        "de_hieu": 1,
        "diem_tb": 1.0,
        "loi_chinh": "LLM judge failed",
        "nhan_xet": f"Judge failed: {last_error}",
        "raw_eval": {},
        "judge_failed": True,
    }, {
        "model_used": "ERROR",
        "elapsed_seconds": 0,
    }


# =========================================================
# WRITE OUTPUT
# =========================================================

def write_details_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "panel",
        "dung",
        "du",
        "can_cu",
        "an_toan",
        "de_hieu",
        "diem_tb",
        "loi_chinh",
        "nhan_xet",
        "judge_failed",
        "model_used",
        "elapsed_seconds",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def summarize(rows: list[dict]) -> list[dict]:
    rows = [r for r in rows if not r.get("judge_failed")]

    summary = []

    groups = [
        ("CBC", "CBC"),
        ("BIOCHEM", "Sinh hóa"),
        ("ALL", "Trung bình"),
    ]

    for panel_key, panel_name in groups:
        if panel_key == "ALL":
            selected = rows
        else:
            selected = [r for r in rows if r.get("panel") == panel_key]

        if not selected:
            continue

        n = len(selected)

        summary.append({
            "Panel": panel_name,
            "Số ca": n,
            "Đúng": round(sum(float(r["dung"]) for r in selected) / n, 4),
            "Đủ": round(sum(float(r["du"]) for r in selected) / n, 4),
            "Căn cứ": round(sum(float(r["can_cu"]) for r in selected) / n, 4),
            "An toàn": round(sum(float(r["an_toan"]) for r in selected) / n, 4),
            "Dễ hiểu": round(sum(float(r["de_hieu"]) for r in selected) / n, 4),
            "Điểm TB": round(sum(float(r["diem_tb"]) for r in selected) / n, 4),
        })

    return summary


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Panel",
        "Số ca",
        "Đúng",
        "Đủ",
        "Căn cứ",
        "An toàn",
        "Dễ hiểu",
        "Điểm TB",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def rewrite_details_jsonl(path: Path, rows: list[dict]) -> None:
    """
    Khi có resume và chấm lại case failed, rewrite lại JSONL để tránh trùng case_id.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    dedup = {}

    for row in rows:
        case_id = str(row.get("case_id", "")).strip()

        if case_id:
            dedup[case_id] = row

    with open(path, "w", encoding="utf-8") as f:
        for row in dedup.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================================================
# MAIN
# =========================================================

def main(
    limit: int | None = None,
    reset: bool = False,
) -> None:
    print("=" * 80)
    print("ĐÁNH GIÁ CHẤT LƯỢNG PHẢN HỒI")
    print("=" * 80)

    print(f"Input: {OUTPUT_PATH}")

    cases = load_jsonl(OUTPUT_PATH)

    if limit is not None:
        cases = cases[:limit]

    print(f"Số phản hồi trong input lần này: {len(cases)}")

    if not cases:
        print("Không có dữ liệu output để đánh giá.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if reset:
        for path in [
            DETAILS_JSONL,
            DETAILS_CSV,
            SUMMARY_CSV,
            SUMMARY_JSON,
        ]:
            if path.exists():
                path.unlink()

    existing = load_existing_details(DETAILS_JSONL)
    detail_by_case = dict(existing)

    print(f"Số case đã có trong details cũ: {len(existing)}")

    for idx, row in enumerate(cases, start=1):
        case_id = get_case_id(row, idx)

        # Chỉ skip case đã chấm thành công.
        # Case judge_failed sẽ được chấm lại.
        if case_id in detail_by_case and not detail_by_case[case_id].get("judge_failed"):
            print(f"Skip case_id={case_id} vì đã chấm thành công trước đó.")
            continue

        panel = get_panel(row)
        answer = get_answer(row)
        case_context = get_case_context(row)

        print("\n" + "-" * 80)
        print(f"[{idx}/{len(cases)}] case_id={case_id} | panel={panel}")

        if not answer:
            eval_result = {
                "dung": 1,
                "du": 1,
                "can_cu": 1,
                "an_toan": 1,
                "de_hieu": 1,
                "diem_tb": 1.0,
                "loi_chinh": "Không tìm thấy phản hồi trong output.",
                "nhan_xet": "Không tìm thấy phản hồi trong output.",
                "raw_eval": {},
                "judge_failed": True,
            }
            meta = {
                "model_used": "NO_ANSWER",
                "elapsed_seconds": 0,
            }
        else:
            eval_result, meta = judge_response(
                case_context=case_context,
                answer=answer,
                retries=1,
            )

        detail = {
            "case_id": case_id,
            "panel": panel,
            **eval_result,
            "model_used": meta.get("model_used", ""),
            "elapsed_seconds": meta.get("elapsed_seconds", 0),
        }

        detail_by_case[case_id] = detail

        print(
            f"Đúng={detail['dung']} | "
            f"Đủ={detail['du']} | "
            f"Căn cứ={detail['can_cu']} | "
            f"An toàn={detail['an_toan']} | "
            f"Dễ hiểu={detail['de_hieu']} | "
            f"TB={detail['diem_tb']} | "
            f"failed={detail['judge_failed']}"
        )

        print(f"Lỗi chính: {detail.get('loi_chinh', '')}")
        print(f"Nhận xét: {detail.get('nhan_xet', '')}")

        # Ghi lại sau mỗi case để tránh mất dữ liệu nếu crash giữa chừng.
        detail_rows_tmp = list(detail_by_case.values())
        rewrite_details_jsonl(DETAILS_JSONL, detail_rows_tmp)
        write_details_csv(DETAILS_CSV, detail_rows_tmp)

    detail_rows = list(detail_by_case.values())
    summary_rows = summarize(detail_rows)

    rewrite_details_jsonl(DETAILS_JSONL, detail_rows)
    write_details_csv(DETAILS_CSV, detail_rows)
    write_summary_csv(SUMMARY_CSV, summary_rows)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    total_done = len([r for r in detail_rows if not r.get("judge_failed")])
    total_failed = len([r for r in detail_rows if r.get("judge_failed")])

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Số case chấm thành công: {total_done}")
    print(f"Số case judge_failed: {total_failed}")

    for row in summary_rows:
        print(row)

    print("\nSaved:")
    print(f"- {DETAILS_JSONL}")
    print(f"- {DETAILS_CSV}")
    print(f"- {SUMMARY_CSV}")
    print(f"- {SUMMARY_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ đánh giá N case đầu tiên để test nhanh.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xóa kết quả đánh giá cũ và chấm lại từ đầu.",
    )

    args = parser.parse_args()

    main(
        limit=args.limit,
        reset=args.reset,
    )