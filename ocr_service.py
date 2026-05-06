import os
import json
import re
import unicodedata
import difflib
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
from PIL import Image

# Tắt log rác của PaddleOCR
logging.getLogger("ppocr").setLevel(logging.WARNING)

# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = "outputs"
ONTOLOGY_JSON = "ontology_tests.json"
ONTOLOGY_UNITS_JSON = "ontology_units.json"

OUTPUT_SCANNED_IMG = "1_scanned_doc.jpg"
OUTPUT_DEBUG_IMG = "2_debug_extraction.jpg"
OUTPUT_JSON = "3_extracted_data.json"
OUTPUT_CLEAN_JSON = "4_extracted_data_clean.json"

USE_ONTOLOGY_FILTER = True 

COLUMN_KEYWORDS = {
    "TEST_NAME": [
        "ten xet nghiem", "test name", "parameter", "investigation", "investigations", "ten chi so", "test"
    ],
    "VALUE": [
        "ket qua", "result", "value"
    ],
    "REF_RANGE": [
        "tri so", "binh thuong", "tri so binh thuong", "chi so binh thuong", "normal range", "reference range", "tham chieu",
        "ref range", "ref", "csbt", "tsbt", "khoang tham chieu", "biological ref", "biological ref. interval", "biological reference"
    ],
    "UNIT": [
        "don vi", "unit", "units"
    ],
    "IGNORE_WALL": [
        "phuong phap", "thiet bi", "ma qt", "kt do", "phuong phap/kt", "method", "methods", "may xn"
    ]
}

# =========================================================
# UNIFIED OCR ENGINE INIT (Tích hợp Unwarp & Det/Rec)
# =========================================================

print("Khởi tạo Unified PaddleOCR (Unwarp + Detection + Recognition)...")
ocr_engine = PaddleOCR(
    device="gpu" if paddle.device.is_compiled_with_cuda() else "cpu",
    lang="vi",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    ocr_version="PP-OCRv5",
    enable_mkldnn=False
)

# Lazy-load cho CPU: Gán bằng None, chỉ load khi GPU bị OOM
ocr_cpu = None

# =========================================================
# HELPERS (TEXT & MATH)
# =========================================================

def strip_accents(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def normalize_text(s):
    if s is None: return ""
    s = str(s).strip().lower()
    s = strip_accents(s)
    repl = {
        "µ": "u", "×": "x", "–": "-", "—": "-", "−": "-",
        "＜": "<", "＞": ">", "\n": " ", "\t": " ",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_for_match(s):
    s = normalize_text(s)
    s = re.sub(r"[^a-z0-9%/\+\-\.\#\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def str_to_float(s):
    try:
        return float(s.replace(',', '.'))
    except Exception:
        return 0.0

def polygon_to_bbox(poly):
    if not poly: return None
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]

def bbox_height(bbox):
    return max(1.0, bbox[3] - bbox[1])

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_union(boxes):
    if not boxes: return None
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return [int(x1), int(y1), int(x2), int(y2)]

def median_token_height(tokens):
    hs = [bbox_height(t["bbox"]) for t in tokens if t.get("bbox") is not None]
    return float(np.median(hs)) if hs else 14.0

def save_image(path, image):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

def pil_to_bgr(img):
    if img is None: return None
    if isinstance(img, Image.Image): return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if isinstance(img, np.ndarray): return img
    return None

def extract_single_unwarped_panel(img):
    if img is None or not isinstance(img, np.ndarray): return img
    h, w = img.shape[:2]
    if w < h * 1.8: return img
    third = w // 3
    if third > 50:
        panels = [img[:, 0:third], img[:, third:2*third], img[:, 2*third:w]]
        widths = [p.shape[1] for p in panels]
        if min(widths) > 0 and max(widths) - min(widths) < max(20, int(0.1 * third)):
            return panels[2].copy()
    return img

# =========================================================
# LỌC VÀ LÀM SẠCH CHUỖI, ĐƠN VỊ
# =========================================================

def extract_strict_number(val_str):
    if not val_str: return None
    m = re.search(r'[-+]?\d+(?:[\.,]\d+)?', val_str)
    if m:
        try:
            num = float(m.group().replace(',', '.'))
            return int(num) if num.is_integer() else num
        except:
            return None
    return None

def fix_unit_typos(u):
    if not u: return ""
    u = u.replace('$', '').replace('{', '').replace('}', '')
    u = u.replace('³3', '^3').replace('³', '^3').replace('²', '^2').replace('⁹', '^9').replace('¹²', '^12')
    if re.fullmatch(r'[HLhl↑↓←→]+', u.strip()): return ""
    u = re.sub(r'^(?:[HLhl↑↓←→]\s+)+', '', u.strip())
    
    u = re.sub(r'10[\'\"]?1?2/?L', '10^12/L', u, flags=re.IGNORECASE)
    u = re.sub(r'10\^?12/?L', '10^12/L', u, flags=re.IGNORECASE)
    u = re.sub(r'10\%/?L', '10^9/L', u, flags=re.IGNORECASE)
    u = re.sub(r'10\?[/]?L', '10^9/L', u, flags=re.IGNORECASE)
    u = re.sub(r'10[\'\"]?9/?L', '10^9/L', u, flags=re.IGNORECASE)
    u = re.sub(r'10\^?9/?L', '10^9/L', u, flags=re.IGNORECASE)
    u = re.sub(r'10[3\?7\*]/uL', '10^3/uL', u, flags=re.IGNORECASE)
    u = re.sub(r'10\^?3/?uL', '10^3/uL', u, flags=re.IGNORECASE)
    
    if u.lower() in ['i1', 'f1', 'fl']: return "fL"
    return u.strip()

def load_unit_ontology(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = {}
        for k, v in data.items():
            if isinstance(v, list):
                result[str(k).strip()] = [str(a).strip() for a in v if a is not None]
        return result
    except Exception as e:
        print(f"Warning: Could not load unit ontology ({e})")
        return {}

def unit_similarity(u1, u2):
    if not u1 or not u2: return 0.0
    u1 = u1.replace(" ", "").lower()
    u2 = u2.replace(" ", "").lower()
    if u1 == u2: return 100.0
    return difflib.SequenceMatcher(None, u1, u2).ratio() * 100.0

def correct_unit_by_ontology(raw_unit, test_name, unit_ontology):
    allowed_units = unit_ontology.get(test_name, [])
    if not allowed_units: return raw_unit
    clean_raw = str(raw_unit).strip().lower()
    for au in allowed_units:
        if clean_raw == str(au).strip().lower():
            return au
    return allowed_units[0]

# =========================================================
# UNIFIED PADDLE OCR: BÓC TÁCH ẢNH & KẾT QUẢ 1 LẦN CHẠY
# =========================================================

def get_result_dict(result_obj):
    if isinstance(result_obj, list): return result_obj
    data = None
    if hasattr(result_obj, "json"):
        try: data = result_obj.json() if callable(result_obj.json) else result_obj.json
        except Exception: data = None
    if data is None and hasattr(result_obj, "res"): data = result_obj.res
    if isinstance(data, dict): return data
    return {}

def flatten_any_boxes(obj):
    found = []
    def is_nested_poly(x): return (isinstance(x, list) and len(x) >= 1 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in x) and all(isinstance(v, (int, float)) for p in x for v in p[:2]))
    def is_flat_poly(x): return (isinstance(x, list) and len(x) >= 4 and len(x) % 2 == 0 and all(isinstance(v, (int, float)) for v in x))
    def _walk(x):
        if isinstance(x, dict):
            for v in x.values(): _walk(v)
            return
        if isinstance(x, list):
            if is_nested_poly(x) or is_flat_poly(x):
                found.append(x)
                return
            for item in x: _walk(item)
    _walk(obj)
    return found

def extract_tokens_from_result(result_obj):
    data = get_result_dict(result_obj)
    payload = data.get("res", data) if isinstance(data, dict) else data
    texts, boxes, scores = [], [], []

    if isinstance(payload, dict):
        rec_texts = payload.get("rec_texts", [])
        rec_boxes = payload.get("rec_boxes") or payload.get("dt_polys") or payload.get("textline_polys") or payload.get("polys") or []
        rec_scores = payload.get("rec_scores", [])
        if isinstance(rec_texts, list): texts = [str(x) for x in rec_texts]
        if isinstance(rec_boxes, list): boxes = rec_boxes
        if isinstance(rec_scores, list): scores = rec_scores
        if not boxes: boxes = flatten_any_boxes(payload)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, list) and len(item) == 2:
                boxes.append(item[0])
                texts.append(item[1][0])
                scores.append(item[1][1])

    items = []
    n = min(len(texts), len(boxes))
    for i in range(n):
        score = float(scores[i]) if i < len(scores) and isinstance(scores[i], (int, float)) else None
        raw_poly = boxes[i]
        poly = []
        if raw_poly:
            if isinstance(raw_poly[0], (int, float)):
                poly = [[float(raw_poly[j]), float(raw_poly[j+1])] for j in range(0, len(raw_poly)-1, 2)]
            else:
                poly = [[float(p[0]), float(p[1])] for p in raw_poly]
                
        items.append({"text": texts[i], "norm_text": normalize_text(texts[i]), "score": score, "poly": poly, "bbox": polygon_to_bbox(poly)})
    return items

def is_gpu_oom_error(exc):
    msg = str(exc).lower()
    return any(k in msg for k in ["out of memory", "memoryerror", "resourceexhaustederror", "cannot allocate", "gpu 0"])

def predict_unified(image_or_path):
    """
    Chạy PaddleOCR một lần duy nhất, lấy ra cả ảnh đã unwarp và kết quả tokens.
    """
    global ocr_cpu
    used_gpu = True
    input_data = str(image_or_path) if isinstance(image_or_path, (str, Path)) else image_or_path
    
    try:
        results = ocr_engine.predict(input_data)
    except Exception as e:
        if is_gpu_oom_error(e):
            print("   Cảnh báo: GPU OOM, tự động chuyển sang CPU...")
            used_gpu = False
            if ocr_cpu is None:
                print("   -> Đang khởi tạo mô hình PaddleOCR (CPU) lần đầu tiên...")
                ocr_cpu = PaddleOCR(
                    device="cpu",
                    lang="vi",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    ocr_version="PP-OCRv5",
                    enable_mkldnn=False
                )
            results = ocr_cpu.predict(input_data)
        else:
            raise

    candidate_img = None
    tokens = []

    if results:
        for res in results:
            if res is None: continue
            
            # 1. Trích xuất ảnh Unwarped
            if candidate_img is None:
                for attr in ["doc_preprocessed_image", "preprocessed_image", "unwarped_image", "doctr_img", "output_img", "img", "image"]:
                    if hasattr(res, attr):
                        candidate = getattr(res, attr)
                        if callable(candidate):
                            try: candidate = candidate()
                            except: candidate = None
                        
                        if isinstance(candidate, dict):
                            extracted = None
                            for k in ["preprocessed_img", "unwarped_image", "doc_preprocessed_image", "preprocessed_image", "output_img", "image", "img", "res"]:
                                if k in candidate and candidate[k] is not None:
                                    extracted = candidate[k]
                                    break
                            if extracted is None:
                                for k, v in candidate.items():
                                    if isinstance(v, (np.ndarray, Image.Image)) and k != "ocr_res_img":
                                        extracted = v
                                        break
                            candidate = extracted

                        candidate = pil_to_bgr(candidate)
                        candidate = extract_single_unwarped_panel(candidate)

                        if candidate is not None and isinstance(candidate, np.ndarray) and candidate.size > 0:
                            candidate_img = candidate
                            break
            
            # 2. Trích xuất Tokens
            extracted_tokens = extract_tokens_from_result(res)
            if extracted_tokens:
                tokens.extend(extracted_tokens)

    valid_tokens = [t for t in tokens if t.get("bbox") is not None]
    return candidate_img, valid_tokens, used_gpu

def resolve_data_text(paddle_txt):
    """
    Hậu xử lý chuyên biệt để rửa lỗi ảo giác của PaddleOCR (Không cần VietOCR).
    """
    p_str = str(paddle_txt).strip()

    if p_str.startswith('20') and p_str.endswith('>'):
        p_str = p_str.replace('20', '<0', 1).replace('>', '', 1)
            
    if re.match(r'^[ZzSs]0.*>$', p_str):
        p_str = p_str.replace('>', '', 1)
        p_str = re.sub(r'^[ZzSs]', '<', p_str)

    p_str = re.sub(r'^[KkZzSs]\s*(?=[0-9])', '<', p_str)
    p_str = re.sub(r'^[_\|\-\.]+|[_\|\-\.]+$', '', p_str).strip()

    return p_str

# =========================================================
# XỬ LÝ LÕI TRÍCH XUẤT (GỘP DÒNG, TÁCH CỘT, LỌC RÁC)
# =========================================================

def split_tall_token_if_needed(img, token, median_h):
    x1, y1, x2, y2 = map(int, token["bbox"])
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    
    if h < 2.4 * median_h or h < 1.2 * w: return [token]
        
    crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
    if crop.size == 0: return [token]
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    proj = np.sum(thresh, axis=1)
    threshold_val = 255 * 1 
    lines_y, in_line, start_y = [], False, 0
    
    for i, val in enumerate(proj):
        if val > threshold_val:
            if not in_line: start_y, in_line = i, True
        else:
            if in_line:
                end_y = i
                if end_y - start_y > 6: lines_y.append((start_y, end_y))
                in_line = False
    if in_line and len(proj) - start_y > 6:
        lines_y.append((start_y, len(proj)))
            
    if len(lines_y) <= 1: return [token]
        
    new_tokens = []
    original_text = token.get("text", "")
    parts = [p for p in re.split(r'[\s\n]+', original_text.strip()) if p]
    
    for idx, (sy, ey) in enumerate(lines_y):
        sy, ey = max(0, sy - 2), min(crop.shape[0], ey + 2)
        new_y1, new_y2 = y1 + sy, y1 + ey
        new_token = token.copy()
        new_token["bbox"] = [x1, new_y1, x2, new_y2]
        new_token["poly"] = [[x1, new_y1], [x2, new_y1], [x2, new_y2], [x1, new_y2]]
        
        if idx < len(parts): new_token["text"] = parts[idx]
        else: new_token["text"] = "" 
        new_tokens.append(new_token)
        
    return new_tokens

def group_tokens_into_lines(tokens):
    tokens = sorted(tokens, key=lambda t: bbox_center(t["bbox"])[1])
    lines = []
    med_h = median_token_height(tokens)
    y_tolerance = med_h * 0.5 

    for token in tokens:
        placed = False
        tok_cy = bbox_center(token["bbox"])[1]

        for line in lines:
            line_cy = float(np.median([bbox_center(t["bbox"])[1] for t in line]))
            if abs(tok_cy - line_cy) <= y_tolerance:
                line.append(token)
                placed = True
                break

        if not placed:
            lines.append([token])

    out = []
    for idx, line in enumerate(lines):
        line.sort(key=lambda t: t["bbox"][0])
        out.append({
            "line_index": idx, "tokens": line,
            "text": " ".join(t["text"] for t in line).strip(),
            "bbox": bbox_union([t["bbox"] for t in line]),
        })

    out.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return out

def string_similarity(a, b):
    a, b = clean_for_match(a), clean_for_match(b)
    if not a or not b: return 0.0
    if f" {a} " in f" {b} " or f" {b} " in f" {a} ": 
        return 100.0 * min(len(a), len(b)) / max(1, max(len(a), len(b)))
    a_no_space, b_no_space = a.replace(" ", ""), b.replace(" ", "")
    if a_no_space and b_no_space:
        if a_no_space == b_no_space: return 100.0
        if a_no_space in b_no_space or b_no_space in a_no_space:
            ratio = min(len(a_no_space), len(b_no_space)) / max(len(a_no_space), len(b_no_space))
            if ratio >= 0.6: return 95.0
    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

def load_ontology(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k).strip(): [str(a).strip() for a in v if str(a).strip()] for k, v in data.items() if isinstance(v, list)}
    except Exception as e:
        print(f"Warning: Could not load ontology ({e})")
        return {}

def build_alias_index(test_ontology):
    alias_index = []
    for canonical, aliases in test_ontology.items():
        for alias in aliases:
            alias_clean = clean_for_match(alias)
            if alias_clean: alias_index.append((canonical, alias_clean))
    return alias_index

def identify_columns_robust(tokens):
    candidates = []
    for t in tokens:
        txt = clean_for_match(t["text"])
        best_col, best_score = None, 0
        for col_type, keywords in COLUMN_KEYWORDS.items():
            for kw in keywords:
                score = 100 if kw in txt else string_similarity(kw, txt)
                if score > 80 and score > best_score:
                    best_score = score
                    best_col = col_type
        if best_col:
            candidates.append({"token": t, "col": best_col, "score": best_score})
            
    candidates.sort(key=lambda c: bbox_center(c["token"]["bbox"])[0])

    best_group, best_unique_cols = [], 0
    for c1 in candidates:
        cy1 = bbox_center(c1["token"]["bbox"])[1]
        current_group, current_cols = [], set()
        for c2 in candidates:
            cy2 = bbox_center(c2["token"]["bbox"])[1]
            if abs(cy1 - cy2) <= 30:
                if c2["col"] == "IGNORE_WALL":
                    too_close = any(c["col"] == "IGNORE_WALL" and abs(bbox_center(c["token"]["bbox"])[0] - bbox_center(c2["token"]["bbox"])[0]) < 50 for c in current_group)
                    if not too_close:
                        current_group.append(c2)
                        current_cols.add(c2["col"])
                else:
                    existing = next((x for x in current_group if x["col"] == c2["col"]), None)
                    if existing:
                        if existing["score"] < 85 and c2["score"] >= 95:
                            current_group.remove(existing)
                            current_group.append(c2)
                    else:
                        current_group.append(c2)
                        current_cols.add(c2["col"])
                    
        if len(current_cols) > best_unique_cols:
            best_unique_cols, best_group = len(current_cols), current_group
            
    if best_unique_cols < 2: return -1, {}, []
        
    col_centers, wall_idx = [], 0
    for c in best_group:
        col_name = c["col"]
        if col_name == "IGNORE_WALL":
            col_name = f"IGNORE_WALL_{wall_idx}"
            wall_idx += 1
        col_centers.append((col_name, bbox_center(c["token"]["bbox"])[0]))
        
    matched_header_tokens = [c["token"] for c in best_group]
    if not any(c[0] == "TEST_NAME" for c in col_centers):
        col_centers.append(("TEST_NAME", 0.0))
        
    sorted_cols = sorted(col_centers, key=lambda x: x[1])
    col_boundaries = {}
    
    for idx, (col_name, center_x) in enumerate(sorted_cols):
        x_min = 0 if idx == 0 else (center_x + sorted_cols[idx-1][1]) / 2
        x_max = float('inf') if idx == len(sorted_cols) - 1 else (center_x + sorted_cols[idx+1][1]) / 2
        if not col_name.startswith("IGNORE_WALL"):
            col_boundaries[col_name] = (x_min, x_max)
            
    data_start_y = max([t["bbox"][3] for t in matched_header_tokens]) + 5 if matched_header_tokens else 0
    return data_start_y, col_boundaries, matched_header_tokens

def parse_ref_range(ref_str):
    if not ref_str: return None, None, "", None, None
    ref_str_clean = ref_str.replace('(', '').replace(')', '').strip()
    ref_min, ref_max, unit_fallback = None, None, ""
    op_min, op_max = None, None
    num_pattern = r'[-+]?\d+(?:[\.,]\d+)?'
    
    range_match = re.search(fr'({num_pattern})\s*[-~=]\s*({num_pattern})', ref_str_clean)
    if range_match:
        ref_min = str_to_float(range_match.group(1))
        ref_max = str_to_float(range_match.group(2))
        op_min, op_max = '>=', '<='
        unit_fallback = ref_str_clean[:range_match.start()] + ref_str_clean[range_match.end():]
    else:
        single_match = re.search(fr'([<>≤≥]|<=|>=)\s*({num_pattern})', ref_str_clean)
        if single_match:
            op = single_match.group(1)
            val = str_to_float(single_match.group(2))
            if op in ['<', '<=', '≤']:
                ref_max = val
                op_max = op
            else:
                ref_min = val
                op_min = op
            unit_fallback = ref_str_clean[:single_match.start()] + ref_str_clean[single_match.end():]
        else: unit_fallback = ref_str_clean

    if ref_min is not None and ref_max is not None and ref_min > ref_max:
        if ref_min > 0 and ref_max < 0: ref_max = abs(ref_max)
        if ref_min > ref_max: 
            ref_min, ref_max = ref_max, ref_min
            op_min, op_max = op_max, op_min

    unit_fallback = re.sub(r'^[-~,:;]+', '', unit_fallback.strip()).strip()
    return ref_min, ref_max, unit_fallback, op_min, op_max

def clean_value_string(val_str):
    val_str = val_str.strip()
    if not val_str: return val_str
    val_str = re.sub(r'(?:\s+(?:[1lIihHL]|tang|giam|cao|thap|tăng|giảm|thấp))+$', '', val_str, flags=re.IGNORECASE)
    val_str = re.sub(r'\s*[↑↓←→\-\—\|]+$', '', val_str)
    m = re.match(r'^([<>]?)\s*([-+]?\d+(?:[\.,]\d+)*)$', val_str)
    if m: return f"{m.group(1)}{m.group(2)}" if m.group(1) else m.group(2)
    return val_str.strip()

def calculate_status(raw_value_str, ref_min, ref_max, op_min=None, op_max=None):
    raw_lower = str(raw_value_str).lower()
    if re.search(r'\s+(h|cao|tăng|tang|↑|←)$', raw_lower): return "High"
    if re.search(r'\s+(l|thấp|thap|giảm|giam|↓|→)$', raw_lower): return "Low"

    try:
        val = str_to_float(re.search(r'[-+]?\d+(?:[\.,]\d+)?', normalize_text(raw_value_str)).group())
        if ref_min is not None:
            if op_min == '>' and val <= ref_min: return "Low"
            elif val < ref_min: return "Low"
        if ref_max is not None:
            if op_max == '<' and val >= ref_max: return "High"
            elif val > ref_max: return "High"
        if ref_min is not None or ref_max is not None: return "Normal"
    except: pass

    return None

def find_test_name_with_score(text, aliases, available_canonicals=None):
    t = clean_for_match(text)
    t_no_space = t.replace(" ", "")
    if not t_no_space: return None, 0
        
    best_canonical, best_score = None, 0
    for canonical, alias in aliases:
        if available_canonicals is not None and canonical not in available_canonicals:
            continue
        score = 0
        alias_no_space = alias.replace(" ", "")
        
        if alias == t: score = 1000 + len(alias)
        elif alias_no_space == t_no_space: score = 500 + len(alias_no_space)
        elif f" {alias} " in f" {t} ": score = 200 + len(alias)
        else:
            sim = difflib.SequenceMatcher(None, alias, t).ratio() * 100.0
            if alias_no_space in t_no_space and len(t_no_space) > 0:
                ratio = len(alias_no_space) / len(t_no_space)
                if ratio >= 0.5: sim = max(sim, 85.0 + 10 * ratio)
            score = sim
            
        if score > best_score and (score >= 80 or score > 200):
            best_score, best_canonical = score, canonical
            
    return best_canonical, best_score

def process_single_image_core(image_path, aliases, unit_ontology, debug_img_path=None, scanned_img_path=None):
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"   ❌ Lỗi: Không thể đọc ảnh {image_path}")
        return {"extracted_data": [], "raw_ocr": [], "used_gpu": False, "timings": {}}

    t0_paddle = time.time()
    warped_img, tokens, used_gpu = predict_unified(image_path)
    t_paddle = time.time() - t0_paddle

    if warped_img is None:
        print(f"   ⚠️ Lỗi trích xuất ảnh Unwarp, fallback lại ảnh gốc: {image_path}")
        warped_img = original_img.copy()
    else:
        print(f"   ✅ Paddle Unified Engine xử lý thành công: {image_path}")

    if scanned_img_path:
        save_image(scanned_img_path, warped_img)

    if not tokens:
        return {"extracted_data": [], "raw_ocr": [], "used_gpu": used_gpu, "timings": {}}

    med_h = median_token_height(tokens)
    split_tokens = []
    for t in tokens:
        split_tokens.extend(split_tall_token_if_needed(warped_img, t, med_h))
        
    filtered_tokens = []
    for t in split_tokens:
        h = bbox_height(t["bbox"])
        if h > med_h * 4.0:
            continue
        filtered_tokens.append(t)
    tokens = filtered_tokens

    # Làm sạch chuỗi đặc thù của PaddleOCR
    for t in tokens:
        t["tag"] = "OTHER"
        t["final_data_text"] = resolve_data_text(t["text"])

    t0_ext = time.time()
    data_start_y_ignore, col_boundaries, matched_header_tokens = identify_columns_robust(tokens)

    if not col_boundaries:
        extracted_data = []
    else:
        for t in matched_header_tokens: t["tag"] = "HEADER"
        
        header_thresholds = {}
        global_min_y = float('inf')
        for t in matched_header_tokens:
            cx, cy = bbox_center(t["bbox"])
            for col_name, (xmin, xmax) in col_boundaries.items():
                if xmin <= cx <= xmax:
                    header_thresholds[col_name] = cy
                    global_min_y = min(global_min_y, cy)
                    break
        if global_min_y == float('inf'): global_min_y = 0

        test_name_xmin, test_name_xmax = col_boundaries.get("TEST_NAME", (0, 0))
        test_name_tokens = []
        data_tokens = []

        for t in tokens:
            if t.get("tag") == "HEADER": continue
            
            cx, cy = bbox_center(t["bbox"])
            x1 = t["bbox"][0]

            col_name_of_t = None
            for col_name, (xmin, xmax) in col_boundaries.items():
                if xmin <= cx <= xmax:
                    col_name_of_t = col_name
                    break
            
            threshold_y = header_thresholds.get(col_name_of_t, global_min_y)

            if cy <= threshold_y: continue

            if test_name_xmin <= x1 <= test_name_xmax or test_name_xmin <= cx <= test_name_xmax:
                test_name_tokens.append(t)
            else:
                data_tokens.append(t)

        name_lines = group_tokens_into_lines(test_name_tokens)

        for line in name_lines:
            line_cy = np.mean([bbox_center(t["bbox"])[1] for t in line["tokens"]])
            line["cy"] = line_cy
            line["x1"] = min(t["bbox"][0] for t in line["tokens"])
            
            y_tol = med_h * 0.6
            line["has_data"] = any(abs(bbox_center(dt["bbox"])[1] - line_cy) < y_tol for dt in data_tokens)
            line["text"] = " ".join([t["text"] for t in line["tokens"]]).strip()

        merged_name_lines = []
        for line in name_lines:
            if not merged_name_lines:
                merged_name_lines.append(line)
            else:
                prev_line = merged_name_lines[-1]
                delta_y = line["cy"] - prev_line["cy"]
                
                is_close = delta_y < med_h * 2.5
                line_has_no_data = not line["has_data"]
                
                lower_text = line["text"].lower()
                is_category_header = bool(re.search(r'(sinh hóa|miễn dịch|huyết học|điện giải|nước tiểu|đông máu|vi sinh|huyết thanh|chức năng|tế bào|nội tiết)', lower_text))
                
                _, s = find_test_name_with_score(line["text"], aliases)
                is_distinct_test = (s > 80)
                
                if is_close and line_has_no_data and not is_category_header and not is_distinct_test:
                    prev_line["tokens"].extend(line["tokens"]) 
                    prev_line["cy"] = np.mean([bbox_center(t["bbox"])[1] for t in prev_line["tokens"]])
                    prev_line["x1"] = min(prev_line["x1"], line["x1"])
                    prev_line["has_data"] = prev_line["has_data"] or line["has_data"]
                    prev_line["text"] += " " + line["text"]
                else:
                    merged_name_lines.append(line)
                    
        name_lines = merged_name_lines

        anchors = []
        for line in name_lines:
            line_tokens = line["tokens"]
            raw_name_p = " ".join([t["text"] for t in line_tokens]).strip()
            best_c, best_s = find_test_name_with_score(raw_name_p, aliases)

            cy = float(np.mean([bbox_center(t["bbox"])[1] for t in line_tokens]))
            for t in line_tokens: t["tag"] = "TEST_NAME"
            anchors.append({
                "canonical": best_c if best_c else raw_name_p,
                "match_score": best_s,
                "center_y": cy,
                "matched_data": {"VALUE": None, "UNIT": None, "REF_RANGE": None},
                "raw_tokens": list(line_tokens),
                "raw_p": raw_name_p
            })

        if anchors:
            valid_xs = [min(t["bbox"][0] for t in a["raw_tokens"]) for a in anchors if a["match_score"] > 0]
            base_x = float(np.median(valid_xs)) if valid_xs else 0.0
            
            filtered_anchors = []
            for a in anchors:
                a_min_x = min(t["bbox"][0] for t in a["raw_tokens"])
                
                if valid_xs and a_min_x < base_x - 60 and a["match_score"] == 0:
                    for t in a["raw_tokens"]: t["tag"] = "OTHER"
                    continue
                    
                raw_text = a["canonical"]
                letters = re.sub(r'[^a-zA-Z]', '', raw_text)
                if a["match_score"] == 0 and len(letters) < 3:
                    for t in a["raw_tokens"]: t["tag"] = "OTHER"
                    continue
                    
                filtered_anchors.append(a)
            anchors = filtered_anchors

        if anchors:
            cols_data = {"VALUE": [], "UNIT": [], "REF_RANGE": []}
            for t in data_tokens:
                cx = bbox_center(t["bbox"])[0]
                for col_name, (xmin, xmax) in col_boundaries.items():
                    if xmin <= cx <= xmax and col_name in cols_data:
                        cols_data[col_name].append(t)
                        break
            
            col_candidates = {"VALUE": [], "UNIT": [], "REF_RANGE": []}
            for col_name, t_list in cols_data.items():
                if not t_list: continue
                t_list.sort(key=lambda x: bbox_center(x["bbox"])[1])
                groups = []
                for t in t_list:
                    placed = False
                    t_cy = bbox_center(t["bbox"])[1]
                    for g in groups:
                        g_cy = np.mean([bbox_center(x["bbox"])[1] for x in g])
                        if abs(t_cy - g_cy) < med_h * 0.4:
                            g.append(t)
                            placed = True
                            break
                    if not placed:
                        groups.append([t])
                
                for g in groups:
                    g.sort(key=lambda x: x["bbox"][0])
                    col_candidates[col_name].append({
                        "center_y": np.mean([bbox_center(x["bbox"])[1] for x in g]),
                        "tokens": g,
                        "text": " ".join([x.get("final_data_text", x["text"]) for x in g])
                    })

            for a in anchors:
                a["current_y"] = a["center_y"]

            target_cols = ["VALUE", "UNIT", "REF_RANGE"]
            target_cols.sort(key=lambda col: col_boundaries[col][0] if col in col_boundaries else float('inf'))

            for col_name in target_cols:
                candidates = col_candidates[col_name]
                if not candidates: continue
                
                n = len(anchors)
                m = len(candidates)
                dp = np.full((n + 1, m + 1), float('inf'))
                dp[0][0] = 0.0
                trace = np.zeros((n + 1, m + 1), dtype=int)
                
                gap_penalty = (med_h * 2.0) ** 2 
                max_y_dist = med_h * 1.0 if col_name == "VALUE" else med_h * 1.5
                
                for i in range(n + 1):
                    for j in range(m + 1):
                        if i > 0:
                            cost = dp[i-1][j] + gap_penalty
                            if cost < dp[i][j]:
                                dp[i][j] = cost
                                trace[i][j] = 1
                        if j > 0:
                            cost = dp[i][j-1] + gap_penalty
                            if cost < dp[i][j]:
                                dp[i][j] = cost
                                trace[i][j] = 2
                        if i > 0 and j > 0:
                            dist = abs(anchors[i-1]["current_y"] - candidates[j-1]["center_y"])
                            if dist <= max_y_dist: 
                                cost = dp[i-1][j-1] + dist ** 2
                                if cost < dp[i][j]:
                                    dp[i][j] = cost
                                    trace[i][j] = 0
                                    
                i, j = n, m
                matches = []
                while i > 0 or j > 0:
                    if trace[i][j] == 0 and i > 0 and j > 0:
                        matches.append((i-1, j-1))
                        i -= 1
                        j -= 1
                    elif trace[i][j] == 1 and i > 0:
                        i -= 1
                    elif trace[i][j] == 2 and j > 0:
                        j -= 1
                    else:
                        break
                        
                matches.reverse()
                
                for a_idx, c_idx in matches:
                    anchors[a_idx]["matched_data"][col_name] = candidates[c_idx]
                    for t in candidates[c_idx]["tokens"]:
                        t["tag"] = "DATA"
                        anchors[a_idx]["raw_tokens"].append(t)
                    anchors[a_idx]["current_y"] = candidates[c_idx]["center_y"]

            valid_anchors = []
            for a in anchors:
                row_data = {"VALUE": "", "UNIT": "", "REF_RANGE": ""}
                for col in ["VALUE", "UNIT", "REF_RANGE"]:
                    if a["matched_data"][col]:
                        row_data[col] = a["matched_data"][col]["text"]

                val_is_empty = not row_data["VALUE"].strip()
                if not val_is_empty and not re.search(r'\d', row_data["VALUE"]):
                    val_is_empty = True

                if val_is_empty:
                    combined_fallback = f'{row_data["UNIT"]} {row_data["REF_RANGE"]}'.strip()
                    match = re.match(r'^([<>]?\s*[-+]?\d+(?:[\.,]\d+)*)\s*(.*)', combined_fallback)
                    if match:
                        possible_val = match.group(1).strip()
                        remainder = match.group(2).strip()
                        if possible_val == "10" and (remainder.startswith("^") or remainder.startswith("*") or remainder.startswith("mũ")):
                            pass 
                        else:
                            row_data["VALUE"] = combined_fallback
                            row_data["UNIT"] = ""
                            row_data["REF_RANGE"] = ""

                ref_min, ref_max, fallback_unit, op_min, op_max = parse_ref_range(row_data["REF_RANGE"])
                
                final_unit = fix_unit_typos(row_data["UNIT"])
                fallback_unit = fix_unit_typos(fallback_unit)

                if ref_min is None and ref_max is None and final_unit and re.search(r'\d', final_unit):
                    ex_min, ex_max, rem_unit, ex_op_min, ex_op_max = parse_ref_range(final_unit)
                    if ex_min is not None or ex_max is not None:
                        ref_min, ref_max = ex_min, ex_max
                        op_min, op_max = ex_op_min, ex_op_max
                        final_unit = fix_unit_typos(rem_unit)

                if ref_min is None and ref_max is None and row_data["VALUE"]:
                    ex_min, ex_max, rem_val, ex_op_min, ex_op_max = parse_ref_range(row_data["VALUE"])
                    if ex_min is not None or ex_max is not None:
                        ref_min, ref_max = ex_min, ex_max
                        op_min, op_max = ex_op_min, ex_op_max
                        row_data["VALUE"] = rem_val

                if row_data["VALUE"] and re.search(r'[a-zA-Z%‰/]', row_data["VALUE"]):
                    match = re.match(r'^([<>]?\s*[-+]?\d+(?:[\.,]\d+)*)\s*(.*)', row_data["VALUE"])
                    if match and not final_unit:
                        row_data["VALUE"] = match.group(1).strip()
                        final_unit = fix_unit_typos(match.group(2).strip("() "))

                if not final_unit and fallback_unit:
                    final_unit = fallback_unit

                final_unit = fix_unit_typos(final_unit)
                cleaned_value = clean_value_string(row_data["VALUE"])
                numeric_value = extract_strict_number(cleaned_value)

                if numeric_value is None: continue

                a["raw_tokens"].sort(key=lambda t: t["bbox"][0])
                
                a["final_value"] = numeric_value
                a["final_unit"] = final_unit
                a["final_ref_min"] = ref_min
                a["final_ref_max"] = ref_max
                a["final_status"] = calculate_status(row_data["VALUE"], ref_min, ref_max, op_min, op_max)
                
                a["target_canonical"] = a["canonical"]
                valid_anchors.append(a)

            all_canons = set(canon for canon, alias in aliases)
            while True:
                canon_to_anchors = {}
                for a in valid_anchors:
                    c = a["target_canonical"]
                    if c in all_canons:
                        if c not in canon_to_anchors: canon_to_anchors[c] = []
                        canon_to_anchors[c].append(a)
                        
                duplicates_exist = False
                for c, anchor_list in canon_to_anchors.items():
                    if len(anchor_list) > 1:
                        duplicates_exist = True
                        used_canons_current = set(a["target_canonical"] for a in valid_anchors if a["target_canonical"] in all_canons)
                        unused_canons_for_tie = all_canons - used_canons_current
                        
                        for a in anchor_list:
                            _, s_p = find_test_name_with_score(a["raw_p"], aliases, unused_canons_for_tie)
                            a["fallback_score"] = s_p

                            unit_bonus = 0
                            if "PERCENT" in a["target_canonical"] and "%" in a["final_unit"]: unit_bonus = 1000
                            elif "ABS" in a["target_canonical"] and re.search(r'[lL]', a["final_unit"]): unit_bonus = 1000
                            a["unit_bonus"] = unit_bonus

                        anchor_list.sort(key=lambda x: (x.get("unit_bonus", 0), x["match_score"], -x.get("fallback_score", 0), -x["center_y"]), reverse=True)
                        used_canons = set(a["target_canonical"] for a in valid_anchors if a["target_canonical"] in all_canons)
                        unused_canons = all_canons - used_canons
                        
                        for loser in anchor_list[1:]:
                            c_p, s_p = find_test_name_with_score(loser["raw_p"], aliases, unused_canons)
                            if c_p:
                                loser["target_canonical"], loser["match_score"] = c_p, s_p
                            else:
                                loser["target_canonical"], loser["match_score"] = loser["raw_p"], 0
                                
                            if loser["target_canonical"] in all_canons:
                                unused_canons.discard(loser["target_canonical"])
                                
                if not duplicates_exist:
                    break

            extracted_data = []
            for a in valid_anchors:
                if USE_ONTOLOGY_FILTER and a["target_canonical"] not in all_canons:
                    continue
                    
                final_unit_corrected = correct_unit_by_ontology(a["final_unit"], a["target_canonical"], unit_ontology)
                    
                raw_line_texts = []
                for t in a["raw_tokens"]:
                    if t.get("tag") == "TEST_NAME":
                        raw_line_texts.append(t["text"])
                    else:
                        raw_line_texts.append(t.get("final_data_text", t["text"]))

                raw_line_str = " ".join(raw_line_texts)

                extracted_data.append({
                    "test_name": a["target_canonical"],
                    "value": a["final_value"],
                    "unit": final_unit_corrected,
                    "ref_range": {"ref_min": a["final_ref_min"], "ref_max": a["final_ref_max"]},
                    "status": a["final_status"],
                    "raw_text_line": raw_line_str
                })
                
    t_extraction = time.time() - t0_ext

    if debug_img_path is not None:
        debug_img = warped_img.copy()
        h, w = debug_img.shape[:2]
        
        for t in tokens:
            if t.get("tag") == "HEADER": t["final_tag"] = "HEADER"
            else: t["final_tag"] = "IGNORED"

        if extracted_data:
            for a in valid_anchors:
                if USE_ONTOLOGY_FILTER and a["target_canonical"] not in all_canons: continue
                for t in a["raw_tokens"]:
                    cx = bbox_center(t["bbox"])[0]
                    if test_name_xmin <= cx <= test_name_xmax: t["final_tag"] = "TEST_NAME_FINAL"
                    else: t["final_tag"] = "DATA_FINAL"

        for col_name, (x_min, x_max) in col_boundaries.items():
            x_min_int = max(0, int(x_min))
            x_max_int = w if x_max == float('inf') else min(w, int(x_max))
            cv2.rectangle(debug_img, (x_min_int, 0), (x_max_int, h), (0, 100, 0), 1)
            cv2.putText(debug_img, col_name, (x_min_int + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if col_name in header_thresholds:
                thresh_y = int(header_thresholds[col_name])
                if thresh_y > 0: cv2.line(debug_img, (x_min_int, thresh_y), (x_max_int, thresh_y), (0, 255, 255), 1)

        for t in tokens:
            b = t["bbox"]
            if not b: continue
            x1, y1, x2, y2 = map(int, b)
            
            ftag = t.get("final_tag", "IGNORED")
            if ftag == "HEADER": color = (255, 0, 255)   
            elif ftag == "TEST_NAME_FINAL": color = (0, 0, 255)     
            elif ftag == "DATA_FINAL": color = (0, 255, 0)
            else: color = (150, 150, 150)
                
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_img, strip_accents(t["text"]), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(str(debug_img_path), debug_img)

    return {
        "extracted_data": extracted_data,
        "raw_ocr": [{"text": t["text"], "bbox": t["bbox"], "tag": t.get("tag", "OTHER")} for t in tokens],
        "used_gpu": used_gpu,
        "timings": {
            "paddle_end_to_end": t_paddle,
            "extraction": t_extraction
        }
    }

# =========================================================
# GIAO DIỆN API (DÙNG CHO OCR_API.PY)
# =========================================================
def run_pipeline_on_image(image_path, aliases, unit_ontology, output_dir):
    start_time = time.time()
    base_name = Path(image_path).stem
    
    print(f"-> Đang chạy Pipeline OCR End-to-End: {image_path}")
    
    scanned_path = Path(output_dir) / f"{base_name}_{OUTPUT_SCANNED_IMG}"
    debug_path = Path(output_dir) / f"{base_name}_{OUTPUT_DEBUG_IMG}"
    json_path = Path(output_dir) / f"{base_name}_{OUTPUT_JSON}"
    clean_json_path = Path(output_dir) / f"{base_name}_{OUTPUT_CLEAN_JSON}"
    
    result_dict = process_single_image_core(image_path, aliases, unit_ontology, str(debug_path), str(scanned_path))
    extracted_data = result_dict["extracted_data"]
    used_gpu = result_dict.get("used_gpu", True)
    timings = result_dict.get("timings", {})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
        
    clean_extracted_data = [{k: v for k, v in item.items() if k != "raw_text_line"} for item in extracted_data]
    with open(clean_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_extracted_data, f, ensure_ascii=False, indent=4)
            
    end_time = time.time()
    processing_time = end_time - start_time
    gpu_status = "GPU" if used_gpu else "CPU Fallback"
    
    print(f"-> Đã trích xuất xong {len(extracted_data)} chỉ số OCR cho ảnh '{base_name}' ({gpu_status} - {processing_time:.2f}s)!")
    return clean_extracted_data, processing_time, used_gpu, timings

def run_end_to_end_pipeline(input_path):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    test_ontology = load_ontology(ONTOLOGY_JSON)
    aliases = build_alias_index(test_ontology)
    unit_ontology = load_unit_ontology(ONTOLOGY_UNITS_JSON)

    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        res, p_time, used_gpu, timings = run_pipeline_on_image(str(input_path_obj), aliases, unit_ontology, OUTPUT_DIR)
        return res
        
    elif input_path_obj.is_dir():
        all_results = {}
        gpu_times = []
        paddle_times = []
        ext_times = []
        
        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = [f for f in input_path_obj.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        
        print(f"==> Tìm thấy {len(image_files)} ảnh trong thư mục {input_path}")
        for i, img_file in enumerate(image_files, 1):
            print(f"\n--- Đang xử lý ảnh {i}/{len(image_files)}: {img_file.name} ---")
            res, p_time, used_gpu, timings = run_pipeline_on_image(str(img_file), aliases, unit_ontology, OUTPUT_DIR)
            all_results[img_file.name] = res
            
            if used_gpu and timings:
                gpu_times.append(p_time)
                paddle_times.append(timings.get("paddle_end_to_end", 0))
                ext_times.append(timings.get("extraction", 0))
            
        print(f"\n==> Đã xử lý xong toàn bộ {len(image_files)} ảnh trong thư mục!")
        if gpu_times:
            n = len(gpu_times)
            avg_total = sum(gpu_times) / n
            avg_paddle = sum(paddle_times) / n
            avg_ext = sum(ext_times) / n
            
            print(f"\n" + "="*55)
            print(f" BÁO CÁO THỜI GIAN XỬ LÝ TRUNG BÌNH (Trên {n} phiếu chạy GPU)")
            print(f" " + "="*55)
            print(f" - Đọc text & Unwarp (Paddle End-to-End) : {avg_paddle:>6.2f}s")
            print(f" - Trích xuất Layout (Bóc tách dữ liệu)  : {avg_ext:>6.2f}s")
            print(f" -------------------------------------------------------")
            print(f" => TỔNG THỜI GIAN TOÀN BỘ PIPELINE     : {avg_total:>6.2f}s / phiếu")
            print(f" " + "="*55 + "\n")
        else:
            print(f"==> Tất cả các phiếu đều fallback sang CPU (Không có phiếu nào chạy hoàn toàn trên GPU để tính trung bình).")
        return all_results
        
    else:
        print(f"❌ Lỗi: Không tìm thấy file hoặc thư mục {input_path}")
        return None

if __name__ == "__main__":
    import sys
    
    input_folder = "input"
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
        
    input_path_obj = Path(input_folder)
    
    if not input_path_obj.exists():
        print(f"[*] Đã tự động tạo thư mục '{input_folder}'.")
        print(f"[*] Vui lòng copy các ảnh cần OCR vào thư mục '{input_folder}' rồi bấm Run lại.")
        input_path_obj.mkdir(parents=True, exist_ok=True)
    else:
        print(f"\n========== BẮT ĐẦU CHẠY OCR TỪ: {input_folder} ==========")
        results = run_end_to_end_pipeline(input_folder)
        print("========== HOÀN TẤT ==========")
        print(f"Kiểm tra kết quả trong thư mục: '{OUTPUT_DIR}'")
