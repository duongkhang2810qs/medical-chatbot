import os
import json
import re
import unicodedata
import difflib
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
from PIL import Image

try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
except ImportError:
    print("Lỗi: Chưa cài đặt VietOCR. Vui lòng chạy: pip install vietocr")
    raise

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
# OCR ENGINES INIT
# =========================================================

print("Khởi tạo PaddleOCR (Detection & Recognition)...")
ocr_gpu = PaddleOCR(
    device="gpu",
    lang="vi",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    ocr_version="PP-OCRv5"
)

ocr_cpu = PaddleOCR(
    device="cpu",
    lang="vi",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    ocr_version="PP-OCRv5"
)

print("Khởi tạo VietOCR (Recognition)...")
vietocr_config = Cfg.load_config_from_name('vgg_transformer') 
vietocr_config['cnn']['pretrained'] = False
vietocr_config['device'] = 'cuda:0' if paddle.device.is_compiled_with_cuda() else 'cpu'
vietocr_config['predictor']['beamsearch'] = False
vietocr_predictor = Predictor(vietocr_config)

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

def crop_image_by_poly(img, poly):
    try:
        is_4_points = (
            isinstance(poly, (list, np.ndarray)) 
            and len(poly) == 4 
            and all(isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2 for p in poly)
        )
        if is_4_points:
            pts = np.array(poly, dtype="float32")
            if pts.shape == (4, 2):
                (tl, tr, br, bl) = pts
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                if maxWidth > 0 and maxHeight > 0:
                    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(pts, dst)
                    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
                    return warped
    except Exception: pass

    b = polygon_to_bbox(poly)
    if b is None: return None
    x1, y1, x2, y2 = map(int, b)
    return img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]

# =========================================================
# THUẬT TOÁN LÀM PHẲNG VÀ KÉO NGHIÊNG
# =========================================================

def get_hough_deskew_angles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=100, minLineLength=150, maxLineGap=20)
    
    h_angles = []
    v_angles = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            if abs(angle) < 15:
                h_angles.append(angle)
            elif abs(angle - 90) < 15:
                v_angles.append(angle - 90)
            elif abs(angle + 90) < 15:
                v_angles.append(angle + 90)
                
    final_h = float(np.median(h_angles)) if h_angles else 0.0
    final_v = float(np.median(v_angles)) if v_angles else final_h
    return final_h, final_v

def get_robust_deskew_angles(boxes, image):
    h_angle, v_angle = get_hough_deskew_angles(image)
    
    ocr_h_angles = []
    for box in boxes:
        if len(box) != 4: continue
        tl, tr, br, bl = box
        w_top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
        h_left = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
        
        if h_left == 0: continue
        if w_top > 60 and w_top > h_left * 3.0:
            a_top = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
            a_bot = math.degrees(math.atan2(br[1] - bl[1], br[0] - bl[0]))
            avg_a = (a_top + a_bot) / 2.0
            if abs(avg_a) < 15:
                ocr_h_angles.append(avg_a)
                
    if h_angle == 0.0 and ocr_h_angles:
        h_angle = float(np.median(ocr_h_angles))
        
    if v_angle == 0.0:
        v_angle = h_angle
        
    return h_angle, v_angle

def deskew_and_deshear_image(image, angle_h, angle_v):
    if abs(angle_h) < 0.05 and abs(angle_v - angle_h) < 0.5:
        return image
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    M_rot = cv2.getRotationMatrix2D(center, angle_h, 1.0)
    cos = np.abs(M_rot[0, 0])
    sin = np.abs(M_rot[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    M_rot[0, 2] += (nW / 2) - center[0]
    M_rot[1, 2] += (nH / 2) - center[1]
    rotated = cv2.warpAffine(image, M_rot, (nW, nH), borderValue=(255, 255, 255))
    
    skew_angle = angle_v - angle_h
    if abs(skew_angle) >= 0.5 and abs(skew_angle) < 20: 
        tan_theta = math.tan(math.radians(skew_angle))
        M_shear = np.float32([[1, tan_theta, 0], [0, 1, 0]])
        offset_x = 0
        if tan_theta < 0:
            offset_x = abs(nH * tan_theta)
            M_shear[0, 2] = offset_x
        nW_shear = int(nW + abs(nH * tan_theta))
        sheared = cv2.warpAffine(rotated, M_shear, (nW_shear, nH), borderValue=(255, 255, 255))
        return sheared
    return rotated

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
    u = re.sub(r'^(?:[HLhl↑↓←→]\s+)+', '', u.strip())
    u = re.sub(r'10[\{\'\"]?2/?L', '10^12/L', u, flags=re.IGNORECASE)
    u = re.sub(r'10\%/?L', '10^9/L', u, flags=re.IGNORECASE)
    u = re.sub(r'10\^9/?L', '10^9/L', u, flags=re.IGNORECASE)
    u = re.sub(r'10[3\?7\*]/uL', '10^3/uL', u, flags=re.IGNORECASE)
    u = re.sub(r'10\^?3/?uL', '10^3/uL', u, flags=re.IGNORECASE)
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
    if not raw_unit:
        return ""
    allowed_units = unit_ontology.get(test_name, [])
    if not allowed_units:
        return raw_unit
    best_unit = raw_unit
    best_score = 0
    for au in allowed_units:
        score = unit_similarity(raw_unit, au)
        if score > best_score:
            best_score = score
            best_unit = au
    if best_score >= 25.0:
        return best_unit
    return raw_unit

# =========================================================
# PADDLE OCR BÓC TÁCH KẾT QUẢ
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

def predict_tokens(image_or_path):
    input_data = str(image_or_path) if isinstance(image_or_path, (str, Path)) else image_or_path
    try:
        results = ocr_gpu.predict(input_data)
    except Exception as e:
        if is_gpu_oom_error(e):
            print("   Cảnh báo: GPU OOM, tự động chuyển sang CPU...")
            results = ocr_cpu.predict(input_data)
        else:
            raise

    tokens = []
    if results:
        for res in results:
            if res: tokens.extend(extract_tokens_from_result(res))
    
    return [t for t in tokens if t.get("bbox") is not None]

def resolve_data_text(paddle_txt, viet_txt, p_score, v_score):
    p_str = str(paddle_txt).strip()
    v_str = str(viet_txt).strip()
    if not p_str: return v_str
    if not v_str: return p_str
    if p_str == v_str: return p_str

    p_has_dec = '.' in p_str or ',' in p_str
    v_has_dec = '.' in v_str or ',' in v_str
    p_is_range = '-' in p_str or '~' in p_str
    
    if v_has_dec and not p_has_dec and not p_is_range and v_score > 0.4: return v_str
    if ('%' in v_str or '‰' in v_str) and ('%' not in p_str and '‰' not in p_str) and v_score > 0.5: return v_str
    if v_score > (p_score or 0) + 0.2: return v_str
    return p_str

# =========================================================
# XỬ LÝ LÕI TRÍCH XUẤT
# =========================================================

def split_tall_token_if_needed(img, token, median_h):
    x1, y1, x2, y2 = map(int, token["bbox"])
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    if h < 2.4 * median_h or h < 1.2 * w:
        return [token]
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
        if idx < len(parts):
            new_token["text"] = parts[idx]
        else:
            new_token["text"] = "" 
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
        v_txt = t.get("vietocr_text", t["text"])
        txt = clean_for_match(v_txt)
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
    if not ref_str: return None, None, ""
    ref_str_clean = ref_str.replace('(', '').replace(')', '').strip()
    ref_min, ref_max, unit_fallback = None, None, ""
    num_pattern = r'[-+]?\d+(?:[\.,]\d+)?'
    
    range_match = re.search(fr'({num_pattern})\s*[-~]\s*({num_pattern})', ref_str_clean)
    if range_match:
        ref_min = str_to_float(range_match.group(1))
        ref_max = str_to_float(range_match.group(2))
        unit_fallback = ref_str_clean[:range_match.start()] + ref_str_clean[range_match.end():]
    else:
        single_match = re.search(fr'([<>])\s*({num_pattern})', ref_str_clean)
        if single_match:
            if single_match.group(1) == '<': ref_max = str_to_float(single_match.group(2))
            else: ref_min = str_to_float(single_match.group(2))
            unit_fallback = ref_str_clean[:single_match.start()] + ref_str_clean[single_match.end():]
        else: unit_fallback = ref_str_clean

    if ref_min is not None and ref_max is not None and ref_min > ref_max:
        if ref_min > 0 and ref_max < 0: ref_max = abs(ref_max)
        if ref_min > ref_max: ref_min, ref_max = ref_max, ref_min

    unit_fallback = re.sub(r'^[-~,:;]+', '', unit_fallback.strip()).strip()
    return ref_min, ref_max, unit_fallback

def clean_value_string(val_str):
    val_str = val_str.strip()
    if not val_str: return val_str
    val_str = re.sub(r'(?:\s+(?:[1lIihHL]|tang|giam|cao|thap|tăng|giảm|thấp))+$', '', val_str, flags=re.IGNORECASE)
    val_str = re.sub(r'\s*[↑↓←→\-\—\|]+$', '', val_str)
    m = re.match(r'^([<>]?)\s*([-+]?\d+(?:[\.,]\d+)*)$', val_str)
    if m: return f"{m.group(1)}{m.group(2)}" if m.group(1) else m.group(2)
    return val_str.strip()

def calculate_status(raw_value_str, ref_min, ref_max):
    try:
        val = str_to_float(re.search(r'[-+]?\d+(?:[\.,]\d+)?', normalize_text(raw_value_str)).group())
        if ref_min is not None and val < ref_min: return "Low"
        if ref_max is not None and val > ref_max: return "High"
        if ref_min is not None or ref_max is not None: return "Normal"
    except: pass
    raw_lower = raw_value_str.lower()
    if re.search(r'\s+(h|cao|tăng|tang|↑|←)$', raw_lower): return "High"
    if re.search(r'\s+(l|thấp|thap|giảm|giam|↓|→)$', raw_lower): return "Low"
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

def extract_table_data_anchored(tokens, data_start_y, col_boundaries, aliases, med_h, unit_ontology):
    test_name_xmin, test_name_xmax = col_boundaries.get("TEST_NAME", (0, 0))
    test_name_tokens, data_tokens = [], []

    for t in tokens:
        if bbox_center(t["bbox"])[1] <= data_start_y: continue
        cx = bbox_center(t["bbox"])[0]
        x1 = t["bbox"][0]
        if test_name_xmin <= x1 <= test_name_xmax or test_name_xmin <= cx <= test_name_xmax:
            test_name_tokens.append(t)
        else:
            data_tokens.append(t)

    name_lines = group_tokens_into_lines(test_name_tokens)
    anchors = []

    for line in name_lines:
        line_tokens = line["tokens"]
        raw_name_v = " ".join([t.get("vietocr_text", t["text"]) for t in line_tokens]).strip()
        raw_name_p = " ".join([t.get("paddle_text", t["text"]) for t in line_tokens]).strip()
        c_v, s_v = find_test_name_with_score(raw_name_v, aliases)
        c_p, s_p = find_test_name_with_score(raw_name_p, aliases)
        best_c, best_s, used_engine, used_raw = None, 0, "VietOCR", raw_name_v
        if c_v and c_p:
            if s_v >= s_p: best_c, best_s, used_engine, used_raw = c_v, s_v, "VietOCR", raw_name_v
            else: best_c, best_s, used_engine, used_raw = c_p, s_p, "PaddleOCR", raw_name_p
        elif c_v: best_c, best_s, used_engine, used_raw = c_v, s_v, "VietOCR", raw_name_v
        elif c_p: best_c, best_s, used_engine, used_raw = c_p, s_p, "PaddleOCR", raw_name_p

        cy = float(np.mean([bbox_center(t["bbox"])[1] for t in line_tokens]))
        for t in line_tokens: t["tag"] = "TEST_NAME"
        anchors.append({
            "canonical": best_c if best_c else used_raw, "match_score": best_s, "used_engine": used_engine, "center_y": cy,
            "matched_data": {"VALUE": None, "UNIT": None, "REF_RANGE": None}, "raw_tokens": list(line_tokens), "raw_v": raw_name_v, "raw_p": raw_name_p
        })

    if not anchors: return []
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
            if not placed: groups.append([t])
        
        for g in groups:
            g.sort(key=lambda x: x["bbox"][0])
            col_candidates[col_name].append({
                "center_y": np.mean([bbox_center(x["bbox"])[1] for x in g]),
                "tokens": g,
                "text": " ".join([x.get("final_data_text", x.get("paddle_text", x["text"])) for x in g])
            })

    for a in anchors: a["current_y"] = a["center_y"]
    target_cols = ["VALUE", "UNIT", "REF_RANGE"]
    target_cols.sort(key=lambda col: col_boundaries[col][0] if col in col_boundaries else float('inf'))

    for col_name in target_cols:
        candidates = col_candidates[col_name]
        if not candidates: continue
        n, m = len(anchors), len(candidates)
        dp = np.full((n + 1, m + 1), float('inf'))
        dp[0][0] = 0.0
        trace = np.zeros((n + 1, m + 1), dtype=int)
        gap_penalty = (med_h * 2.0) ** 2 
        max_y_dist = med_h * 1.0 if col_name == "VALUE" else med_h * 1.5
        
        for i in range(n + 1):
            for j in range(m + 1):
                if i > 0:
                    cost = dp[i-1][j] + gap_penalty
                    if cost < dp[i][j]: dp[i][j], trace[i][j] = cost, 1
                if j > 0:
                    cost = dp[i][j-1] + gap_penalty
                    if cost < dp[i][j]: dp[i][j], trace[i][j] = cost, 2
                if i > 0 and j > 0:
                    dist = abs(anchors[i-1]["current_y"] - candidates[j-1]["center_y"])
                    if dist <= max_y_dist: 
                        cost = dp[i-1][j-1] + dist ** 2
                        if cost < dp[i][j]: dp[i][j], trace[i][j] = cost, 0
                            
        i, j = n, m
        matches = []
        while i > 0 or j > 0:
            if trace[i][j] == 0 and i > 0 and j > 0: matches.append((i-1, j-1)); i -= 1; j -= 1
            elif trace[i][j] == 1 and i > 0: i -= 1
            elif trace[i][j] == 2 and j > 0: j -= 1
            else: break
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
            if a["matched_data"][col]: row_data[col] = a["matched_data"][col]["text"]

        val_is_empty = not row_data["VALUE"].strip()
        if not val_is_empty and not re.search(r'\d', row_data["VALUE"]): val_is_empty = True
        if val_is_empty:
            combined_fallback = f'{row_data["UNIT"]} {row_data["REF_RANGE"]}'.strip()
            match = re.match(r'^([<>]?\s*[-+]?\d+(?:[\.,]\d+)*)\s*(.*)', combined_fallback)
            if match:
                possible_val, remainder = match.group(1).strip(), match.group(2).strip()
                if not (possible_val == "10" and (remainder.startswith("^") or remainder.startswith("*") or remainder.startswith("mũ"))):
                    row_data["VALUE"] = combined_fallback
                    row_data["UNIT"] = ""
                    row_data["REF_RANGE"] = ""

        ref_min, ref_max, fallback_unit = parse_ref_range(row_data["REF_RANGE"])
        final_unit = row_data["UNIT"]

        if ref_min is None and ref_max is None and final_unit and re.search(r'\d', final_unit):
            ex_min, ex_max, rem_unit = parse_ref_range(final_unit)
            if ex_min is not None or ex_max is not None:
                ref_min, ref_max, final_unit = ex_min, ex_max, rem_unit

        if ref_min is None and ref_max is None and row_data["VALUE"]:
            ex_min, ex_max, rem_val = parse_ref_range(row_data["VALUE"])
            if ex_min is not None or ex_max is not None:
                ref_min, ref_max, row_data["VALUE"] = ex_min, ex_max, rem_val

        if row_data["VALUE"] and re.search(r'[a-zA-Z%‰/]', row_data["VALUE"]):
            match = re.match(r'^([<>]?\s*[-+]?\d+(?:[\.,]\d+)*)\s*(.*)', row_data["VALUE"])
            if match and not final_unit:
                row_data["VALUE"], final_unit = match.group(1).strip(), match.group(2).strip("() ")

        if not final_unit and fallback_unit: final_unit = fallback_unit
        final_unit = fix_unit_typos(final_unit)
        cleaned_value = clean_value_string(row_data["VALUE"])
        numeric_value = extract_strict_number(cleaned_value)

        if numeric_value is None: continue
        a["raw_tokens"].sort(key=lambda t: t["bbox"][0])
        a["final_value"] = numeric_value
        a["final_unit"] = final_unit
        a["final_ref_min"] = ref_min
        a["final_ref_max"] = ref_max
        a["final_status"] = calculate_status(row_data["VALUE"], ref_min, ref_max)
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
                    _, s_v = find_test_name_with_score(a["raw_v"], aliases, unused_canons_for_tie)
                    _, s_p = find_test_name_with_score(a["raw_p"], aliases, unused_canons_for_tie)
                    a["fallback_score"] = max(s_v, s_p)
                    unit_bonus = 0
                    if "PERCENT" in a["target_canonical"] and "%" in a["final_unit"]: unit_bonus = 1000
                    elif "ABS" in a["target_canonical"] and re.search(r'[lL]', a["final_unit"]): unit_bonus = 1000
                    a["unit_bonus"] = unit_bonus

                anchor_list.sort(key=lambda x: (x.get("unit_bonus", 0), x["match_score"], -x.get("fallback_score", 0), -x["center_y"]), reverse=True)
                used_canons = set(a["target_canonical"] for a in valid_anchors if a["target_canonical"] in all_canons)
                unused_canons = all_canons - used_canons
                
                for loser in anchor_list[1:]:
                    c_v, s_v = find_test_name_with_score(loser["raw_v"], aliases, unused_canons)
                    c_p, s_p = find_test_name_with_score(loser["raw_p"], aliases, unused_canons)
                    
                    if c_v and c_p:
                        if s_v >= s_p: loser["target_canonical"], loser["match_score"], loser["used_engine"] = c_v, s_v, "VietOCR"
                        else: loser["target_canonical"], loser["match_score"], loser["used_engine"] = c_p, s_p, "PaddleOCR"
                    elif c_v: loser["target_canonical"], loser["match_score"], loser["used_engine"] = c_v, s_v, "VietOCR"
                    elif c_p: loser["target_canonical"], loser["match_score"], loser["used_engine"] = c_p, s_p, "PaddleOCR"
                    else:
                        loser["target_canonical"] = loser["raw_v"] if loser["used_engine"] == "VietOCR" else loser["raw_p"]
                        loser["match_score"] = 0
                    if loser["target_canonical"] in all_canons: unused_canons.discard(loser["target_canonical"])
        if not duplicates_exist: break

    final_results = []
    for a in valid_anchors:
        if USE_ONTOLOGY_FILTER and a["target_canonical"] not in all_canons: continue
        final_unit_corrected = correct_unit_by_ontology(a["final_unit"], a["target_canonical"], unit_ontology)
        raw_line_texts = []
        for t in a["raw_tokens"]:
            if t.get("tag") == "TEST_NAME":
                if a.get("used_engine") == "PaddleOCR": raw_line_texts.append(t.get("paddle_text", t["text"]))
                else: raw_line_texts.append(t.get("vietocr_text", t["text"]))
            else: raw_line_texts.append(t.get("final_data_text", t.get("paddle_text", t["text"])))

        final_results.append({
            "test_name": a["target_canonical"], "value": a["final_value"], "unit": final_unit_corrected,
            "ref_range": {"ref_min": a["final_ref_min"], "ref_max": a["final_ref_max"]},
            "status": a["final_status"], "raw_text_line": " ".join(raw_line_texts)
        })
    return final_results

# =========================================================
# MAIN PIPELINE 
# =========================================================

def run_end_to_end_pipeline(image_path):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    test_ontology = load_ontology(ONTOLOGY_JSON)
    aliases = build_alias_index(test_ontology)
    unit_ontology = load_unit_ontology(ONTOLOGY_UNITS_JSON)

    print(f"-> Đang nạp ảnh: {image_path}")
    
    # FIX LỖI: Dùng np.fromfile và cv2.imdecode để đọc được ảnh có đường dẫn/tên tiếng Việt trên Windows
    try:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"-> LỖI: Không thể giải mã ảnh: {str(e)}")
        return None

    if img is None: 
        print("-> LỖI: Không thể đọc được ảnh (File hỏng hoặc trống).")
        return None

    print("-> Đang quét text và định vị (PaddleOCR)...")
    tokens_pass1 = predict_tokens(img) # Đổi thành truyền thẳng object ảnh vào thay vì path
    if not tokens_pass1: 
        print("-> LỖI: Không tìm thấy chữ trên ảnh gốc.")
        return None

    boxes_pass1 = [t["poly"] for t in tokens_pass1]
    if boxes_pass1:
        print("-> Đang nắn phẳng ảnh (Deskew)...")
        angle_h, angle_v = get_robust_deskew_angles(boxes_pass1, img)
        warped_img = deskew_and_deshear_image(img, angle_h, angle_v)
    else: 
        warped_img = img.copy()

    scanned_path = Path(OUTPUT_DIR) / OUTPUT_SCANNED_IMG
    cv2.imwrite(str(scanned_path), warped_img)

    print("-> Đang quét lại ảnh phẳng (PaddleOCR)...")
    tokens = predict_tokens(warped_img) # Đổi thành truyền thẳng object ảnh
    if not tokens: return None

    print("-> Đang xử lý tách cột và chẻ dòng...")
    med_h = median_token_height(tokens)
    split_tokens = []
    for t in tokens: split_tokens.extend(split_tall_token_if_needed(warped_img, t, med_h))
        
    filtered_tokens = []
    for t in split_tokens:
        h = bbox_height(t["bbox"])
        if h > med_h * 4.0: continue
        filtered_tokens.append(t)
    tokens = filtered_tokens

    for t in tokens:
        t["paddle_text"] = t["text"]
        t["tag"] = "OTHER"

    print("-> Đang đọc nội dung tiếng Việt chuyên sâu (VietOCR)...")
    for t in tokens:
        crop = crop_image_by_poly(warped_img, t["poly"])
        if crop is not None and crop.size > 0:
            try:
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                v_text, v_score = vietocr_predictor.predict(crop_pil, return_prob=True)
                t["vietocr_text"] = v_text
                t["vietocr_score"] = v_score
                if not t["text"]: t["text"] = v_text; t["paddle_text"] = v_text
            except: t["vietocr_text"] = t["text"]
        else: t["vietocr_text"] = t["text"]
        t["final_data_text"] = resolve_data_text(t.get("paddle_text", t["text"]), t.get("vietocr_text", ""), t.get("score", 0), t.get("vietocr_score", 0))

    data_start_y, col_boundaries, matched_header_tokens = identify_columns_robust(tokens)

    if not col_boundaries: extracted_data = []
    else:
        for t in matched_header_tokens: t["tag"] = "HEADER"
        extracted_data = extract_table_data_anchored(tokens, data_start_y, col_boundaries, aliases, med_h, unit_ontology)

    final_output = {
        "extracted_data": extracted_data,
        "raw_ocr": [{"text": t["text"], "paddle_text": t.get("paddle_text", ""), "vietocr_text": t.get("vietocr_text", ""), "bbox": t["bbox"], "tag": t.get("tag", "OTHER")} for t in tokens]
    }
    with open(Path(OUTPUT_DIR) / OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    clean_extracted_data = [{k: v for k, v in item.items() if k != "raw_text_line"} for item in extracted_data]
    with open(Path(OUTPUT_DIR) / OUTPUT_CLEAN_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_extracted_data, f, ensure_ascii=False, indent=4)
        
    # VẼ DEBUG LÊN ẢNH ĐÃ SCAN
    h, w = warped_img.shape[:2]
    if data_start_y > 0: cv2.line(warped_img, (0, int(data_start_y)), (w, int(data_start_y)), (0, 255, 255), 2)
    for col_name, (x_min, x_max) in col_boundaries.items():
        x_min, x_max_int = max(0, int(x_min)), w if x_max == float('inf') else min(w, int(x_max))
        cv2.rectangle(warped_img, (x_min, 0), (x_max_int, h), (0, 100, 0), 1)
        cv2.putText(warped_img, col_name, (x_min + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for t in tokens:
        b = t["bbox"]
        if not b: continue
        x1, y1, x2, y2 = map(int, b)
        tag = t.get("tag")
        if tag == "HEADER": color = (255, 0, 255)   
        elif tag == "TEST_NAME": color = (0, 0, 255)     
        elif tag == "DATA": color = (0, 255, 0)     
        else: color = (0, 165, 255)   
        cv2.rectangle(warped_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(warped_img, strip_accents(t.get("chosen_text", t.get("vietocr_text", t["text"]))), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    debug_path = Path(OUTPUT_DIR) / OUTPUT_DEBUG_IMG
    cv2.imwrite(str(debug_path), warped_img)
    
    print("-> Đã trích xuất xong dữ liệu OCR!")
    # TRẢ DỮ LIỆU ĐỂ API FASTAPI SỬ DỤNG
    return clean_extracted_data