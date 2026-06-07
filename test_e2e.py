# test_cases.py
import os
os.environ.setdefault("NEO4J_PASSWORD", "25251325")

import rag_service

# ── CASE 2.jpg: WBC cao + NEUT_ABS cao + MCV thấp + MCH thấp ──
case2 = [
    {"test_name": "WBC",          "value": 19.47, "unit": "G/L", "status": "High",
     "ref_range": {"ref_min": "4.0",  "ref_max": "10.0"}},
    {"test_name": "NEUT_ABS",     "value": 13.08, "unit": "G/L", "status": "High",
     "ref_range": {"ref_min": "2.0",  "ref_max": "7.5"}},
    {"test_name": "MONO_ABS",     "value": 1.63,  "unit": "G/L", "status": "High",
     "ref_range": {"ref_min": "0.0",  "ref_max": "1.0"}},
    {"test_name": "BASO_PERCENT", "value": 0.0,   "unit": "%",   "status": "Low",
     "ref_range": {"ref_min": "0.1",  "ref_max": "1.5"}},
    {"test_name": "MCV",          "value": 80.5,  "unit": "fL",  "status": "Low",
     "ref_range": {"ref_min": "85.0", "ref_max": "95.0"}},
    {"test_name": "MCH",          "value": 26.9,  "unit": "pg",  "status": "Low",
     "ref_range": {"ref_min": "28.0", "ref_max": "32.0"}},
]

# ── CASE 3.jpg: Gần normal, chỉ IG% cao nhẹ ──
case3 = [
    {"test_name": "IG_PERCENT", "value": 0.9, "unit": "%", "status": "High",
     "ref_range": {"ref_min": "0.0", "ref_max": "0.45"}},
]

for name, indicators in [("2.jpg", case2), ("3.jpg", case3)]:
    print("=" * 60)
    print(f"TEST CASE: {name}")
    print("=" * 60)
    result = rag_service.analyze_indicators_with_llm(
        user_indicators=indicators,
        session_id=f"test_{name}"
    )
    print()
    print(result)
    print()