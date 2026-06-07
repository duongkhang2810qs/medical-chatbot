# indicates_mapping.py
"""
Bảng mapping Test → Condition với direction và confidence.
Được trích xuất tự động từ PANEL_PATTERNS và CROSS_PANEL_PATTERNS trong config.py.

Quy tắc confidence:
- Test nằm trong requires → confidence cao hơn (0.75-0.90)
- Test nằm trong optional → confidence thấp hơn (0.55-0.70)
- Confidence được xác định dựa trên mức độ đặc hiệu lâm sàng của từng chỉ số
"""

# Format: (test_code, direction, condition_name, confidence, panel, source_pattern)
INDICATES_MAPPING = [

    # ── CBC: Microcytic anemia pattern ────────────────────────
    ("HGB",  "low",  "microcytic_anemia",      0.80, "CBC", "microcytic_anemia_pattern"),
    ("HGB",  "low",  "iron_deficiency_anemia",  0.75, "CBC", "microcytic_anemia_pattern"),
    ("MCV",  "low",  "microcytic_anemia",       0.85, "CBC", "microcytic_anemia_pattern"),
    ("MCV",  "low",  "iron_deficiency_anemia",  0.80, "CBC", "microcytic_anemia_pattern"),
    ("RDW",  "high", "microcytic_anemia",       0.60, "CBC", "microcytic_anemia_pattern"),
    ("MCH",  "low",  "iron_deficiency_anemia",  0.65, "CBC", "microcytic_anemia_pattern"),
    ("MCHC", "low",  "iron_deficiency_anemia",  0.60, "CBC", "microcytic_anemia_pattern"),

    # ── CBC: Macrocytic anemia pattern ────────────────────────
    ("HGB",  "low",  "macrocytic_anemia",           0.80, "CBC", "macrocytic_anemia_pattern"),
    ("MCV",  "high", "macrocytic_anemia",            0.85, "CBC", "macrocytic_anemia_pattern"),
    ("MCV",  "high", "b12_or_folate_related_anemia", 0.75, "CBC", "macrocytic_anemia_pattern"),

    # ── CBC: Bacterial infection pattern ──────────────────────
    ("WBC",  "high", "bacterial_infection",         0.80, "CBC", "bacterial_infection_pattern"),
    ("WBC",  "high", "infection",                   0.75, "CBC", "bacterial_infection_pattern"),
    ("WBC",  "high", "left_shift_stress_response",  0.70, "CBC", "bacterial_infection_pattern"),
    ("NEUT", "high", "bacterial_infection",         0.85, "CBC", "bacterial_infection_pattern"),
    ("NEUT", "high", "infection",                   0.80, "CBC", "bacterial_infection_pattern"),
    ("NEUT", "high", "left_shift_stress_response",  0.75, "CBC", "bacterial_infection_pattern"),
    ("IG",   "high", "left_shift_stress_response",  0.65, "CBC", "bacterial_infection_pattern"),
    ("IG",   "high", "bacterial_infection",         0.60, "CBC", "bacterial_infection_pattern"),

    # ── CBC: Viral infection pattern ──────────────────────────
    ("LYMPH", "high", "viral_infection",  0.80, "CBC", "viral_infection_pattern"),
    ("LYMPH", "high", "lymphocytosis",    0.85, "CBC", "viral_infection_pattern"),
    ("WBC",   "low",  "viral_infection",  0.60, "CBC", "viral_infection_pattern"),

    # ── CBC: Thrombocytopenia pattern ─────────────────────────
    ("PLT", "low", "thrombocytopenia", 0.90, "CBC", "thrombocytopenia_pattern"),
    ("PLT", "low", "bleeding_risk",   0.80, "CBC", "thrombocytopenia_pattern"),

    # ── CBC: Polycythemia pattern ─────────────────────────────
    ("HGB", "high", "polycythemia",    0.80, "CBC", "polycythemia_pattern"),
    ("HGB", "high", "erythrocytosis",  0.75, "CBC", "polycythemia_pattern"),
    ("HCT", "high", "polycythemia",    0.80, "CBC", "polycythemia_pattern"),
    ("RBC", "high", "erythrocytosis",  0.65, "CBC", "polycythemia_pattern"),

    # ── BIOCHEM: Hepatocellular injury pattern ────────────────
    ("ALT", "high", "hepatocellular_injury", 0.85, "BIOCHEM", "hepatocellular_injury_pattern"),
    ("ALT", "high", "liver_injury",          0.80, "BIOCHEM", "hepatocellular_injury_pattern"),
    ("AST", "high", "hepatocellular_injury", 0.85, "BIOCHEM", "hepatocellular_injury_pattern"),
    ("AST", "high", "liver_injury",          0.80, "BIOCHEM", "hepatocellular_injury_pattern"),

    # ── BIOCHEM: Renal impairment pattern ────────────────────
    ("CREATININE", "high", "renal_impairment",          0.90, "BIOCHEM", "renal_impairment_pattern"),
    ("CREATININE", "high", "reduced_kidney_function",   0.85, "BIOCHEM", "renal_impairment_pattern"),
    ("UREA",       "high", "renal_impairment",          0.75, "BIOCHEM", "renal_impairment_pattern"),
    ("K",          "high", "renal_impairment",          0.65, "BIOCHEM", "renal_impairment_pattern"),

    # ── BIOCHEM: Electrolyte disorder pattern ────────────────
    ("NA", "low",  "electrolyte_disorder", 0.85, "BIOCHEM", "electrolyte_disorder_pattern"),
    ("K",  "high", "electrolyte_disorder", 0.70, "BIOCHEM", "electrolyte_disorder_pattern"),
    ("K",  "low",  "electrolyte_disorder", 0.70, "BIOCHEM", "electrolyte_disorder_pattern"),
    ("CL", "low",  "electrolyte_disorder", 0.65, "BIOCHEM", "electrolyte_disorder_pattern"),
    ("CL", "high", "electrolyte_disorder", 0.65, "BIOCHEM", "electrolyte_disorder_pattern"),

    # ── BIOCHEM: Diabetes/glycemic pattern ───────────────────
    ("GLUCOSE", "high", "hyperglycemia",         0.90, "BIOCHEM", "diabetes_glycemic_pattern"),
    ("GLUCOSE", "high", "diabetes_risk",         0.80, "BIOCHEM", "diabetes_glycemic_pattern"),
    ("GLUCOSE", "high", "poor_glycemic_control", 0.70, "BIOCHEM", "diabetes_glycemic_pattern"),
    ("HBA1C",   "high", "poor_glycemic_control", 0.90, "BIOCHEM", "diabetes_glycemic_pattern"),
    ("HBA1C",   "high", "diabetes_risk",         0.85, "BIOCHEM", "diabetes_glycemic_pattern"),

    # ── BIOCHEM: Dyslipidemia pattern ────────────────────────
    ("LDL_C",       "high", "dyslipidemia",        0.85, "BIOCHEM", "dyslipidemia_pattern"),
    ("LDL_C",       "high", "cardiovascular_risk", 0.80, "BIOCHEM", "dyslipidemia_pattern"),
    ("TRIGLYCERIDE","high", "dyslipidemia",        0.70, "BIOCHEM", "dyslipidemia_pattern"),
    ("HDL_C",       "low",  "cardiovascular_risk", 0.70, "BIOCHEM", "dyslipidemia_pattern"),
    ("CHOLESTEROL", "high", "dyslipidemia",        0.65, "BIOCHEM", "dyslipidemia_pattern"),

    # ── BIOCHEM: Cardiac biomarker pattern ───────────────────
    ("TROPONIN_T", "high", "myocardial_injury", 0.90, "BIOCHEM", "cardiac_biomarker_pattern"),
    ("TROPONIN_T", "high", "cardiac_stress",    0.85, "BIOCHEM", "cardiac_biomarker_pattern"),
    ("CK_MB",      "high", "myocardial_injury", 0.75, "BIOCHEM", "cardiac_biomarker_pattern"),
    ("PRO_BNP",    "high", "cardiac_stress",    0.75, "BIOCHEM", "cardiac_biomarker_pattern"),

    # ── BIOCHEM: Iron store pattern ───────────────────────────
    ("FERRITIN", "high", "iron_store_abnormality",    0.80, "BIOCHEM", "iron_store_pattern"),
    ("FERRITIN", "high", "inflammatory_iron_pattern", 0.70, "BIOCHEM", "iron_store_pattern"),

    # ── BIOCHEM: Hypoalbuminemia pattern ─────────────────────
    ("ALBUMIN", "low", "hypoalbuminemia",             0.90, "BIOCHEM", "hypoalbuminemia_pattern"),
    ("ALBUMIN", "low", "protein_loss_or_inflammation",0.80, "BIOCHEM", "hypoalbuminemia_pattern"),

    # ── CROSS-PANEL: Renal anemia ─────────────────────────────
    ("HGB",        "low",  "renal_related_anemia",      0.75, "CBC",    "renal_anemia_pattern"),
    ("CREATININE", "high", "renal_related_anemia",      0.80, "BIOCHEM","renal_anemia_pattern"),
    ("CREATININE", "high", "reduced_kidney_function",   0.85, "BIOCHEM","renal_anemia_pattern"),

    # ── CROSS-PANEL: Anemia of inflammation ──────────────────
    ("HGB",     "low",  "anemia_of_inflammation",     0.75, "CBC",    "anemia_inflammation_pattern"),
    ("FERRITIN","high", "anemia_of_inflammation",     0.70, "BIOCHEM","anemia_inflammation_pattern"),
    ("FERRITIN","high", "iron_metabolism_abnormality", 0.65, "BIOCHEM","anemia_inflammation_pattern"),

    # ── CROSS-PANEL: Infection stress ────────────────────────
    ("WBC",  "high", "infection_or_inflammatory_stress", 0.80, "CBC",    "infection_stress_pattern"),
    ("NEUT", "high", "infection_or_inflammatory_stress", 0.85, "CBC",    "infection_stress_pattern"),
    ("GLUCOSE","high","infection_or_inflammatory_stress",0.55, "BIOCHEM","infection_stress_pattern"),
    ("ALBUMIN","low", "infection_or_inflammatory_stress",0.55, "BIOCHEM","infection_stress_pattern"),

    # ── CROSS-PANEL: Metabolic cardiovascular risk ────────────
    ("GLUCOSE",     "high","metabolic_syndrome_risk",  0.75, "BIOCHEM","metabolic_cardiovascular_risk_pattern"),
    ("GLUCOSE",     "high","cardiovascular_risk",      0.70, "BIOCHEM","metabolic_cardiovascular_risk_pattern"),
    ("TRIGLYCERIDE","high","metabolic_syndrome_risk",  0.75, "BIOCHEM","metabolic_cardiovascular_risk_pattern"),
    ("HDL_C",       "low", "cardiovascular_risk",      0.70, "BIOCHEM","metabolic_cardiovascular_risk_pattern"),
    ("LDL_C",       "high","cardiovascular_risk",      0.75, "BIOCHEM","metabolic_cardiovascular_risk_pattern"),
    ("CHOLESTEROL", "high","metabolic_syndrome_risk",  0.65, "BIOCHEM","metabolic_cardiovascular_risk_pattern"),
]