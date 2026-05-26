# config.py
from pathlib import Path

# =========================================================
# BASE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw_sources"
CBC_RAW_DIR = RAW_DIR / "cbc"
BIOCHEM_RAW_DIR = RAW_DIR / "biochem"

CASE_DIR = DATA_DIR / "cases"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

ENV_PATH = BASE_DIR / ".env"
NEO4J_DIR = BASE_DIR / "neo4j_csv"


# =========================================================
# PDF SOURCES
# Đổi tên file ở đây nếu PDF của bạn khác tên
# =========================================================

CBC_PDF_PATHS = [
    CBC_RAW_DIR / "clinical_hematology.pdf",
    CBC_RAW_DIR / "harrison.pdf",
]

BIOCHEM_PDF_PATHS = [
    BIOCHEM_RAW_DIR / "henry.pdf",
    BIOCHEM_RAW_DIR / "tietz.pdf",
]


# =========================================================
# CASE INPUTS
# =========================================================

CBC_CASE_PATH = CASE_DIR / "all_results.jsonl"
BIOCHEM_CASE_PATH = CASE_DIR / "all_results_biochemistry.jsonl"
LAB_CASE_PATH = CASE_DIR / "all_results_lab.jsonl"


# =========================================================
# PROCESSED OUTPUTS
# =========================================================

CBC_KB_PATH = PROCESSED_DIR / "cbc_kb.json"
BIOCHEM_KB_PATH = PROCESSED_DIR / "biochem_kb.json"
LAB_KB_PATH = PROCESSED_DIR / "lab_kb.json"
LAB_GRAPH_PATH = PROCESSED_DIR / "lab_graph.json"

OUTPUT_PATH = OUTPUT_DIR / "final_output.jsonl"
FAILED_PATH = OUTPUT_DIR / "failed_cases.jsonl"


# =========================================================
# QDRANT / EMBEDDING
# =========================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab_kb"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K_PER_QUERY = 5
MAX_RAW_EVIDENCE = 20
MAX_FINAL_EVIDENCE = 6


# =========================================================
# LLM CONFIG
# =========================================================

REQUEST_TIMEOUT = 300
COLAB_MAX_NEW_TOKENS = 970
COLAB_TEMPERATURE = 0.2


# =========================================================
# KB BUILD CONFIG
# =========================================================

MIN_WORDS = 8
MAX_WORDS = 160

BOUNDARY_REGEX_TEXT = (
    r"\b("
    r"however|therefore|in contrast|in addition|because|although|whereas|"
    r"caused by|due to|associated with|suggests|indicates|consistent with|"
    r"elevation of|decrease in|low levels|high levels|is seen in|reflects"
    r")\b"
)


# =========================================================
# SOURCE TRUST
# =========================================================

SOURCE_TRUST = {
    "harrison.pdf": 1.0,
    "clinical_hematology.pdf": 0.95,
    "henry.pdf": 0.98,
    "tietz.pdf": 0.98,
}


# =========================================================
# CBC TEST MAP FOR PDF KB BUILD
# =========================================================

CBC_TEST_MAP = {
    "RBC": [
        "red blood cell", "rbc", "rbcs", "erythrocyte", "red cell",
        "red blood corpuscle"
    ],
    "HGB": [
        "hemoglobin", "haemoglobin", "hgb", "hb", "hgb concentration"
    ],
    "HCT": [
        "hematocrit", "hct", "packed cell volume"
    ],
    "MCV": [
        "mcv", "mean cell volume", "mean corpuscular volume",
        "microcytic", "macrocytic", "normocytic"
    ],
    "MCH": [
        "mch", "mean corpuscular hemoglobin",
        "mean cell hemoglobin", "hypochromic", "hyperchromic"
    ],
    "MCHC": [
        "mchc", "mean corpuscular hemoglobin concentration",
        "mean cell hemoglobin concentration"
    ],
    "RDW": [
        "rdw", "red cell distribution width", "red blood cell distribution",
        "anisocytosis", "rdw sd", "rdw cv", "rdw-sd", "rdw-cv"
    ],
    "WBC": [
        "wbc", "leukocyte", "white blood cell", "white cell count",
        "leukocytosis", "leukopenia"
    ],
    "NEUT": [
        "neutrophil", "neutrophilia", "neutropenia",
        "absolute neutrophil count", "neutrophil percentage",
        "granulocyte", "band cell", "left shift"
    ],
    "LYMPH": [
        "lymphocyte", "lymphocytosis", "lymphopenia",
        "absolute lymphocyte count", "lymphocyte percentage"
    ],
    "MONO": [
        "monocyte", "monocytosis",
        "absolute monocyte count", "monocyte percentage"
    ],
    "EOS": [
        "eosinophil", "eosinophilia",
        "absolute eosinophil count", "eosinophil percentage"
    ],
    "BASO": [
        "basophil", "basophilia",
        "absolute basophil count", "basophil percentage"
    ],
    "PLT": [
        "platelet", "plt", "thrombocyte",
        "thrombocytopenia", "thrombocytosis"
    ],
    "IG": [
        "immature granulocyte", "metamyelocyte", "myelocyte",
        "band", "left shift", "immature granulocyte percentage",
        "absolute immature granulocyte"
    ],
}


# =========================================================
# BIOCHEM TEST MAP FOR PDF KB BUILD
# =========================================================

BIOCHEM_TEST_MAP = {
    "AST": [
        "ast", "aspartate aminotransferase", "sgot", "transaminase"
    ],
    "ALT": [
        "alt", "alanine aminotransferase", "sgpt", "transaminase"
    ],
    "UREA": [
        "urea", "bun", "blood urea nitrogen", "azotemia", "uremia"
    ],
    "CREATININE": [
        "creatinine", "serum creatinine", "renal function",
        "kidney function"
    ],
    "NA": [
        "sodium", "na", "na+", "hyponatremia", "hypernatremia"
    ],
    "K": [
        "potassium", "k", "k+", "hypokalemia", "hyperkalemia"
    ],
    "CL": [
        "chloride", "cl", "cl-", "hypochloremia", "hyperchloremia"
    ],
    "GLUCOSE": [
        "glucose", "blood glucose", "hyperglycemia", "hypoglycemia",
        "diabetes", "glycemic"
    ],
    "HBA1C": [
        "hba1c", "hemoglobin a1c", "glycated hemoglobin",
        "glycemic control"
    ],
    "CHOLESTEROL": [
        "cholesterol", "total cholesterol"
    ],
    "TRIGLYCERIDE": [
        "triglyceride", "triglycerides", "tg"
    ],
    "HDL_C": [
        "hdl", "hdl-c", "hdl cholesterol", "high-density lipoprotein"
    ],
    "LDL_C": [
        "ldl", "ldl-c", "ldl cholesterol", "low-density lipoprotein"
    ],
    "CK_MB": [
        "ck-mb", "ck_mb", "creatine kinase-mb", "creatine kinase mb"
    ],
    "TROPONIN_T": [
        "troponin t", "troponin-t", "troponin_t", "cardiac troponin"
    ],
    "PRO_BNP": [
        "pro-bnp", "pro_bnp", "nt-probnp", "natriuretic peptide",
        "brain natriuretic peptide"
    ],
    "FERRITIN": [
        "ferritin", "iron stores", "iron storage"
    ],
    "ALBUMIN": [
        "albumin", "hypoalbuminemia", "protein loss"
    ],
    "CALCIUM_ION": [
        "ionized calcium", "calcium ion", "ca++", "calcium"
    ],
    "PTH": [
        "pth", "parathyroid hormone"
    ],
    "URIC_ACID": [
        "uric acid", "urate", "hyperuricemia", "gout"
    ],
}


# =========================================================
# TOPIC KEYWORDS FOR PDF KB BUILD
# =========================================================

CBC_TOPIC_KEYWORDS = {
    "anemia": [
        "anemia", "anaemia", "iron deficiency", "megaloblastic",
        "hemolytic", "microcytic", "macrocytic"
    ],
    "infection": [
        "infection", "bacterial", "viral", "sepsis", "pathogen"
    ],
    "inflammation": [
        "inflammation", "inflammatory", "acute phase"
    ],
    "bleeding": [
        "bleeding", "hemorrhage", "haemorrhage", "coagulation", "thrombosis"
    ],
    "bone_marrow": [
        "bone marrow", "aplastic", "myeloid", "hematopoiesis"
    ],
    "thalassemia": [
        "thalassemia", "thalassaemia", "hemoglobinopathy"
    ],
    "polycythemia": [
        "polycythemia", "erythrocytosis"
    ],
    "coagulation": [
        "coagulation", "fibrinogen", "prothrombin"
    ],
}


BIOCHEM_TOPIC_KEYWORDS = {
    "liver_injury": [
        "hepatitis", "hepatocellular", "transaminase",
        "aminotransferase", "liver injury", "liver disease"
    ],
    "renal_function": [
        "azotemia", "uremia", "acute kidney injury", "chronic kidney disease",
        "renal function", "kidney function", "creatinine", "urea"
    ],
    "electrolyte_disorder": [
        "electrolyte", "hyponatremia", "hypernatremia",
        "hypokalemia", "hyperkalemia", "chloride"
    ],
    "diabetes": [
        "diabetes", "hyperglycemia", "hypoglycemia",
        "glycemic control", "hba1c", "glucose"
    ],
    "dyslipidemia": [
        "dyslipidemia", "atherosclerosis", "cardiovascular risk",
        "lipoprotein", "cholesterol", "triglyceride"
    ],
    "cardiac_biomarker": [
        "myocardial infarction", "acute coronary syndrome",
        "heart failure", "troponin", "natriuretic peptide", "ck-mb"
    ],
    "iron_metabolism": [
        "ferritin", "iron deficiency", "iron stores", "iron metabolism"
    ],
    "protein_nutrition": [
        "albumin", "hypoalbuminemia", "malnutrition", "protein loss"
    ],
    "bone_mineral": [
        "ionized calcium", "parathyroid hormone",
        "hypercalcemia", "hypocalcemia", "pth"
    ],
    "purine_metabolism": [
        "urate", "gout", "hyperuricemia", "uric acid"
    ],
}


# =========================================================
# TYPE CLASSIFICATION PATTERNS
# =========================================================

TYPE_PATTERNS = {
    "definition": [
        "is defined as",
        "refers to",
        "is characterized by",
        "is a condition characterized by",
        "is a disorder characterized by",
        "is a type of",
        "is the most common",
        "is defined by the presence of",
    ],
    "cause": [
        "caused by",
        "due to",
        "result of",
        "secondary to",
        "etiology",
        "can cause",
        "may cause",
        "leads to",
        "results in",
        "can lead to",
        "can result in",
        "is a common cause of",
        "is a major cause of",
        "is often caused by",
        "is the main cause of",
    ],
    "interpretation": [
        "indicates",
        "suggests",
        "associated with",
        "consistent with",
        "is seen in",
        "reflects",
        "elevation of",
        "decrease in",
        "low levels",
        "high levels",
        "correlates with",
        "is linked to",
        "is related to",
        "is commonly seen in",
        "is marked by",
        "is associated with the presence of",
    ],
}


# =========================================================
# TEST NORMALIZATION FOR CASE INPUT
# =========================================================

TEST_NORMALIZATION = {
    "CBC": {
        "RBC": "RBC",

        "HGB": "HGB",
        "HB": "HGB",
        "HEMOGLOBIN": "HGB",
        "HAEMOGLOBIN": "HGB",

        "HCT": "HCT",
        "HEMATOCRIT": "HCT",

        "MCV": "MCV",
        "MCH": "MCH",
        "MCHC": "MCHC",

        "RDW": "RDW",
        "RDW_SD": "RDW",
        "RDW-SD": "RDW",
        "RDW_CV": "RDW",
        "RDW-CV": "RDW",

        "WBC": "WBC",

        "NEUT": "NEUT",
        "NEUT_PERCENT": "NEUT",
        "NEUT%": "NEUT",
        "NEUT_ABS": "NEUT",
        "NEUTROPHIL": "NEUT",

        "LYMPH": "LYMPH",
        "LYM": "LYMPH",
        "LYM_PERCENT": "LYMPH",
        "LYM%": "LYMPH",
        "LYM_ABS": "LYMPH",
        "LYMPH_PERCENT": "LYMPH",
        "LYMPH%": "LYMPH",
        "LYMPH_ABS": "LYMPH",
        "LYMPHOCYTE": "LYMPH",

        "MONO": "MONO",
        "MONO_PERCENT": "MONO",
        "MONO%": "MONO",
        "MONO_ABS": "MONO",

        "EOS": "EOS",
        "EOS_PERCENT": "EOS",
        "EOS%": "EOS",
        "EOS_ABS": "EOS",

        "BASO": "BASO",
        "BASO_PERCENT": "BASO",
        "BASO%": "BASO",
        "BASO_ABS": "BASO",

        "PLT": "PLT",
        "PLATELET": "PLT",

        "IG": "IG",
        "IG_PERCENT": "IG",
        "IG%": "IG",
        "IG_ABS": "IG",
    },

    "BIOCHEM": {
        "AST": "AST",
        "SGOT": "AST",

        "ALT": "ALT",
        "SGPT": "ALT",

        "UREA": "UREA",
        "BUN": "UREA",
        "BLOOD_UREA_NITROGEN": "UREA",

        "CREATININE": "CREATININE",
        "CRE": "CREATININE",
        "CR": "CREATININE",

        "NA": "NA",
        "NA+": "NA",
        "SODIUM": "NA",

        "K": "K",
        "K+": "K",
        "POTASSIUM": "K",

        "CL": "CL",
        "CL-": "CL",
        "CHLORIDE": "CL",

        "GLUCOSE": "GLUCOSE",
        "GLU": "GLUCOSE",
        "BLOOD_GLUCOSE": "GLUCOSE",

        "HBA1C": "HBA1C",
        "HbA1c": "HBA1C",

        "CHOLESTEROL": "CHOLESTEROL",
        "TOTAL_CHOLESTEROL": "CHOLESTEROL",
        "TC": "CHOLESTEROL",

        "TRIGLYCERIDE": "TRIGLYCERIDE",
        "TRIGLYCERIDES": "TRIGLYCERIDE",
        "TG": "TRIGLYCERIDE",

        "HDL_C": "HDL_C",
        "HDL-C": "HDL_C",
        "HDL": "HDL_C",

        "LDL_C": "LDL_C",
        "LDL-C": "LDL_C",
        "LDL": "LDL_C",

        "CK_MB": "CK_MB",
        "CK-MB": "CK_MB",

        "TROPONIN_T": "TROPONIN_T",
        "TROPONIN T": "TROPONIN_T",
        "TNT": "TROPONIN_T",

        "PRO_BNP": "PRO_BNP",
        "PRO-BNP": "PRO_BNP",
        "NT_PRO_BNP": "PRO_BNP",
        "NT-PROBNP": "PRO_BNP",

        "FERRITIN": "FERRITIN",

        "ALBUMIN": "ALBUMIN",
        "ALB": "ALBUMIN",

        "CALCIUM_ION": "CALCIUM_ION",
        "IONIZED_CALCIUM": "CALCIUM_ION",
        "CA_ION": "CALCIUM_ION",
        "CA++": "CALCIUM_ION",

        "PTH": "PTH",
        "PARATHYROID_HORMONE": "PTH",

        "URIC_ACID": "URIC_ACID",
        "URATE": "URIC_ACID",
    },
}


# =========================================================
# TEST LABELS
# =========================================================

TEST_LABELS = {
    "RBC": "Red Blood Cell",
    "HGB": "Hemoglobin",
    "HCT": "Hematocrit",
    "MCV": "Mean Corpuscular Volume",
    "MCH": "Mean Corpuscular Hemoglobin",
    "MCHC": "Mean Corpuscular Hemoglobin Concentration",
    "RDW": "Red Cell Distribution Width",
    "WBC": "White Blood Cell",
    "NEUT": "Neutrophil",
    "LYMPH": "Lymphocyte",
    "MONO": "Monocyte",
    "EOS": "Eosinophil",
    "BASO": "Basophil",
    "PLT": "Platelet",
    "IG": "Immature Granulocyte",

    "AST": "Aspartate Aminotransferase",
    "ALT": "Alanine Aminotransferase",
    "UREA": "Urea",
    "CREATININE": "Creatinine",
    "NA": "Sodium",
    "K": "Potassium",
    "CL": "Chloride",
    "GLUCOSE": "Glucose",
    "HBA1C": "Hemoglobin A1c",
    "CHOLESTEROL": "Total Cholesterol",
    "TRIGLYCERIDE": "Triglyceride",
    "HDL_C": "HDL Cholesterol",
    "LDL_C": "LDL Cholesterol",
    "CK_MB": "CK-MB",
    "TROPONIN_T": "Troponin T",
    "PRO_BNP": "Pro-BNP",
    "FERRITIN": "Ferritin",
    "ALBUMIN": "Albumin",
    "CALCIUM_ION": "Ionized Calcium",
    "PTH": "Parathyroid Hormone",
    "URIC_ACID": "Uric Acid",
}


# =========================================================
# PANEL PATTERNS
# =========================================================

PANEL_PATTERNS = {
    "CBC": [
        {
            "pattern_id": "microcytic_anemia_pattern",
            "name": "Microcytic anemia pattern",
            "requires": [("HGB", "low"), ("MCV", "low")],
            "optional": [("RDW", "high"), ("MCH", "low"), ("MCHC", "low")],
            "conditions": ["microcytic_anemia", "iron_deficiency_anemia"],
            "description": "Low hemoglobin with low MCV, often with high RDW, suggests a microcytic anemia pattern.",
        },
        {
            "pattern_id": "macrocytic_anemia_pattern",
            "name": "Macrocytic anemia pattern",
            "requires": [("HGB", "low"), ("MCV", "high")],
            "optional": [],
            "conditions": ["macrocytic_anemia", "b12_or_folate_related_anemia"],
            "description": "Low hemoglobin with elevated MCV suggests a macrocytic anemia pattern.",
        },
        {
            "pattern_id": "bacterial_infection_pattern",
            "name": "Bacterial infection pattern",
            "requires": [("WBC", "high"), ("NEUT", "high")],
            "optional": [("IG", "high")],
            "conditions": ["bacterial_infection", "infection", "left_shift_stress_response"],
            "description": "High WBC with neutrophilia, optionally immature granulocytes, suggests bacterial infection or inflammatory stress.",
        },
        {
            "pattern_id": "viral_infection_pattern",
            "name": "Viral infection pattern",
            "requires": [("LYMPH", "high")],
            "optional": [("WBC", "low")],
            "conditions": ["viral_infection", "lymphocytosis"],
            "description": "Lymphocytosis, sometimes with low WBC, may suggest viral infection pattern.",
        },
        {
            "pattern_id": "thrombocytopenia_pattern",
            "name": "Thrombocytopenia pattern",
            "requires": [("PLT", "low")],
            "optional": [],
            "conditions": ["thrombocytopenia", "bleeding_risk"],
            "description": "Low platelet count suggests thrombocytopenia pattern and possible bleeding risk context.",
        },
        {
            "pattern_id": "polycythemia_pattern",
            "name": "Polycythemia pattern",
            "requires": [("HGB", "high"), ("HCT", "high")],
            "optional": [("RBC", "high")],
            "conditions": ["polycythemia", "erythrocytosis"],
            "description": "High HGB/HCT, optionally elevated RBC, suggests polycythemia or erythrocytosis pattern.",
        },
    ],

    "BIOCHEM": [
        {
            "pattern_id": "hepatocellular_injury_pattern",
            "name": "Hepatocellular injury pattern",
            "requires": [("ALT", "high"), ("AST", "high")],
            "optional": [],
            "conditions": ["hepatocellular_injury", "liver_injury"],
            "description": "Elevation of ALT and AST suggests hepatocellular injury pattern.",
        },
        {
            "pattern_id": "renal_impairment_pattern",
            "name": "Renal impairment pattern",
            "requires": [("CREATININE", "high")],
            "optional": [("UREA", "high"), ("K", "high")],
            "conditions": ["renal_impairment", "reduced_kidney_function"],
            "description": "High creatinine, often with high urea or potassium, suggests impaired renal function.",
        },
        {
            "pattern_id": "electrolyte_disorder_pattern",
            "name": "Electrolyte disorder pattern",
            "requires": [("NA", "low")],
            "optional": [("K", "high"), ("K", "low"), ("CL", "low"), ("CL", "high")],
            "conditions": ["electrolyte_disorder"],
            "description": "Abnormal sodium, potassium, or chloride suggests electrolyte disorder.",
        },
        {
            "pattern_id": "diabetes_glycemic_pattern",
            "name": "Diabetes/glycemic abnormality pattern",
            "requires": [("GLUCOSE", "high")],
            "optional": [("HBA1C", "high")],
            "conditions": ["hyperglycemia", "diabetes_risk", "poor_glycemic_control"],
            "description": "High glucose, especially with high HbA1c, suggests abnormal glycemic control.",
        },
        {
            "pattern_id": "dyslipidemia_pattern",
            "name": "Dyslipidemia pattern",
            "requires": [("LDL_C", "high")],
            "optional": [("TRIGLYCERIDE", "high"), ("HDL_C", "low"), ("CHOLESTEROL", "high")],
            "conditions": ["dyslipidemia", "cardiovascular_risk"],
            "description": "Abnormal lipid profile suggests dyslipidemia and cardiovascular risk.",
        },
        {
            "pattern_id": "cardiac_biomarker_pattern",
            "name": "Cardiac biomarker pattern",
            "requires": [("TROPONIN_T", "high")],
            "optional": [("CK_MB", "high"), ("PRO_BNP", "high")],
            "conditions": ["myocardial_injury", "cardiac_stress"],
            "description": "Elevated cardiac biomarkers suggest myocardial injury or cardiac stress.",
        },
        {
            "pattern_id": "iron_store_pattern",
            "name": "Iron store abnormality pattern",
            "requires": [("FERRITIN", "high")],
            "optional": [],
            "conditions": ["iron_store_abnormality", "inflammatory_iron_pattern"],
            "description": "High ferritin may reflect increased iron stores or inflammatory context.",
        },
        {
            "pattern_id": "hypoalbuminemia_pattern",
            "name": "Hypoalbuminemia pattern",
            "requires": [("ALBUMIN", "low")],
            "optional": [],
            "conditions": ["hypoalbuminemia", "protein_loss_or_inflammation"],
            "description": "Low albumin suggests hypoalbuminemia, which may relate to inflammation, malnutrition, liver disease, or protein loss.",
        },
    ],
}


# =========================================================
# CROSS-PANEL PATTERNS
# =========================================================

CROSS_PANEL_PATTERNS = [
    {
        "pattern_id": "renal_anemia_pattern",
        "name": "Renal-related anemia pattern",
        "requires": [
            ("CBC", "HGB", "low"),
            ("BIOCHEM", "CREATININE", "high"),
        ],
        "optional": [
            ("BIOCHEM", "UREA", "high"),
            ("CBC", "HCT", "low"),
        ],
        "conditions": ["renal_related_anemia", "reduced_kidney_function"],
        "description": "Anemia together with elevated creatinine may suggest renal-related anemia or kidney impairment context.",
    },
    {
        "pattern_id": "anemia_inflammation_pattern",
        "name": "Anemia with inflammatory or iron-store pattern",
        "requires": [
            ("CBC", "HGB", "low"),
            ("BIOCHEM", "FERRITIN", "high"),
        ],
        "optional": [
            ("CBC", "MCV", "low"),
            ("BIOCHEM", "ALBUMIN", "low"),
        ],
        "conditions": ["anemia_of_inflammation", "iron_metabolism_abnormality"],
        "description": "Low hemoglobin with high ferritin may suggest inflammatory or iron-store related anemia context.",
    },
    {
        "pattern_id": "infection_stress_pattern",
        "name": "Infection or inflammatory stress cross-panel pattern",
        "requires": [
            ("CBC", "WBC", "high"),
            ("CBC", "NEUT", "high"),
        ],
        "optional": [
            ("BIOCHEM", "GLUCOSE", "high"),
            ("BIOCHEM", "ALBUMIN", "low"),
        ],
        "conditions": ["infection_or_inflammatory_stress"],
        "description": "Leukocytosis with neutrophilia, optionally with biochemical stress markers, suggests infection or inflammatory stress context.",
    },
    {
        "pattern_id": "metabolic_cardiovascular_risk_pattern",
        "name": "Metabolic cardiovascular risk pattern",
        "requires": [
            ("BIOCHEM", "GLUCOSE", "high"),
            ("BIOCHEM", "TRIGLYCERIDE", "high"),
        ],
        "optional": [
            ("BIOCHEM", "HDL_C", "low"),
            ("BIOCHEM", "LDL_C", "high"),
            ("BIOCHEM", "CHOLESTEROL", "high"),
        ],
        "conditions": ["metabolic_syndrome_risk", "cardiovascular_risk"],
        "description": "Combined glycemic and lipid abnormalities suggest metabolic cardiovascular risk.",
    },
]


# =========================================================
# STATUS NORMALIZATION
# =========================================================

STATUS_NORMALIZATION = {
    "high": "high",
    "h": "high",
    "elevated": "high",
    "increase": "high",
    "increased": "high",
    "above": "high",
    "cao": "high",

    "low": "low",
    "l": "low",
    "decreased": "low",
    "decrease": "low",
    "below": "low",
    "thấp": "low",
    "thap": "low",

    "normal": "normal",
    "n": "normal",
    "within range": "normal",
    "bình thường": "normal",
    "binh thuong": "normal",

    "positive": "positive",
    "pos": "positive",
    "negative": "negative",
    "neg": "negative",
}
PDF_SKIP_FIRST_PAGES = 17
PDF_SKIP_PAGES_BY_FILE = {
    "clinical_hematology.pdf": 17,
    "harrison.pdf": 17,
    "henry.pdf": 17,
    "tietz.pdf": 17,
}
KB_QUALITY_THRESHOLD = 0.25
MAX_CHUNKS_PER_PDF = None

PATTERN_DIR = DATA_DIR / "patterns"

CBC_DEMO_PATTERN_PATH = PATTERN_DIR / "cbc_patterns.jsonl"
BIOCHEM_PATTERN_PATH = PATTERN_DIR / "biochem_patterns.json"