# neo4j_retriever.py
"""
Neo4j Graph Retriever cho GraphRAG pipeline.

Path A: Test ←[MENTIONS_TEST]- Evidence  (bổ sung)
Path B: Condition ←[SUPPORTS]- Evidence  (nguồn chính)
Path C: Full reasoning chain cho build_reasoning_paths()
Path D: Test -[INDICATES]→ Condition ←[SUPPORTS]- Evidence (chain hoàn chỉnh nhất)
"""
from __future__ import annotations
import os

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("[neo4j_retriever] WARNING: neo4j package chưa cài. Chạy: pip install neo4j")

# =========================================================
# CONFIG
# =========================================================
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "25251325")

_NEO4J_DRIVER = None


def get_neo4j_driver():
    global _NEO4J_DRIVER
    if not NEO4J_AVAILABLE:
        return None
    if _NEO4J_DRIVER is None:
        try:
            _NEO4J_DRIVER = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
            )
            _NEO4J_DRIVER.verify_connectivity()
            print(f"[neo4j_retriever] Connected: {NEO4J_URI}")
        except Exception as exc:
            print(f"[neo4j_retriever] Cannot connect to Neo4j: {exc}")
            _NEO4J_DRIVER = None
    return _NEO4J_DRIVER


# =========================================================
# NORMALIZE TEST NAME
# OCR trích ra NEUT%, NEUT#, LYM%, LYM#...
# Graph lưu theo tên chuẩn: NEUT, LYMPH, MONO...
# =========================================================
_TEST_NORMALIZE = {
    # NEUT: tỉ lệ % và số lượng tuyệt đối # đều về NEUT
    "NEUT%": "NEUT", "NEUT#": "NEUT",
    # LYMPH
    "LYM%": "LYMPH", "LYM#": "LYMPH",
    "LYMPH%": "LYMPH", "LYMPH#": "LYMPH",
    # MONO
    "MONO%": "MONO", "MONO#": "MONO",
    # EOS
    "EOS%": "EOS", "EOS#": "EOS",
    # BASO
    "BASO%": "BASO", "BASO#": "BASO",
    # IG
    "IG%": "IG", "IG#": "IG",
    # NRBC
    "NRBC%": "NRBC", "NRBC#": "NRBC",
    # RDW — CV và SD đều là RDW
    "RDW_CV": "RDW", "RDW_SD": "RDW",
    # Tên đầy đủ từ test_cases.py
    "NEUT_ABS": "NEUT", "NEUT_PERCENT": "NEUT",
    "LYM_ABS": "LYMPH", "LYM_PERCENT": "LYMPH",
    "MONO_ABS": "MONO", "MONO_PERCENT": "MONO",
    "EOS_ABS": "EOS",   "EOS_PERCENT": "EOS",
    "BASO_ABS": "BASO", "BASO_PERCENT": "BASO",
    "IG_ABS": "IG",     "IG_PERCENT": "IG",
}


def normalize_test_code(test_code: str) -> str:
    """Chuẩn hóa tên test từ OCR về tên trong Knowledge Graph."""
    return _TEST_NORMALIZE.get(test_code.upper(), test_code.upper())


# =========================================================
# HELPERS
# =========================================================
def _row_to_evidence(record: dict, score_boost: float = 0.0) -> dict:
    text     = record.get("text") or ""
    ev_id    = record.get("ev_id") or record.get("id") or ""
    source   = record.get("src_name") or record.get("source") or "Unknown"
    page     = record.get("page") or ""
    ev_type  = record.get("ev_type") or record.get("type") or "general"
    trust    = float(record.get("trust") or 0.8)
    kb_score = float(record.get("kb_score") or 0.0)
    panel    = record.get("panel") or "UNKNOWN"
    return {
        "evidence_id":    ev_id,
        "panel":          panel,
        "text":           text,
        "bm25_text":      text,
        "tests":          [],
        "topics":         [],
        "conditions":     [],
        "keywords":       [],
        "type":           ev_type,
        "score":          kb_score + score_boost,
        "kb_score":       kb_score,
        "trust":          trust,
        "source":         source,
        "page":           str(page),
        "source_type":    "pdf_book",
        "is_static":      False,
        "retrieval_path": record.get("retrieval_path", "neo4j"),
    }


# =========================================================
# PATH A: Test ←[MENTIONS_TEST]- Evidence
# =========================================================
_QUERY_BY_TEST = """
MATCH (t:Test {test_code: $test_code})
      <-[:MENTIONS_TEST]-(e:Evidence)
      -[:FROM_SOURCE]->(src:Source)
WHERE e.type IN ['interpretation', 'cause']
RETURN e.id AS ev_id, e.text AS text, e.type AS ev_type,
       e.page AS page, e.trust AS trust, e.score AS kb_score,
       e.panel AS panel, src.name AS src_name
ORDER BY e.score DESC, e.trust DESC
LIMIT $limit
"""


def retrieve_by_tests(test_codes: list[str], limit_per_test: int = 2) -> list[dict]:
    driver = get_neo4j_driver()
    if driver is None:
        return []
    results: list[dict] = []
    with driver.session() as session:
        for test_code in test_codes:
            norm = normalize_test_code(test_code)
            try:
                records = session.run(_QUERY_BY_TEST, test_code=norm, limit=limit_per_test)
                for rec in records:
                    ev = _row_to_evidence(dict(rec), score_boost=0.0)
                    ev["retrieval_path"] = f"neo4j_test:{norm}"
                    results.append(ev)
            except Exception as exc:
                print(f"[neo4j_retriever] Path A error (test={norm}): {exc}")
    return results


# =========================================================
# PATH B: Condition ←[SUPPORTS]- Evidence
# =========================================================
_QUERY_BY_CONDITIONS = """
MATCH (cond:Condition)
      <-[:SUPPORTS]-(e:Evidence)
      -[:FROM_SOURCE]->(src:Source)
WHERE cond.name IN $condition_names
  AND e.type IN ['interpretation', 'cause']
RETURN e.id AS ev_id, e.text AS text, e.type AS ev_type,
       e.page AS page, e.trust AS trust, e.score AS kb_score,
       e.panel AS panel, src.name AS src_name,
       cond.name AS matched_condition
ORDER BY e.score DESC, e.trust DESC
LIMIT $limit
"""


def retrieve_by_conditions(condition_names: list[str], limit: int = 10) -> list[dict]:
    driver = get_neo4j_driver()
    if driver is None or not condition_names:
        return []
    results: list[dict] = []
    with driver.session() as session:
        try:
            records = session.run(
                _QUERY_BY_CONDITIONS,
                condition_names=condition_names,
                limit=limit,
            )
            for rec in records:
                row = dict(rec)
                ev = _row_to_evidence(row, score_boost=0.05)
                ev["retrieval_path"] = f"neo4j_condition:{row.get('matched_condition', '')}"
                ev["conditions"]     = [row.get("matched_condition", "")]
                results.append(ev)
        except Exception as exc:
            print(f"[neo4j_retriever] Path B error: {exc}")
    return results


# =========================================================
# PATH C: Full reasoning chain (Case→Finding→Pattern→Condition←Evidence)
# =========================================================
_QUERY_REASONING_CHAIN = """
MATCH (c:Case {case_id: $case_id})
      -[:HAS_FINDING]->(f:Finding)
      -[:PART_OF_PATTERN]->(p:Pattern)
      -[:SUGGESTS]->(cond:Condition)
      <-[:SUPPORTS]-(e:Evidence)
      -[:FROM_SOURCE]->(src:Source)
WITH DISTINCT
       f.name        AS finding_name,
       f.test_code   AS test_code,
       f.direction   AS direction,
       f.value       AS value,
       f.unit        AS unit,
       p.name        AS pattern_name,
       p.match_score AS pattern_score,
       cond.name     AS condition_name,
       e.id          AS ev_id,
       e.text        AS text,
       e.type        AS ev_type,
       e.page        AS page,
       e.trust       AS trust,
       e.score       AS ev_score,
       src.name      AS src_name
RETURN finding_name, test_code, direction, value, unit,
       pattern_name, pattern_score, condition_name,
       ev_id, text, ev_type, page, trust, src_name
ORDER BY pattern_score DESC, ev_score DESC, trust DESC
LIMIT $limit
"""


def retrieve_reasoning_chain(case_id: str, limit: int = 15) -> list[dict]:
    driver = get_neo4j_driver()
    if driver is None:
        return []
    results: list[dict] = []
    with driver.session() as session:
        try:
            records = session.run(_QUERY_REASONING_CHAIN, case_id=case_id, limit=limit)
            for rec in records:
                results.append(dict(rec))
        except Exception as exc:
            print(f"[neo4j_retriever] Path C error (case={case_id}): {exc}")
    return results


# =========================================================
# PATH D: Test -[INDICATES]→ Condition ←[SUPPORTS]- Evidence
# Chain hoàn chỉnh nhất — graph tự reasoning từ chỉ số → bệnh → bằng chứng
# =========================================================
_QUERY_INDICATES_CHAIN = """
MATCH (t:Test {test_code: $test_code})
      -[:INDICATES {direction: $direction}]->(cond:Condition)
      <-[:SUPPORTS]-(e:Evidence)
      -[:FROM_SOURCE]->(src:Source)
WHERE e.type IN ['interpretation', 'cause']
RETURN e.id AS ev_id, e.text AS text, e.type AS ev_type,
       e.page AS page, e.trust AS trust, e.score AS kb_score,
       e.panel AS panel, src.name AS src_name,
       cond.name AS matched_condition,
       t.test_code AS test_code
ORDER BY e.score DESC, e.trust DESC
LIMIT $limit
"""


def retrieve_by_indicates(abnormal_items: list[dict], limit_per_test: int = 4) -> list[dict]:
    """
    Path D: Test(WBC, high) -[INDICATES]→ Condition ←[SUPPORTS]- Evidence.
    abnormal_items từ reasoning_context, mỗi item có 'test' và 'status'.
    """
    driver = get_neo4j_driver()
    if driver is None:
        return []

    results: list[dict] = []
    with driver.session() as session:
        for item in abnormal_items:
            raw_test  = item.get("test", "") or item.get("test_name", "")
            test_code = normalize_test_code(raw_test)
            direction = (item.get("status", "") or "").lower()

            if not test_code or direction not in ("high", "low"):
                continue

            try:
                records = session.run(
                    _QUERY_INDICATES_CHAIN,
                    test_code=test_code,
                    direction=direction,
                    limit=limit_per_test,
                )
                for rec in records:
                    row = dict(rec)
                    ev = _row_to_evidence(row, score_boost=0.25)
                    ev["retrieval_path"] = (
                        f"neo4j_indicates:{test_code}_{direction}"
                        f"→{row.get('matched_condition', '')}"
                    )
                    ev["conditions"] = [row.get("matched_condition", "")]
                    results.append(ev)
            except Exception as exc:
                if "INDICATES" not in str(exc):
                    print(f"[neo4j_retriever] Path D error ({test_code}_{direction}): {exc}")
    return results


# =========================================================
# MAIN ENTRY — gọi từ lab_core.retrieve_evidence()
# =========================================================
def neo4j_retrieve(reasoning_context: dict) -> list[dict]:
    """
    GraphRAG retrieval từ Neo4j — 3 path song song:
    Path D: Test -[INDICATES]→ Condition ←[SUPPORTS]- Evidence  (boost 0.25)
    Path B: Condition ←[SUPPORTS]- Evidence                      (boost 0.05)
    Path A: Test ←[MENTIONS_TEST]- Evidence                      (không boost)
    """
    driver = get_neo4j_driver()
    if driver is None:
        print("[neo4j_retriever] Neo4j không khả dụng, bỏ qua graph retrieval.")
        return []

    abnormal_items = reasoning_context.get("abnormal_items", [])
    abnormal_tests = reasoning_context.get("abnormal_tests", [])
    conditions     = reasoning_context.get("conditions", [])

    # Normalize test names trong abnormal_tests
    norm_tests = [normalize_test_code(t) for t in abnormal_tests]

    # Lọc condition quá chung
    GENERIC_CONDITIONS = {
        "infection", "inflammation", "anemia", "bleeding",
        "disorder", "abnormality", "pattern", "risk",
    }
    specific_conditions = [
        c for c in conditions
        if c not in GENERIC_CONDITIONS and len(c) > 8
    ]
    query_conditions = specific_conditions if specific_conditions else conditions

    all_evidence: list[dict] = []

    # Path D — chain hoàn chỉnh nhất
    if abnormal_items:
        print(f"[neo4j_retriever] Path D — INDICATES chain: {norm_tests}")
        ev_d = retrieve_by_indicates(abnormal_items, limit_per_test=4)
        print(f"[neo4j_retriever] Path D → {len(ev_d)} evidence")
        all_evidence.extend(ev_d)

    # Path B — theo Condition
    if query_conditions:
        print(f"[neo4j_retriever] Path B — conditions: {query_conditions}")
        ev_b = retrieve_by_conditions(query_conditions, limit=10)
        print(f"[neo4j_retriever] Path B → {len(ev_b)} evidence")
        all_evidence.extend(ev_b)

    # Path A — theo Test, bổ sung
    if norm_tests:
        print(f"[neo4j_retriever] Path A — tests: {norm_tests}")
        ev_a = retrieve_by_tests(norm_tests, limit_per_test=2)
        print(f"[neo4j_retriever] Path A → {len(ev_a)} evidence")
        all_evidence.extend(ev_a)

    return all_evidence


# =========================================================
# QUICK TEST
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NEO4J RETRIEVER — QUICK TEST")
    print("=" * 60)

    mock_context = {
        "abnormal_tests": ["WBC", "NEUT%", "PLT", "LYM%"],
        "conditions":     ["bacterial_infection", "infection", "thrombocytopenia"],
        "abnormal_items": [
            {"test": "WBC",   "status": "high"},
            {"test": "NEUT%", "status": "high"},  # sẽ normalize → NEUT
            {"test": "PLT",   "status": "low"},
            {"test": "LYM%",  "status": "low"},   # sẽ normalize → LYMPH
        ],
    }

    results = neo4j_retrieve(mock_context)
    print(f"\nTổng evidence từ Neo4j: {len(results)}")

    for i, ev in enumerate(results[:8], 1):
        print(f"\n--- Evidence {i} ---")
        print(f"  Path:  {ev['retrieval_path']}")
        print(f"  Type:  {ev['type']}")
        print(f"  Score: {ev['score']:.3f}  Trust: {ev['trust']}")
        print(f"  Src:   {ev['source']} p.{ev['page']}")
        print(f"  Text:  {ev['text'][:180]}...")

    print("\n" + "=" * 60)
    print("REASONING CHAIN TEST (case_id = '1.jpg')")
    print("=" * 60)
    chain = retrieve_reasoning_chain("1.jpg", limit=8)
    seen = set()
    for row in chain:
        key = f"{row.get('test_code')}_{row.get('condition_name')}_{row.get('page')}"
        if key in seen:
            continue
        seen.add(key)
        print(f"  {row.get('finding_name')} → {row.get('pattern_name')} "
              f"→ {row.get('condition_name')} | {row.get('src_name')} p.{row.get('page')}")