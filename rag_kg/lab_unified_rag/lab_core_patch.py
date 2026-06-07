# lab_core_patch.py
# =============================================================
# HƯỚNG DẪN: Áp dụng 3 thay đổi vào lab_core.py
# Chạy script này một lần: python lab_core_patch.py
# Nó sẽ tự động sửa lab_core.py đúng chỗ.
# =============================================================

import re
from pathlib import Path

LAB_CORE = Path("lab_core.py")

if not LAB_CORE.exists():
    print("ERROR: Không tìm thấy lab_core.py. Chạy script này trong thư mục gốc project.")
    exit(1)

content = LAB_CORE.read_text(encoding="utf-8")
original = content  # backup để so sánh


# =============================================================
# THAY ĐỔI 1: Thêm import neo4j_retriever ở đầu file
# Thêm sau dòng "from qdrant_client import QdrantClient"
# =============================================================

OLD_IMPORT = "from qdrant_client import QdrantClient"
NEW_IMPORT = """from qdrant_client import QdrantClient

# Neo4j GraphRAG retriever (thêm vào để hỗ trợ graph retrieval)
try:
    from neo4j_retriever import neo4j_retrieve, retrieve_reasoning_chain
    NEO4J_RETRIEVER_AVAILABLE = True
except ImportError:
    NEO4J_RETRIEVER_AVAILABLE = False
    print("[lab_core] WARNING: neo4j_retriever.py chưa có hoặc thiếu package neo4j.")"""

if OLD_IMPORT in content and "neo4j_retriever" not in content:
    content = content.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("✓ Thay đổi 1: Đã thêm import neo4j_retriever")
elif "neo4j_retriever" in content:
    print("~ Thay đổi 1: Đã có import neo4j_retriever, bỏ qua")
else:
    print("✗ Thay đổi 1 THẤT BẠI: Không tìm thấy dòng import qdrant_client")


# =============================================================
# THAY ĐỔI 2: Sửa retrieve_evidence() để gọi cả Qdrant + Neo4j
# =============================================================

OLD_RETRIEVE = '''def retrieve_evidence(reasoning_context: dict) -> list[dict]:
    queries = build_query_hints(reasoning_context)

    all_evidence: list[dict] = []

    print(f"Query hints: {queries[:5]}")

    for query in queries[:10]:
        try:
            hits = qdrant_search(query, top_k=TOP_K_PER_QUERY)
            all_evidence.extend(hits)
        except Exception as exc:
            print(f"Qdrant search failed for query='{query}': {exc}")

    all_evidence = dedup_evidence(all_evidence)
    all_evidence = rerank_evidence(all_evidence, reasoning_context)

    return all_evidence[:MAX_RAW_EVIDENCE]'''

NEW_RETRIEVE = '''def retrieve_evidence(reasoning_context: dict) -> list[dict]:
    """
    GraphRAG retrieval: kết hợp Qdrant (semantic) + Neo4j (graph relation).
    - Qdrant: tìm evidence gần nghĩa với query
    - Neo4j Path A: Evidence theo Test bất thường (MENTIONS_TEST)
    - Neo4j Path B: Evidence theo Condition từ Pattern (SUPPORTS)
    """
    all_evidence: list[dict] = []

    # --- Qdrant semantic retrieval ---
    queries = build_query_hints(reasoning_context)
    print(f"[retrieve] Qdrant queries: {queries[:3]}")

    for query in queries[:10]:
        try:
            hits = qdrant_search(query, top_k=TOP_K_PER_QUERY)
            all_evidence.extend(hits)
        except Exception as exc:
            print(f"[retrieve] Qdrant failed for query='{query}': {exc}")

    qdrant_count = len(all_evidence)
    print(f"[retrieve] Qdrant → {qdrant_count} evidence (before dedup)")

    # --- Neo4j graph retrieval ---
    if NEO4J_RETRIEVER_AVAILABLE:
        try:
            graph_evidence = neo4j_retrieve(reasoning_context)
            all_evidence.extend(graph_evidence)
            print(f"[retrieve] Neo4j → {len(graph_evidence)} evidence")
        except Exception as exc:
            print(f"[retrieve] Neo4j retrieval failed: {exc}")
    else:
        print("[retrieve] Neo4j không khả dụng, chỉ dùng Qdrant")

    # --- Dedup + rerank ---
    all_evidence = dedup_evidence(all_evidence)
    all_evidence = rerank_evidence(all_evidence, reasoning_context)

    print(f"[retrieve] Sau dedup+rerank: {len(all_evidence)} → trả về top {MAX_RAW_EVIDENCE}")
    return all_evidence[:MAX_RAW_EVIDENCE]'''

if OLD_RETRIEVE in content:
    content = content.replace(OLD_RETRIEVE, NEW_RETRIEVE, 1)
    print("✓ Thay đổi 2: Đã sửa retrieve_evidence() thành GraphRAG")
else:
    print("✗ Thay đổi 2 THẤT BẠI: Không tìm thấy hàm retrieve_evidence() khớp.")
    print("  → Tìm hàm retrieve_evidence() trong lab_core.py và thay thủ công.")


# =============================================================
# THAY ĐỔI 3: Sửa build_reasoning_paths() để dùng Neo4j chain
# Thêm nhánh Neo4j vào đầu hàm trước khi fallback memory
# =============================================================

OLD_PATHS = '''def build_reasoning_paths(reasoning_context: dict, evidence: list[dict]) -> list[dict]:
    paths: list[dict] = []

    abnormal_items = reasoning_context.get("abnormal_items", [])
    patterns = reasoning_context.get("detected_patterns", [])

    for item in abnormal_items:'''

NEW_PATHS = '''def build_reasoning_paths(reasoning_context: dict, evidence: list[dict]) -> list[dict]:
    """
    Xây dựng reasoning paths để đưa vào prompt LLM.
    Ưu tiên dùng Neo4j chain (graph thật), fallback về memory nếu không có.
    """
    # --- Thử Neo4j chain trước ---
    if NEO4J_RETRIEVER_AVAILABLE:
        case_id = reasoning_context.get("case_id", "")
        try:
            chain_rows = retrieve_reasoning_chain(case_id, limit=12)
            if chain_rows:
                # Group theo finding
                from collections import defaultdict
                grouped: dict = defaultdict(lambda: {"patterns": [], "evidence": []})
                for row in chain_rows:
                    key = f"{row.get('test_code')}_{row.get('direction')}"
                    node = grouped[key]
                    node["finding"] = {
                        "panel":           "CBC",
                        "test":            row.get("test_code", ""),
                        "test_label":      row.get("test_code", ""),
                        "status":          row.get("direction", ""),
                        "value":           row.get("value", ""),
                        "unit":            row.get("unit", ""),
                        "reference_range": "",
                    }
                    pat = {
                        "pattern_name": row.get("pattern_name", ""),
                        "conditions":   [row.get("condition_name", "")],
                        "confidence":   row.get("pattern_score", 0),
                    }
                    if pat not in node["patterns"]:
                        node["patterns"].append(pat)
                    ev = {
                        "evidence_id": row.get("ev_id", ""),
                        "source":      row.get("src_name", ""),
                        "page":        row.get("page", ""),
                        "score":       float(row.get("trust") or 0),
                    }
                    if ev not in node["evidence"]:
                        node["evidence"].append(ev)
                paths_neo4j = [
                    {
                        "finding":  v["finding"],
                        "patterns": v["patterns"][:3],
                        "evidence": v["evidence"][:3],
                    }
                    for v in grouped.values() if v.get("finding")
                ]
                if paths_neo4j:
                    return paths_neo4j
        except Exception as exc:
            print(f"[build_reasoning_paths] Neo4j chain failed: {exc}")

    # --- Fallback: build từ memory (logic cũ) ---
    paths: list[dict] = []

    abnormal_items = reasoning_context.get("abnormal_items", [])
    patterns = reasoning_context.get("detected_patterns", [])

    for item in abnormal_items:'''

if OLD_PATHS in content:
    content = content.replace(OLD_PATHS, NEW_PATHS, 1)
    print("✓ Thay đổi 3: Đã sửa build_reasoning_paths() dùng Neo4j chain")
else:
    print("✗ Thay đổi 3 THẤT BẠI: Không tìm thấy hàm build_reasoning_paths() khớp.")
    print("  → Có thể đã được sửa trước đó, kiểm tra thủ công.")


# =============================================================
# GHI FILE
# =============================================================

if content != original:
    # Backup trước khi ghi
    backup = LAB_CORE.with_suffix(".py.bak")
    backup.write_text(original, encoding="utf-8")
    print(f"\n✓ Backup: {backup}")

    LAB_CORE.write_text(content, encoding="utf-8")
    print(f"✓ Đã ghi {LAB_CORE}")
else:
    print("\n~ Không có thay đổi nào được áp dụng.")

print("\nXong! Kiểm tra lại bằng: grep -n 'neo4j_retrieve\\|NEO4J' lab_core.py")