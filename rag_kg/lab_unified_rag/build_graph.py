# build_graph.py
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from config import (
    CBC_CASE_PATH,
    BIOCHEM_CASE_PATH,
    LAB_CASE_PATH,
    LAB_KB_PATH,
    LAB_GRAPH_PATH,
    NEO4J_DIR,
    TEST_LABELS,
    CBC_DEMO_PATTERN_PATH,
    BIOCHEM_PATTERN_PATH,
)
from lab_core import (
    load_json,
    load_jsonl,
    save_json,
    merge_case_lists,
    build_reasoning_context,
    normalize_test_name,
    normalize_status,
    slugify,
    stable_hash,
)


# =========================================================
# CSV SCHEMA giống style KG cũ của bạn
# =========================================================

NODE_SCHEMAS = {
    "Panel": ["id", "name", "display_name"],
    "Test": ["id", "test_code", "name", "panel"],
    "Status": ["id", "name", "display_name"],
    "Condition": ["id", "name", "canonical_name"],
    "Source": ["id", "name", "kind"],
    "Topic": ["id", "name"],
    "Case": ["id", "case_id", "summary", "combined_interpretation", "warning", "confidence", "panel"],
    "Finding": ["id", "test_code", "direction", "value", "ref", "raw_status", "name", "panel", "unit"],
    "Pattern": ["id", "name", "interpretation", "confidence_label", "match_score", "source", "panel"],
    "Evidence": ["id", "text", "record_type", "type", "page", "trust", "score", "panel", "source"],
}

EDGE_SCHEMA = [
    "id",
    "source_type",
    "source_id",
    "relation",
    "target_type",
    "target_id",
    "confidence",
    "provenance",
]


# =========================================================
# HELPERS
# =========================================================

def as_json_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def node_id_condition(name: str) -> str:
    return f"cond_{slugify(name)}"


def node_id_topic(name: str) -> str:
    return f"topic_{slugify(name)}"


def node_id_source(name: str) -> str:
    return f"src_{slugify(name)}"


def node_id_panel(panel: str) -> str:
    return f"panel_{slugify(panel)}"


def node_id_test(test: str) -> str:
    return f"test_{test}"


def node_id_status(status: str) -> str:
    return f"status_{slugify(status)}"


def node_id_case(case_id: str) -> str:
    return f"case_{slugify(case_id)}"


def node_id_finding(case_id: str, panel: str, test: str, status: str) -> str:
    return f"finding_{slugify(case_id)}_{panel}_{test}_{status}"


def node_id_pattern(pattern_name: str, panel: str = "", source: str = "") -> str:
    raw = f"{panel}_{source}_{pattern_name}"
    return f"pattern_{slugify(raw)}"


def normalize_static_tag(tag: str, panel: str) -> tuple[str, str] | None:
    """
    Input examples:
    - WBC_high
    - NEUT_PERCENT_high
    - UREA_High
    - CREATININE_High
    """

    if not tag:
        return None

    raw = str(tag).strip()

    if "_" not in raw:
        return None

    parts = raw.split("_")
    status_raw = parts[-1]
    test_raw = "_".join(parts[:-1])

    test = normalize_test_name(test_raw, panel)
    status = normalize_status(status_raw)

    if not test or not status:
        return None

    return test, status


def case_tag_sets(ctx: dict) -> dict[str, set[str]]:
    """
    Tạo nhiều variant để match static pattern:
    - UREA_High
    - UREA_high
    - BIOCHEM_UREA_High
    """

    out = {
        "simple_title": set(),
        "simple_lower": set(),
        "panel_title": set(),
        "panel_lower": set(),
    }

    for item in ctx.get("abnormal_items", []):
        panel = item.get("panel")
        test = item.get("test")
        status = normalize_status(item.get("status"))

        if not panel or not test or not status:
            continue

        status_title = status.capitalize()

        out["simple_title"].add(f"{test}_{status_title}")
        out["simple_lower"].add(f"{test}_{status}")
        out["panel_title"].add(f"{panel}_{test}_{status_title}")
        out["panel_lower"].add(f"{panel}_{test}_{status}")

    return out


def has_case_tag(ctx: dict, panel: str, test: str, status: str) -> bool:
    panel = str(panel).upper()
    test = normalize_test_name(test, panel)
    status = normalize_status(status)

    for item in ctx.get("abnormal_items", []):
        if (
            item.get("panel") == panel
            and item.get("test") == test
            and normalize_status(item.get("status")) == status
        ):
            return True

    return False


def find_matching_finding_ids(ctx: dict, panel: str, tag_list: list[str]) -> list[str]:
    case_id = ctx["case_id"]
    out = []

    for tag in tag_list:
        parsed = normalize_static_tag(tag, panel)

        if not parsed:
            continue

        test, status = parsed

        for item in ctx.get("abnormal_items", []):
            if (
                item.get("panel") == panel
                and item.get("test") == test
                and normalize_status(item.get("status")) == status
            ):
                out.append(node_id_finding(case_id, panel, test, status))

    return sorted(set(out))


def confidence_to_float(value: Any, default: float = 0.7) -> float:
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip().lower()

    label_map = {
        "low": 0.55,
        "medium": 0.7,
        "high": 0.9,
    }

    if s in label_map:
        return label_map[s]

    try:
        return float(s)
    except Exception:
        return default


def condition_from_pattern_name(name: str) -> str:
    return slugify(name)


# =========================================================
# GRAPH CONTAINER
# =========================================================

class GraphBuilder:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.edge_count = 0

    def add_node(self, node_type: str, node_id: str, **props):
        current = self.nodes.get(node_id)

        payload = {
            "id": node_id,
            "node_type": node_type,
            **props,
        }

        if current is None:
            self.nodes[node_id] = payload
        else:
            current.update({k: v for k, v in payload.items() if v not in [None, ""]})

    def add_edge(
        self,
        source_type: str,
        source_id: str,
        relation: str,
        target_type: str,
        target_id: str,
        confidence: float = 1.0,
        provenance: str = "",
    ):
        if not source_id or not target_id:
            return

        self.edge_count += 1

        self.edges.append({
            "id": f"edge_{self.edge_count:07d}",
            "source_type": source_type,
            "source_id": source_id,
            "relation": relation,
            "target_type": target_type,
            "target_id": target_id,
            "confidence": round(float(confidence or 0), 4),
            "provenance": provenance,
        })

    def dedupe_edges(self):
        seen = set()
        out = []

        for edge in self.edges:
            key = (
                edge["source_type"],
                edge["source_id"],
                edge["relation"],
                edge["target_type"],
                edge["target_id"],
            )

            if key not in seen:
                seen.add(key)
                out.append(edge)

        for idx, edge in enumerate(out, start=1):
            edge["id"] = f"edge_{idx:07d}"

        self.edges = out
        self.edge_count = len(out)

    def to_graph_json(self) -> dict:
        self.dedupe_edges()

        node_type_counts = {}
        edge_type_counts = {}

        for node in self.nodes.values():
            t = node.get("node_type", "Unknown")
            node_type_counts[t] = node_type_counts.get(t, 0) + 1

        for edge in self.edges:
            r = edge.get("relation", "Unknown")
            edge_type_counts[r] = edge_type_counts.get(r, 0) + 1

        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "stats": {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "node_type_counts": node_type_counts,
                "edge_type_counts": edge_type_counts,
            },
        }


# =========================================================
# LOAD INPUT
# =========================================================

def load_cases_for_graph() -> list[dict]:
    if LAB_CASE_PATH.exists():
        print(f"Loading merged case file: {LAB_CASE_PATH}")
        return load_jsonl(LAB_CASE_PATH)

    print("Merged case file chưa có, đọc 2 file riêng CBC + BIOCHEM...")

    cbc_cases = load_jsonl(CBC_CASE_PATH)
    biochem_cases = load_jsonl(BIOCHEM_CASE_PATH)

    print(f"CBC cases: {len(cbc_cases)}")
    print(f"BIOCHEM cases: {len(biochem_cases)}")

    merged_cases = merge_case_lists(cbc_cases, biochem_cases)

    print(f"Merged cases by same id/case_id: {len(merged_cases)}")

    return merged_cases


def load_cbc_demo_patterns() -> list[dict]:
    if not CBC_DEMO_PATTERN_PATH.exists():
        print(f"Warning: missing CBC demo pattern file: {CBC_DEMO_PATTERN_PATH}")
        return []

    return load_jsonl(CBC_DEMO_PATTERN_PATH)


def load_biochem_patterns() -> dict:
    if not BIOCHEM_PATTERN_PATH.exists():
        print(f"Warning: missing BIOCHEM pattern file: {BIOCHEM_PATTERN_PATH}")
        return {}

    data = load_json(BIOCHEM_PATTERN_PATH)

    if not isinstance(data, dict):
        raise ValueError("biochem_patterns.json must be a JSON object")

    return data


# =========================================================
# ADD KB TO GRAPH
# =========================================================

def add_kb_to_graph(g: GraphBuilder, kb: list[dict]):
    print("Adding KB evidence nodes...")

    for idx, item in enumerate(kb):
        panel = str(item.get("panel", "UNKNOWN")).upper()
        text = str(item.get("text", ""))
        source = str(item.get("source", "Unknown"))
        evidence_id = item.get("evidence_id") or f"ev_{panel.lower()}_{idx:06d}_{stable_hash(text)}"

        panel_id = node_id_panel(panel)
        source_id = node_id_source(source)

        g.add_node("Panel", panel_id, name=panel, display_name=panel)

        g.add_node(
            "Source",
            source_id,
            name=source,
            kind="pdf_book",
        )

        g.add_node(
            "Evidence",
            evidence_id,
            text=text,
            record_type="book_evidence",
            type=item.get("type", "general"),
            page=item.get("page", ""),
            trust=item.get("trust", 0.8),
            score=item.get("score", 0),
            panel=panel,
            source=source,
        )

        g.add_edge("Evidence", evidence_id, "FROM_SOURCE", "Source", source_id, item.get("trust", 0.8), "kb_pdf")
        g.add_edge("Evidence", evidence_id, "BELONGS_TO_PANEL", "Panel", panel_id, 1.0, "kb_pdf")

        for test in item.get("tests", []):
            test_id = node_id_test(test)

            g.add_node(
                "Test",
                test_id,
                test_code=test,
                name=TEST_LABELS.get(test, test),
                panel=panel,
            )

            g.add_edge("Evidence", evidence_id, "MENTIONS_TEST", "Test", test_id, 0.95, "kb_pdf")
            g.add_edge("Test", test_id, "BELONGS_TO_PANEL", "Panel", panel_id, 1.0, "ontology")

        for topic in item.get("topics", []):
            topic_id = node_id_topic(topic)

            g.add_node("Topic", topic_id, name=topic)
            g.add_edge("Evidence", evidence_id, "MENTIONS_TOPIC", "Topic", topic_id, 0.8, "kb_pdf")

        for condition in item.get("conditions", []):
            condition_id = node_id_condition(condition)

            g.add_node(
                "Condition",
                condition_id,
                name=condition,
                canonical_name=condition,
            )

            g.add_edge("Evidence", evidence_id, "SUPPORTS", "Condition", condition_id, 0.75, "kb_pdf")




# =========================================================
# ADD INDICATES EDGES (Test → Condition)
# =========================================================

def add_indicates_to_graph(g: GraphBuilder):
    """
    Thêm quan hệ INDICATES từ Test đến Condition dựa trên
    bảng mapping lâm sàng trong indicates_mapping.py.

    Quan hệ này cho phép graph tự reasoning:
    Test(WBC, direction=high) --[INDICATES]--> Condition(bacterial_infection)
    thay vì phải đi qua Python pattern matching.
    """
    try:
        from indicates_mapping import INDICATES_MAPPING
    except ImportError:
        print("WARNING: Không tìm thấy indicates_mapping.py — bỏ qua INDICATES edges.")
        return

    print(f"Adding INDICATES edges từ {len(INDICATES_MAPPING)} mapping rules...")

    for test_code, direction, condition_name, confidence, panel, source_pattern in INDICATES_MAPPING:
        test_id      = node_id_test(test_code)
        condition_id = node_id_condition(condition_name)
        panel_id     = node_id_panel(panel)

        # Đảm bảo các node đã tồn tại
        g.add_node(
            "Test", test_id,
            test_code=test_code,
            name=TEST_LABELS.get(test_code, test_code),
            panel=panel,
        )
        g.add_node(
            "Condition", condition_id,
            name=condition_name,
            canonical_name=condition_name,
        )
        g.add_node("Panel", panel_id, name=panel, display_name=panel)

        # Thêm edge INDICATES với thuộc tính direction và source_pattern
        g.add_edge(
            "Test", test_id,
            "INDICATES",
            "Condition", condition_id,
            confidence=confidence,
            provenance=source_pattern,
        )

    print(f"Done: {len(INDICATES_MAPPING)} INDICATES edges added.")


# =========================================================
# STATIC PATTERN MATCHING
# =========================================================

def match_cbc_demo_patterns(ctx: dict, demo_patterns: list[dict]) -> list[dict]:
    """
    Match CBC demo pattern file:
    demo input: {"WBC": "high", "NEUT": "high"}
    """

    matches = []

    for row in demo_patterns:
        raw_input = row.get("input", {}) or {}

        required = []

        for raw_test, raw_status in raw_input.items():
            test = normalize_test_name(raw_test, "CBC")
            status = normalize_status(raw_status)
            required.append((test, status))

        if not required:
            continue

        hit = 0

        for test, status in required:
            if has_case_tag(ctx, "CBC", test, status):
                hit += 1

        ratio = hit / max(len(required), 1)

        # Giữ demo pattern nếu match đủ tốt.
        # Case 1 tag vẫn match nếu full hit.
        if hit == 0:
            continue

        if len(required) >= 2 and ratio < 0.6:
            continue

        if len(required) == 1 and ratio < 1.0:
            continue

        patterns = row.get("patterns", []) or []

        for pattern in patterns:
            matched_conditions = pattern.get("matched_conditions", []) or []

            matches.append({
                "pattern_id": f"cbc_demo_{pattern.get('pattern_name', row.get('case_id', 'pattern'))}",
                "pattern_name": pattern.get("pattern_name", "CBC demo pattern"),
                "panel": "CBC",
                "interpretation": pattern.get("interpretation") or row.get("combined_interpretation", ""),
                "confidence_label": pattern.get("confidence", row.get("confidence", "")),
                "match_score": confidence_to_float(pattern.get("match_score", ratio), ratio),
                "conditions": [
                    condition_from_pattern_name(pattern.get("pattern_name", "cbc_pattern"))
                ],
                "matched_tags": matched_conditions,
                "source": "cbc_demo_cases",
                "raw": pattern,
            })

    return matches


def match_biochem_static_patterns(ctx: dict, biochem_patterns: dict) -> list[dict]:
    matches = []

    tags = case_tag_sets(ctx)
    simple_title = tags["simple_title"]

    # -------------------------
    # Single-test patterns
    # -------------------------
    single = biochem_patterns.get("single_test_patterns", {}) or {}

    for item in ctx.get("abnormal_items", []):
        if item.get("panel") != "BIOCHEM":
            continue

        test = item.get("test")
        status = normalize_status(item.get("status"))

        if not test or not status:
            continue

        test_rule = single.get(test, {}) or {}
        status_rule = test_rule.get(status, {}) or {}

        if not status_rule:
            continue

        pattern_name = status_rule.get("label") or f"{test} {status}"
        interpretation = status_rule.get("clinical_meaning") or status_rule.get("note") or ""

        matches.append({
            "pattern_id": f"biochem_single_{test}_{status}",
            "pattern_name": pattern_name,
            "panel": "BIOCHEM",
            "interpretation": interpretation,
            "confidence_label": "high",
            "match_score": 0.85,
            "conditions": [condition_from_pattern_name(pattern_name)],
            "matched_tags": [f"{test}_{status.capitalize()}"],
            "source": "biochem_patterns_single",
            "raw": status_rule,
        })

    # -------------------------
    # Combination patterns
    # -------------------------
    combos = biochem_patterns.get("pattern_combinations", []) or []

    for combo in combos:
        required_tags = combo.get("required_tags", []) or []
        optional_tags = combo.get("optional_tags", []) or []
        confidence_required = int(combo.get("confidence_required", 1) or 1)

        required_hits = [tag for tag in required_tags if tag in simple_title]
        optional_hits = [tag for tag in optional_tags if tag in simple_title]

        if required_tags:
            if len(required_hits) < min(len(required_tags), confidence_required):
                continue

            if len(required_hits) < len(required_tags):
                continue
        else:
            if len(optional_hits) < confidence_required:
                continue

        total_possible = len(required_tags) + len(optional_tags)
        total_hits = len(required_hits) + len(optional_hits)
        match_score = total_hits / max(total_possible, 1)

        matches.append({
            "pattern_id": combo.get("pattern_id", "biochem_combo_pattern"),
            "pattern_name": combo.get("name", "BIOCHEM combination pattern"),
            "panel": "BIOCHEM",
            "interpretation": combo.get("interpretation", ""),
            "confidence_label": "high" if match_score >= 0.7 else "medium",
            "match_score": round(match_score, 3),
            "conditions": [condition_from_pattern_name(combo.get("pattern_id", combo.get("name", "biochem_pattern")))],
            "matched_tags": required_hits + optional_hits,
            "source": "biochem_patterns_combo",
            "raw": combo,
        })

    return matches


# =========================================================
# ADD CASES TO GRAPH
# =========================================================

def add_cases_to_graph(
    g: GraphBuilder,
    cases: list[dict],
    cbc_demo_patterns: list[dict],
    biochem_patterns: dict,
):
    print("Adding case/finding/pattern nodes...")

    for idx, case in enumerate(cases):
        ctx = build_reasoning_context(case, idx)
        case_id = ctx["case_id"]
        case_node = node_id_case(case_id)

        panels = ctx.get("panels", [])
        panel_value = "+".join(panels)

        detected_patterns = ctx.get("detected_patterns", [])
        static_cbc_matches = match_cbc_demo_patterns(ctx, cbc_demo_patterns)
        static_biochem_matches = match_biochem_static_patterns(ctx, biochem_patterns)

        all_static_matches = static_cbc_matches + static_biochem_matches

        combined_interpretation_parts = []

        for p in detected_patterns[:3]:
            if p.get("description"):
                combined_interpretation_parts.append(p["description"])

        for p in all_static_matches[:3]:
            if p.get("interpretation"):
                combined_interpretation_parts.append(p["interpretation"])

        g.add_node(
            "Case",
            case_node,
            case_id=case_id,
            summary=f"{len(ctx.get('abnormal_items', []))} abnormal findings",
            combined_interpretation=" ; ".join(combined_interpretation_parts[:5]),
            warning="This is not a medical diagnosis. Results are for reference only.",
            confidence="",
            panel=panel_value,
        )

        # -------------------------
        # Findings
        # -------------------------
        for item in ctx.get("abnormal_items", []):
            panel = item.get("panel")
            test = item.get("test")
            status = normalize_status(item.get("status"))
            value = item.get("value")
            unit = item.get("unit", "")
            ref = item.get("reference_range", "")

            panel_id = node_id_panel(panel)
            test_id = node_id_test(test)
            status_id = node_id_status(status)
            finding_id = node_id_finding(case_id, panel, test, status)

            g.add_node("Panel", panel_id, name=panel, display_name=panel)

            g.add_node(
                "Test",
                test_id,
                test_code=test,
                name=TEST_LABELS.get(test, test),
                panel=panel,
            )

            g.add_node(
                "Status",
                status_id,
                name=status,
                display_name=status.capitalize(),
            )

            g.add_node(
                "Finding",
                finding_id,
                test_code=test,
                direction=status,
                value=value,
                ref=ref,
                raw_status=item.get("status", status),
                name=f"{panel} {test} {status}",
                panel=panel,
                unit=unit,
            )

            g.add_edge("Case", case_node, "HAS_FINDING", "Finding", finding_id, 1.0, case_id)
            g.add_edge("Finding", finding_id, "OF_TEST", "Test", test_id, 1.0, case_id)
            g.add_edge("Finding", finding_id, "HAS_STATUS", "Status", status_id, 1.0, case_id)
            g.add_edge("Test", test_id, "BELONGS_TO_PANEL", "Panel", panel_id, 1.0, "ontology")

        # -------------------------
        # Rule patterns from config.py / lab_core.py
        # -------------------------
        for pattern in detected_patterns:
            pattern_id = node_id_pattern(pattern.get("pattern_id"), pattern.get("panel", ""), "rule")

            g.add_node(
                "Pattern",
                pattern_id,
                name=pattern.get("pattern_name"),
                interpretation=pattern.get("description", ""),
                confidence_label="",
                match_score=pattern.get("confidence", 0),
                source=pattern.get("source", "rule"),
                panel=pattern.get("panel", ""),
            )

            g.add_edge("Case", case_node, "MATCHES_PATTERN", "Pattern", pattern_id, pattern.get("confidence", 0.7), case_id)

            # Link finding -> pattern by matched tags
            matched_tags = pattern.get("matched_required", []) + pattern.get("matched_optional", [])

            for panel in ["CBC", "BIOCHEM"]:
                for finding_id in find_matching_finding_ids(ctx, panel, matched_tags):
                    g.add_edge("Finding", finding_id, "PART_OF_PATTERN", "Pattern", pattern_id, pattern.get("confidence", 0.7), case_id)

            for condition in pattern.get("conditions", []):
                condition_id = node_id_condition(condition)

                g.add_node(
                    "Condition",
                    condition_id,
                    name=condition,
                    canonical_name=condition,
                )

                g.add_edge("Pattern", pattern_id, "SUGGESTS", "Condition", condition_id, pattern.get("confidence", 0.7), case_id)

        # -------------------------
        # Static patterns from demo_cases_all_clean + biochem_patterns
        # -------------------------
        for pattern in all_static_matches:
            pattern_id = node_id_pattern(pattern.get("pattern_id"), pattern.get("panel", ""), pattern.get("source", "static"))

            confidence = confidence_to_float(pattern.get("match_score"), 0.7)

            g.add_node(
                "Pattern",
                pattern_id,
                name=pattern.get("pattern_name"),
                interpretation=pattern.get("interpretation", ""),
                confidence_label=pattern.get("confidence_label", ""),
                match_score=pattern.get("match_score", confidence),
                source=pattern.get("source", "static"),
                panel=pattern.get("panel", ""),
            )

            g.add_edge("Case", case_node, "MATCHES_PATTERN", "Pattern", pattern_id, confidence, case_id)

            panel = pattern.get("panel")
            matched_tags = pattern.get("matched_tags", []) or []

            for finding_id in find_matching_finding_ids(ctx, panel, matched_tags):
                g.add_edge("Finding", finding_id, "PART_OF_PATTERN", "Pattern", pattern_id, confidence, case_id)

            for condition in pattern.get("conditions", []):
                condition_id = node_id_condition(condition)

                g.add_node(
                    "Condition",
                    condition_id,
                    name=condition,
                    canonical_name=condition,
                )

                g.add_edge("Pattern", pattern_id, "SUGGESTS", "Condition", condition_id, confidence, case_id)


# =========================================================
# EXPORT
# =========================================================

def export_old_style_csv(graph: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_by_type: dict[str, list[dict]] = {k: [] for k in NODE_SCHEMAS.keys()}

    for node in graph.get("nodes", []):
        node_type = node.get("node_type")

        if node_type in nodes_by_type:
            nodes_by_type[node_type].append(node)

    file_map = {
        "Panel": "nodes_panel.csv",
        "Test": "nodes_test.csv",
        "Status": "nodes_status.csv",
        "Condition": "nodes_condition.csv",
        "Source": "nodes_source.csv",
        "Topic": "nodes_topic.csv",
        "Case": "nodes_case.csv",
        "Finding": "nodes_finding.csv",
        "Pattern": "nodes_pattern.csv",
        "Evidence": "nodes_evidence.csv",
    }

    for node_type, filename in file_map.items():
        path = output_dir / filename
        schema = NODE_SCHEMAS[node_type]

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=schema)
            writer.writeheader()

            for node in nodes_by_type[node_type]:
                row = {key: as_json_text(node.get(key, "")) for key in schema}
                writer.writerow(row)

        print(f"Exported {filename}: {len(nodes_by_type[node_type])}")

    edges_path = output_dir / "edges_all.csv"

    with open(edges_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EDGE_SCHEMA)
        writer.writeheader()

        for edge in graph.get("edges", []):
            row = {key: as_json_text(edge.get(key, "")) for key in EDGE_SCHEMA}
            writer.writerow(row)

    print(f"Exported edges_all.csv: {len(graph.get('edges', []))}")


def write_neo4j_import_cypher(output_dir: Path):
    cypher = r"""
// ==============================================
// CLEAN OLD GRAPH OPTIONAL
// Chỉ chạy dòng dưới nếu muốn xóa graph cũ:
// MATCH (n) DETACH DELETE n;
// ==============================================

// ==============================================
// CONSTRAINTS
// ==============================================
CREATE CONSTRAINT panel_id IF NOT EXISTS FOR (n:Panel) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT test_id IF NOT EXISTS FOR (n:Test) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT status_id IF NOT EXISTS FOR (n:Status) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT condition_id IF NOT EXISTS FOR (n:Condition) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (n:Topic) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT case_id IF NOT EXISTS FOR (n:Case) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT finding_id IF NOT EXISTS FOR (n:Finding) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (n:Pattern) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:Evidence) REQUIRE n.id IS UNIQUE;

// ==============================================
// IMPORT NODES
// ==============================================
LOAD CSV WITH HEADERS FROM 'file:///nodes_panel.csv' AS row
MERGE (n:Panel {id: row.id})
SET n.name = row.name,
    n.display_name = row.display_name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_test.csv' AS row
MERGE (n:Test {id: row.id})
SET n.test_code = row.test_code,
    n.name = row.name,
    n.panel = row.panel;

LOAD CSV WITH HEADERS FROM 'file:///nodes_status.csv' AS row
MERGE (n:Status {id: row.id})
SET n.name = row.name,
    n.display_name = row.display_name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_condition.csv' AS row
MERGE (n:Condition {id: row.id})
SET n.name = row.name,
    n.canonical_name = row.canonical_name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_source.csv' AS row
MERGE (n:Source {id: row.id})
SET n.name = row.name,
    n.kind = row.kind;

LOAD CSV WITH HEADERS FROM 'file:///nodes_topic.csv' AS row
MERGE (n:Topic {id: row.id})
SET n.name = row.name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_case.csv' AS row
MERGE (n:Case {id: row.id})
SET n.case_id = row.case_id,
    n.summary = row.summary,
    n.combined_interpretation = row.combined_interpretation,
    n.warning = row.warning,
    n.confidence = row.confidence,
    n.panel = row.panel;

LOAD CSV WITH HEADERS FROM 'file:///nodes_finding.csv' AS row
MERGE (n:Finding {id: row.id})
SET n.test_code = row.test_code,
    n.direction = row.direction,
    n.value = row.value,
    n.ref = row.ref,
    n.raw_status = row.raw_status,
    n.name = row.name,
    n.panel = row.panel,
    n.unit = row.unit;

LOAD CSV WITH HEADERS FROM 'file:///nodes_pattern.csv' AS row
MERGE (n:Pattern {id: row.id})
SET n.name = row.name,
    n.interpretation = row.interpretation,
    n.confidence_label = row.confidence_label,
    n.match_score = toFloat(coalesce(row.match_score,'0')),
    n.source = row.source,
    n.panel = row.panel;

LOAD CSV WITH HEADERS FROM 'file:///nodes_evidence.csv' AS row
MERGE (n:Evidence {id: row.id})
SET n.text = row.text,
    n.record_type = row.record_type,
    n.type = row.type,
    n.page = row.page,
    n.trust = toFloat(coalesce(row.trust, '0')),
    n.score = toFloat(coalesce(row.score, '0')),
    n.panel = row.panel,
    n.source = row.source;

// ==============================================
// IMPORT EDGES
// Dùng 1 relationship type chung RELATED_TO
// property r.type giữ relation thật.
// ==============================================
LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:RELATED_TO {id: row.id}]->(t)
SET r.type = row.relation,
    r.confidence = toFloat(coalesce(row.confidence,'0')),
    r.provenance = row.provenance,
    r.source_type = row.source_type,
    r.target_type = row.target_type;

// ==============================================
// DEMO QUERIES
// ==============================================

// 1. Thống kê node
MATCH (n)
RETURN labels(n) AS label, count(n) AS cnt
ORDER BY cnt DESC;

// 2. Thống kê relation
MATCH ()-[r:RELATED_TO]->()
RETURN r.type AS relation_type, count(*) AS cnt
ORDER BY cnt DESC;

// 3. Case -> Finding -> Test -> Status
MATCH (c:Case)-[r1:RELATED_TO]->(f:Finding),
      (f)-[r2:RELATED_TO]->(t:Test),
      (f)-[r3:RELATED_TO]->(s:Status)
WHERE r1.type = 'HAS_FINDING'
  AND r2.type = 'OF_TEST'
  AND r3.type = 'HAS_STATUS'
RETURN c, r1, f, r2, t, r3, s
LIMIT 30;

// 4. Case -> Finding -> Pattern -> Condition
MATCH (c:Case)-[r1:RELATED_TO]->(f:Finding),
      (f)-[r2:RELATED_TO]->(p:Pattern),
      (p)-[r3:RELATED_TO]->(cond:Condition)
WHERE r1.type = 'HAS_FINDING'
  AND r2.type = 'PART_OF_PATTERN'
  AND r3.type = 'SUGGESTS'
RETURN c, r1, f, r2, p, r3, cond
LIMIT 30;

// 5. Reasoning đầy đủ + evidence text
MATCH (c:Case)-[r1:RELATED_TO]->(f:Finding),
      (f)-[r2:RELATED_TO]->(p:Pattern),
      (p)-[r3:RELATED_TO]->(cond:Condition),
      (e:Evidence)-[r4:RELATED_TO]->(cond)
WHERE r1.type = 'HAS_FINDING'
  AND r2.type = 'PART_OF_PATTERN'
  AND r3.type = 'SUGGESTS'
  AND r4.type = 'SUPPORTS'
RETURN c.case_id AS case_id,
       f.name AS finding,
       p.name AS pattern,
       cond.name AS condition,
       left(e.text, 220) AS evidence_text,
       e.source AS source,
       e.page AS page,
       e.score AS score,
       e.trust AS trust
LIMIT 30;

// 6. Cross-panel cases
MATCH (c:Case)
WHERE c.panel CONTAINS 'CBC' AND c.panel CONTAINS 'BIOCHEM'
RETURN c
LIMIT 20;

// 7. Evidence theo panel
MATCH (e:Evidence)
RETURN e.panel AS panel, count(e) AS evidence_count
ORDER BY evidence_count DESC;
""".strip()

    path = output_dir / "neo4j_import_unified.cypher"

    with open(path, "w", encoding="utf-8") as f:
        f.write(cypher)

    print(f"Exported neo4j_import_unified.cypher")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 80)
    print("BUILD UNIFIED CBC + BIOCHEM KNOWLEDGE GRAPH")
    print("=" * 80)

    if not LAB_KB_PATH.exists():
        raise FileNotFoundError(
            f"Không thấy {LAB_KB_PATH}. Hãy chạy build_kb.py và build_index.py trước."
        )

    print(f"Loading LAB KB: {LAB_KB_PATH}")
    kb = load_json(LAB_KB_PATH)

    if not isinstance(kb, list):
        raise ValueError("lab_kb.json must be a JSON list")

    print(f"KB chunks: {len(kb)}")

    cases = load_cases_for_graph()
    print(f"Cases: {len(cases)}")

    cbc_demo_patterns = load_cbc_demo_patterns()
    biochem_patterns = load_biochem_patterns()

    print(f"CBC demo pattern rows: {len(cbc_demo_patterns)}")
    print(f"BIOCHEM pattern sections: {list(biochem_patterns.keys())[:5]}")

    g = GraphBuilder()

    add_kb_to_graph(g, kb)
    add_indicates_to_graph(g)

    add_cases_to_graph(
        g=g,
        cases=cases,
        cbc_demo_patterns=cbc_demo_patterns,
        biochem_patterns=biochem_patterns,
    )

    graph = g.to_graph_json()

    save_json(LAB_GRAPH_PATH, graph)

    print("\nExporting Neo4j CSV files...")
    export_old_style_csv(graph, NEO4J_DIR)
    write_neo4j_import_cypher(NEO4J_DIR)

    print("\nDONE")
    print(f"Graph JSON: {LAB_GRAPH_PATH}")
    print(f"Neo4j CSV dir: {NEO4J_DIR}")
    print(f"Nodes: {graph['stats']['num_nodes']}")
    print(f"Edges: {graph['stats']['num_edges']}")
    print("\nNode types:")
    for k, v in graph["stats"]["node_type_counts"].items():
        print(f"  {k}: {v}")
    print("\nEdge types:")
    for k, v in graph["stats"]["edge_type_counts"].items():
        print(f"  {k}: {v}")

    print("\nNext:")
    print("Copy CSV files in neo4j_csv/ to Neo4j import folder, then run neo4j_import_unified.cypher")


if __name__ == "__main__":
    main()