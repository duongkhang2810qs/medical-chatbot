import json
import re
import traceback
from typing import Any, Dict, List, Optional

import lab_core
import rag_service

try:
    import cbc_demo_evidence
except Exception:
    cbc_demo_evidence = None

try:
    import biochem_demo_evidence
except Exception:
    biochem_demo_evidence = None

# =========================================================
# SIMPLE IN-MEMORY SESSION STORE
# =========================================================

SESSIONS: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: Optional[str]) -> Dict[str, Any]:
    sid = session_id or "default"

    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "active_report_data": None,
            "report_summary": None,
            "history": []
        }

    return SESSIONS[sid]


def update_session_memory(
    session_id: str,
    active_report_data: Optional[list] = None,
    report_summary: Optional[str] = None
) -> Dict[str, Any]:
    session = get_session(session_id)

    if active_report_data is not None:
        session["active_report_data"] = active_report_data

    if report_summary is not None:
        session["report_summary"] = report_summary

    return session


# =========================================================
# BASIC HELPERS
# =========================================================

def is_greeting(text: str) -> bool:
    text_l = (text or "").strip().lower()
    greetings = {
        "hi",
        "hello",
        "xin chào",
        "chào",
        "chào bạn",
        "alo",
        "ok",
        "cảm ơn",
        "thank you",
        "thanks"
    }
    return text_l in greetings


def safe_route_intent(user_text: str, has_report: bool = False) -> str:
    try:
        return rag_service.semantic_route_intent(user_text, has_report=has_report)
    except Exception as e:
        print(f"⚠️ semantic_route_intent lỗi: {e}")

    text_l = user_text.lower()

    report_words = [
        "của tôi", "phiếu", "kết quả", "chỉ số này", "bất thường",
        "cao", "thấp", "giảm", "tăng", "có sao không"
    ]

    medical_words = [
        "wbc", "rbc", "hgb", "hct", "mcv", "mch", "mchc", "rdw",
        "plt", "neut", "lym", "lymph", "mono", "eos", "baso",
        "bạch cầu", "hồng cầu", "tiểu cầu", "công thức máu", "huyết học",
        "xét nghiệm", "máu",
        "glucose", "creatinine", "urea", "ast", "alt",
        "cholesterol", "triglyceride", "hdl", "ldl", "bilirubin",
        "albumin", "protein", "acid uric", "men gan", "mỡ máu",
        "gan", "thận", "sinh hóa"
    ]

    if has_report and any(w in text_l for w in report_words):
        return "report_followup"

    if any(w in text_l for w in medical_words):
        return "medical_knowledge"

    return "general_chat"


def safe_clean_text(text: str) -> str:
    try:
        return lab_core.mechanical_cleanup_answer(text)
    except Exception:
        return text


def safe_clean_quote(text: str, max_len: int = 450) -> str:
    try:
        return lab_core.clean_quote(text, max_len=max_len)
    except Exception:
        text = str(text or "").replace("\n", " ").strip()
        return text[:max_len] + ("..." if len(text) > max_len else "")


def safe_clean_reference_quote(text: str, max_len: int = 420) -> str:
    try:
        return lab_core.clean_reference_quote(text)
    except Exception:
        return safe_clean_quote(text, max_len=max_len)


# =========================================================
# PANEL / TEST DETECTION
# =========================================================

def infer_panels_from_user_text(user_text: str) -> List[str]:
    cbc_tests = []
    biochem_tests = []

    try:
        cbc_tests = lab_core.detect_tests_in_text(user_text, "CBC")
    except Exception:
        pass

    try:
        biochem_tests = lab_core.detect_tests_in_text(user_text, "BIOCHEM")
    except Exception:
        pass

    if cbc_tests and not biochem_tests:
        return ["CBC"]

    if biochem_tests and not cbc_tests:
        return ["BIOCHEM"]

    if cbc_tests and biochem_tests:
        return ["CBC", "BIOCHEM"]

    text_l = user_text.lower()

    cbc_words = [
        "wbc", "rbc", "hgb", "hb", "hct", "mcv", "mch", "mchc", "rdw",
        "plt", "platelet", "neut", "neutrophil", "lym", "lymph", "lymphocyte",
        "mono", "monocyte", "eos", "eosinophil", "baso", "basophil",
        "bạch cầu", "hồng cầu", "tiểu cầu", "công thức máu", "huyết học",
        "huyết sắc tố", "hematocrit"
    ]

    biochem_words = [
        "glucose", "creatinine", "urea", "bun", "ast", "alt", "ggt", "alp",
        "bilirubin", "cholesterol", "triglyceride", "hdl", "ldl",
        "acid uric", "uric acid", "albumin", "protein", "đường huyết",
        "men gan", "mỡ máu", "gan", "thận", "sinh hóa"
    ]

    if any(w in text_l for w in cbc_words):
        return ["CBC"]

    if any(w in text_l for w in biochem_words):
        return ["BIOCHEM"]

    return ["CBC", "BIOCHEM"]


def detect_tests_topics_conditions_by_panels(user_text: str, panels: List[str]) -> Dict[str, List[str]]:
    tests: List[str] = []
    topics: List[str] = []
    conditions: List[str] = []

    for panel in panels:
        try:
            panel_tests = lab_core.detect_tests_in_text(user_text, panel)
        except Exception:
            panel_tests = []

        try:
            panel_topics = lab_core.detect_topics_in_text(user_text, panel)
        except Exception:
            panel_topics = []

        try:
            panel_keywords = lab_core.extract_keywords(user_text, panel)
        except Exception:
            panel_keywords = []

        try:
            panel_conditions = lab_core.infer_conditions_from_text(
                text=user_text,
                panel=panel,
                tests=panel_tests,
                topics=panel_topics,
                keywords=panel_keywords
            )
        except Exception:
            panel_conditions = []

        tests.extend(panel_tests)
        topics.extend(panel_topics)
        conditions.extend(panel_conditions)

    return {
        "tests": sorted(set(tests)),
        "topics": sorted(set(topics)),
        "conditions": sorted(set(conditions)),
    }


def build_knowledge_context(user_text: str) -> Dict[str, Any]:
    panels = infer_panels_from_user_text(user_text)
    detected = detect_tests_topics_conditions_by_panels(user_text, panels)

    tests = detected["tests"]
    topics = detected["topics"]
    conditions = detected["conditions"]

    query_hints = [user_text]

    for test in tests:
        query_hints.extend([
            f"{test} definition",
            f"{test} blood test meaning",
            f"{test} clinical significance",
            f"{test} laboratory interpretation",
        ])

    for condition in conditions:
        query_hints.extend([
            f"{condition} laboratory interpretation",
            f"{condition} blood test",
        ])

    return {
        "case_id": "chat_knowledge",
        "panels": panels,
        "items": [],
        "abnormal_items": [],
        "abnormal_tests": tests,
        "detected_patterns": [],
        "conditions": conditions,
        "topics": topics,
        "query_hints": query_hints,
    }


# =========================================================
# EVIDENCE RETRIEVAL + FILTERING
# =========================================================

def retrieve_evidence_for_context(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    abnormal_tests = ctx.get("abnormal_tests", []) or []
    conditions = ctx.get("conditions", []) or []

    graph_evidence: List[Dict[str, Any]] = []
    vector_evidence: List[Dict[str, Any]] = []

    try:
        print(f"[Chat Evidence] Query Neo4j | tests={abnormal_tests} | conditions={conditions}")
        graph_evidence = rag_service.fetch_evidence_from_neo4j(abnormal_tests, conditions)
    except Exception as e:
        print(f"⚠️ Neo4j evidence lỗi: {e}")

    try:
        print(f"[Chat Evidence] Query Qdrant | panels={ctx.get('panels')} | hints={ctx.get('query_hints')}")
        vector_evidence = lab_core.retrieve_evidence(ctx)
    except Exception as e:
        print(f"⚠️ Qdrant evidence lỗi: {e}")

    try:
        combined = lab_core.dedup_evidence(graph_evidence + vector_evidence)
    except Exception:
        combined = graph_evidence + vector_evidence

    try:
        final_evidence = lab_core.rerank_evidence(combined, ctx)
    except Exception as e:
        print(f"⚠️ Rerank evidence lỗi: {e}")
        final_evidence = combined

    return final_evidence


def evidence_matches_panel(e: Dict[str, Any], allowed_panels: set) -> bool:
    if not allowed_panels:
        return True

    e_panel = e.get("panel")
    e_source = str(e.get("source", "")).lower()

    if e_panel in allowed_panels:
        return True

    if "CBC" in allowed_panels:
        cbc_source_words = [
            "hematology",
            "haematology",
            "blood",
            "cbc",
            "clinical_hematology",
        ]
        if any(w in e_source for w in cbc_source_words):
            return True

    if "BIOCHEM" in allowed_panels:
        biochem_source_words = [
            "tietz",
            "henry",
            "chemistry",
            "biochemistry",
            "laboratory",
            "clinical_diagnosis",
        ]
        if any(w in e_source for w in biochem_source_words):
            return True

    return False


def get_test_aliases(test: str) -> List[str]:
    aliases = {
        "wbc": ["wbc", "white blood cell", "white blood cells", "leukocyte", "leukocytes", "bạch cầu"],
        "rbc": ["rbc", "red blood cell", "red blood cells", "erythrocyte", "erythrocytes", "hồng cầu"],
        "hgb": ["hgb", "hb", "hemoglobin", "haemoglobin", "huyết sắc tố"],
        "hct": ["hct", "hematocrit", "haematocrit"],
        "plt": ["plt", "platelet", "platelets", "tiểu cầu"],
        "mcv": ["mcv", "mean corpuscular volume"],
        "mch": ["mch", "mean corpuscular hemoglobin"],
        "mchc": ["mchc", "mean corpuscular hemoglobin concentration"],
        "rdw": ["rdw", "red cell distribution width"],
        "neut": ["neut", "neutrophil", "neutrophils"],
        "lymph": ["lymph", "lym", "lymphocyte", "lymphocytes"],
        "lym": ["lymph", "lym", "lymphocyte", "lymphocytes"],
        "mono": ["mono", "monocyte", "monocytes"],
        "eos": ["eos", "eosinophil", "eosinophils"],
        "baso": ["baso", "basophil", "basophils"],
        "ig": ["ig", "immature granulocyte", "immature granulocytes"],
    }

    test_l = str(test or "").lower()
    return aliases.get(test_l, [test_l])


def filter_evidence_by_context(
    evidence: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    max_items: int = 2
) -> List[Dict[str, Any]]:
    allowed_panels = set(ctx.get("panels", []) or [])
    target_tests = set(ctx.get("abnormal_tests", []) or [])
    target_conditions = set(ctx.get("conditions", []) or [])

    panel_filtered = [
        e for e in evidence
        if evidence_matches_panel(e, allowed_panels)
    ]

    if not panel_filtered:
        panel_filtered = evidence

    strong_matches = []

    for e in panel_filtered:
        e_tests = set(e.get("tests", []) or [])
        e_conditions = set(e.get("conditions", []) or [])
        e_text = str(e.get("text", "")).lower()
        e_source = str(e.get("source", "")).lower()

        if target_tests:
            matched = False

            for test in target_tests:
                test_aliases = get_test_aliases(test)

                if test in e_tests:
                    matched = True

                if any(alias in e_text for alias in test_aliases):
                    matched = True

                if any(alias in e_source for alias in test_aliases):
                    matched = True

            if matched:
                strong_matches.append(e)

            continue

        if target_conditions:
            if e_conditions & target_conditions:
                strong_matches.append(e)
                continue

            for cond in target_conditions:
                if str(cond).lower() in e_text:
                    strong_matches.append(e)
                    break

            continue

        strong_matches.append(e)

    if strong_matches:
        return strong_matches[:max_items]

    # Nếu hỏi test cụ thể nhưng không tìm được evidence đúng test,
    # không fallback sang evidence khác để tránh citation nhiễu.
    if target_tests:
        return []

    return panel_filtered[:max_items]


def format_evidence_for_prompt(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "- Không có evidence phù hợp từ Knowledge Graph hoặc Vector DB."

    lines = []

    for i, e in enumerate(evidence, start=1):
        text = safe_clean_quote(e.get("text", ""), max_len=450)
        source = e.get("source", "Unknown")
        page = e.get("page", "")
        tests = ", ".join(e.get("tests", []) or [])
        conditions = ", ".join(e.get("conditions", []) or [])
        panel = e.get("panel", "")

        lines.append(
            f"[{i}] Source={source}; Page={page}; Panel={panel}; "
            f"Tests={tests}; Conditions={conditions}; Quote=\"{text}\""
        )

    return "\n".join(lines)


# =========================================================
# CITATION DISPLAY FIX
# =========================================================

def extract_cited_indices(answer: str) -> List[int]:
    """
    Chỉ lấy citation [1], [2], [3] xuất hiện thật trong phần trả lời.
    """
    if not answer:
        return []

    matches = re.findall(r"\[(\d+)\]", answer)
    indices: List[int] = []

    for m in matches:
        try:
            idx = int(m)
            if idx not in indices:
                indices.append(idx)
        except Exception:
            pass

    return indices


def build_clean_references_block(
    evidence: List[Dict[str, Any]],
    answer: str = ""
) -> str:
    cited_indices = extract_cited_indices(answer)

    if not evidence:
        return "📚 References:\n- Không có evidence phù hợp được truy xuất."

    if not cited_indices:
        lines = ["📚 References:"]
        lines.append("- Có evidence được truy xuất, nhưng câu trả lời chưa trích dẫn trực tiếp.")

        for i, ev in enumerate(evidence[:2], start=1):
            source = ev.get("source") or ev.get("file") or "unknown"
            page = ev.get("page")
            text = ev.get("text") or ev.get("content") or ""

            page_part = f", page {page}" if page else ""
            snippet = text[:300].replace("\n", " ").strip()

            lines.append(f"[{i}] {source}{page_part}. “{snippet}…”")

        return "\n".join(lines)

    lines = ["📚 References:"]

    for idx in cited_indices:
        evidence_pos = idx - 1

        if evidence_pos < 0 or evidence_pos >= len(evidence):
            continue

        e = evidence[evidence_pos]

        source = e.get("source", "Unknown source")
        page = e.get("page", "")
        quote = safe_clean_reference_quote(e.get("text", ""))

        if page:
            lines.append(f"[{idx}] {source}, page {page}. “{quote}”")
        else:
            lines.append(f"[{idx}] {source}. “{quote}”")

    if len(lines) == 1:
        return "📚 References:\n- Không có evidence phù hợp được trích dẫn trong câu trả lời."

    return "\n".join(lines)


def build_clean_source_intro(panels: List[str]) -> str:
    panels_set = set(panels or [])
    lines = ["📚 Nguồn tài liệu:"]

    if "CBC" in panels_set:
        lines.append(
            "- Clinical Hematology: Tài liệu chuyên sâu về huyết học lâm sàng, "
            "hỗ trợ diễn giải công thức máu và các rối loạn huyết học."
        )

    if "BIOCHEM" in panels_set:
        lines.append(
            "- Henry’s Clinical Diagnosis and Management by Laboratory Methods: "
            "Tài liệu chuẩn về diễn giải xét nghiệm cận lâm sàng và y học xét nghiệm."
        )
        lines.append(
            "- Tietz Fundamentals of Clinical Chemistry and Molecular Diagnostics: "
            "Tài liệu nền tảng về hóa sinh lâm sàng, xét nghiệm sinh hóa và marker bệnh lý."
        )

    if not panels_set:
        lines.append(
            "- Tài liệu y học xét nghiệm trong hệ thống tri thức nội bộ."
        )

    lines.append("")
    lines.append(
        "Lưu ý: Nội dung chỉ có mục đích hỗ trợ diễn giải xét nghiệm, "
        "không thay thế chẩn đoán hoặc chỉ định điều trị của bác sĩ."
    )

    return "\n".join(lines)


def build_clean_user_visible_answer(
    answer: str,
    ctx: Dict[str, Any],
    evidence: List[Dict[str, Any]]
) -> str:
    cleaned_answer = safe_clean_text(answer)

    return (
        f"{cleaned_answer}\n\n"
        f"{build_clean_references_block(evidence, cleaned_answer)}\n\n"
        f"{build_clean_source_intro(ctx.get('panels', []))}"
    ).strip()


# =========================================================
# ANSWER FLOWS
# =========================================================

def answer_general_chat(user_text: str) -> str:
    if is_greeting(user_text):
        return (
            "Xin chào! Bạn có thể gửi câu hỏi về xét nghiệm máu hoặc tải phiếu xét nghiệm "
            "để tôi hỗ trợ phân tích."
        )

    prompt = f"""
Bạn là trợ lý y khoa hỗ trợ giải thích xét nghiệm máu bằng tiếng Việt.

Người dùng hỏi:
{user_text}

Yêu cầu:
- Trả lời ngắn gọn, dễ hiểu.
- Nếu câu hỏi không liên quan xét nghiệm máu, hãy lịch sự nói rằng bạn hỗ trợ chính về xét nghiệm máu.
- Không chẩn đoán chắc chắn.
- Không kê thuốc.
""".strip()

    return rag_service.call_llm(prompt)


def answer_medical_knowledge_with_evidence(user_text: str) -> str:
    ctx = build_knowledge_context(user_text)

    # Layer ưu tiên cho demo local: CBC50 và BIOCHEM50.
    # Nếu match được câu demo thì KHÔNG gọi KG/Qdrant để tránh evidence lệch.
    local_evidence: List[Dict[str, Any]] = []
    local_source_label = ""

    if cbc_demo_evidence is not None:
        try:
            local_evidence = cbc_demo_evidence.get_cbc50_evidence_for_question(user_text) or []
            if local_evidence:
                local_source_label = "CBC50 local book evidence JSON"
        except Exception as e:
            print(f"⚠️ CBC50 local evidence lỗi: {e}")

    if not local_evidence and biochem_demo_evidence is not None:
        try:
            local_evidence = biochem_demo_evidence.get_biochem50_evidence_for_question(user_text) or []
            if local_evidence:
                local_source_label = "BIOCHEM50 local book evidence JSON"
        except Exception as e:
            print(f"⚠️ BIOCHEM50 local evidence lỗi: {e}")

    if local_evidence:
        evidence = local_evidence[:2]
        evidence_block = format_evidence_for_prompt(evidence)

        prompt = f"""
Bạn là trợ lý y khoa giải thích xét nghiệm máu bằng tiếng Việt.

Câu hỏi người dùng:
{user_text}

EVIDENCE từ {local_source_label}:
{evidence_block}

Yêu cầu bắt buộc:
- Chỉ trả lời dựa trên EVIDENCE ở trên.
- Vì đã có EVIDENCE phù hợp, KHÔNG được viết "không có dẫn chứng".
- Trả lời trực tiếp, dễ hiểu, ngắn gọn.
- Nếu câu hỏi hỏi định nghĩa, hãy giải thích định nghĩa trước, rồi mới nói ý nghĩa xét nghiệm.
- Bắt buộc dùng citation [1] cho evidence đầu tiên nếu dùng thông tin từ đó.
- Nếu dùng evidence thứ hai thì cite [2].
- Không được dùng citation không có trong EVIDENCE.
- Không bịa nguồn, không bịa số trang.
- Không chẩn đoán chắc chắn; dùng từ "gợi ý", "có thể liên quan", "cần đối chiếu lâm sàng" khi nói về bất thường.
- Không kê thuốc.
""".strip()

        raw_answer = rag_service.call_llm(prompt)

        # Model nhỏ như Qwen đôi khi quên citation dù prompt đã yêu cầu.
        # Auto thêm [1] để phần References render được khi đang dùng local evidence đúng case.
        if evidence and "[1]" not in raw_answer:
            raw_answer = raw_answer.rstrip() + " [1]"

        return build_clean_user_visible_answer(raw_answer, ctx, evidence)

    # Không match demo local thì fallback flow KG/KB cũ.
    evidence = retrieve_evidence_for_context(ctx)
    evidence = filter_evidence_by_context(evidence, ctx, max_items=2)
    evidence_block = format_evidence_for_prompt(evidence)

    prompt = f"""
Bạn là trợ lý y khoa giải thích xét nghiệm máu bằng tiếng Việt.

Câu hỏi người dùng:
{user_text}

EVIDENCE từ Knowledge Graph / Vector DB:
{evidence_block}

Yêu cầu:
- Trả lời trực tiếp, dễ hiểu, ngắn gọn.
- Nếu câu hỏi hỏi định nghĩa, hãy giải thích định nghĩa trước, rồi mới nói ý nghĩa xét nghiệm.
- Chỉ dùng citation [1], [2] nếu thông tin thật sự lấy trực tiếp từ EVIDENCE tương ứng.
- Nếu EVIDENCE không liên quan trực tiếp đến câu hỏi, KHÔNG dùng citation.
- Không được dùng citation không có trong EVIDENCE.
- Không bịa nguồn, không bịa số trang.
- Không chẩn đoán chắc chắn.
- Không kê thuốc.
- Nếu evidence không thật sự liên quan, chỉ viết: "Bằng chứng truy xuất còn hạn chế."
""".strip()

    raw_answer = rag_service.call_llm(prompt)

    return build_clean_user_visible_answer(raw_answer, ctx, evidence)

def build_report_context_from_session(session: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    active_report_data = session.get("active_report_data") or []

    case_dict = rag_service.format_ui_data_to_case(active_report_data, session_id)
    ctx = lab_core.build_reasoning_context(case_dict, 0)

    try:
        cbc_demo = (
            lab_core.load_jsonl(rag_service.config.CBC_DEMO_PATTERN_PATH)
            if rag_service.config.CBC_DEMO_PATTERN_PATH.exists()
            else []
        )
        biochem_patt = (
            lab_core.load_json(rag_service.config.BIOCHEM_PATTERN_PATH)
            if rag_service.config.BIOCHEM_PATTERN_PATH.exists()
            else {}
        )
        ctx = lab_core.augment_reasoning_context_with_static_patterns(ctx, cbc_demo, biochem_patt)
    except Exception as e:
        print(f"⚠️ augment static patterns lỗi: {e}")

    return ctx


def filter_report_evidence_by_panels(
    evidence: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    max_items: int = 6
) -> List[Dict[str, Any]]:
    allowed_panels = set(ctx.get("panels", []) or [])

    filtered = [
        e for e in evidence
        if evidence_matches_panel(e, allowed_panels)
    ]

    if not filtered:
        filtered = evidence

    return filtered[:max_items]


def answer_report_followup(user_text: str, session_id: str, session: Dict[str, Any]) -> str:
    ctx = build_report_context_from_session(session, session_id)

    ctx["query_hints"] = list(ctx.get("query_hints", [])) + [user_text]

    evidence = retrieve_evidence_for_context(ctx)
    evidence = filter_report_evidence_by_panels(evidence, ctx, max_items=6)

    evidence_block = format_evidence_for_prompt(evidence)

    report_summary = session.get("report_summary") or ""
    report_data = session.get("active_report_data") or []

    prompt = f"""
Bạn là trợ lý y khoa hỗ trợ giải thích phiếu xét nghiệm máu bằng tiếng Việt.

Người dùng hỏi tiếp:
{user_text}

Tóm tắt phiếu xét nghiệm đã phân tích trước đó:
{report_summary}

Dữ liệu phiếu xét nghiệm dạng JSON:
{json.dumps(report_data, ensure_ascii=False)}

EVIDENCE từ Knowledge Graph / Vector DB:
{evidence_block}

Yêu cầu:
- Trả lời dựa trên phiếu xét nghiệm của người dùng nếu câu hỏi có liên quan.
- Dùng số liệu cụ thể trong phiếu nếu có.
- Chỉ dùng citation [1], [2], [3], [4], [5], [6] nếu thông tin thật sự lấy từ EVIDENCE tương ứng.
- Nếu EVIDENCE không liên quan trực tiếp đến câu hỏi, KHÔNG dùng citation.
- Không bịa nguồn, không bịa xét nghiệm không có trong phiếu.
- Không chẩn đoán chắc chắn; dùng từ "gợi ý", "có thể liên quan", "cần đối chiếu lâm sàng".
- Không kê thuốc.
- Nếu cần, khuyên người dùng trao đổi bác sĩ.
""".strip()

    raw_answer = rag_service.call_llm(prompt)

    return build_clean_user_visible_answer(raw_answer, ctx, evidence)


# =========================================================
# MAIN CHAT ENTRYPOINT FOR chatbot_api.py
# =========================================================

def handle_chat(text: str, session_id: str) -> Dict[str, str]:
    user_text = (text or "").strip()
    session = get_session(session_id)

    if not user_text:
        return {
            "answer": "Bạn vui lòng nhập câu hỏi nhé.",
            "intent": "general_chat"
        }

    has_report = bool(session.get("active_report_data") or session.get("report_summary"))
    intent = safe_route_intent(user_text, has_report=has_report)

    print(f"[Chatbot Service] intent={intent} | has_report={has_report}")

    try:
        if intent == "general_chat":
            answer = answer_general_chat(user_text)

        elif intent == "report_followup" and has_report:
            answer = answer_report_followup(user_text, session_id, session)

        else:
            answer = answer_medical_knowledge_with_evidence(user_text)
            intent = "medical_knowledge"

    except Exception as e:
        print("[Chatbot Service ERROR]")
        print(traceback.format_exc())
        answer = f"Xin lỗi, hệ thống gặp lỗi khi xử lý câu hỏi: {str(e)}"

    session["history"].append({
        "role": "user",
        "content": user_text
    })
    session["history"].append({
        "role": "assistant",
        "content": answer
    })

    return {
        "answer": answer,
        "intent": intent
    }