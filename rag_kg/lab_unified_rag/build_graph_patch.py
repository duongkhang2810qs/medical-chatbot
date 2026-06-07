# build_graph_patch.py
"""
Patch thêm INDICATES edges vào build_graph.py.
Chạy: python build_graph_patch.py
Script tự backup build_graph.py → build_graph.py.bak rồi sửa 2 chỗ.
"""
from pathlib import Path

TARGET = Path("build_graph.py")

if not TARGET.exists():
    print("ERROR: Không tìm thấy build_graph.py. Chạy trong thư mục project.")
    exit(1)

content = TARGET.read_text(encoding="utf-8")
original = content

# ── THAY ĐỔI 1: Thêm hàm add_indicates_to_graph sau hàm add_kb_to_graph ──

NEW_FUNC = '''

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

'''

# Tìm vị trí sau hàm add_kb_to_graph để chèn hàm mới
ANCHOR = "# =========================================================\n# STATIC PATTERN MATCHING"

if ANCHOR in content:
    content = content.replace(ANCHOR, NEW_FUNC + "\n" + ANCHOR, 1)
    print("✓ Thay đổi 1: Đã thêm hàm add_indicates_to_graph()")
else:
    print("✗ Thay đổi 1 THẤT BẠI: Không tìm thấy anchor. Thêm thủ công.")


# ── THAY ĐỔI 2: Gọi add_indicates_to_graph trong main() ──

OLD_MAIN_CALL = "    add_cases_to_graph(\n        g=g,"
NEW_MAIN_CALL = """    add_indicates_to_graph(g)

    add_cases_to_graph(
        g=g,"""

if OLD_MAIN_CALL in content:
    content = content.replace(OLD_MAIN_CALL, NEW_MAIN_CALL, 1)
    print("✓ Thay đổi 2: Đã thêm lệnh gọi add_indicates_to_graph() trong main()")
else:
    print("✗ Thay đổi 2 THẤT BẠI: Không tìm thấy anchor trong main(). Thêm thủ công.")


# ── GHI FILE ──
if content != original:
    backup = TARGET.with_suffix(".py.bak")
    backup.write_text(original, encoding="utf-8")
    print(f"\n✓ Backup: {backup}")
    TARGET.write_text(content, encoding="utf-8")
    print(f"✓ Đã ghi {TARGET}")
else:
    print("\n~ Không có thay đổi nào được áp dụng.")

print("\nXong! Kiểm tra: grep -n 'INDICATES\\|add_indicates' build_graph.py")