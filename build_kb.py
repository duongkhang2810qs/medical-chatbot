# build_kb.py
from __future__ import annotations

from config import (
    CBC_PDF_PATHS,
    BIOCHEM_PDF_PATHS,
    CBC_KB_PATH,
    BIOCHEM_KB_PATH,
    LAB_KB_PATH,
)
from lab_core import (
    build_kb_from_pdf_paths,
    save_json,
)


def main():
    print("=" * 80)
    print("BUILD FULL KB FROM PDF SOURCES")
    print("=" * 80)

    print("\nStep 1: Build CBC KB")
    cbc_kb = build_kb_from_pdf_paths(
        pdf_paths=CBC_PDF_PATHS,
        panel="CBC",
    )

    save_json(CBC_KB_PATH, cbc_kb)
    print(f"Saved CBC KB: {CBC_KB_PATH}")
    print(f"CBC chunks: {len(cbc_kb)}")

    print("\nStep 2: Build BIOCHEM KB")
    biochem_kb = build_kb_from_pdf_paths(
        pdf_paths=BIOCHEM_PDF_PATHS,
        panel="BIOCHEM",
    )

    save_json(BIOCHEM_KB_PATH, biochem_kb)
    print(f"Saved BIOCHEM KB: {BIOCHEM_KB_PATH}")
    print(f"BIOCHEM chunks: {len(biochem_kb)}")

    print("\nStep 3: Save merged LAB KB preview")
    lab_kb = cbc_kb + biochem_kb
    save_json(LAB_KB_PATH, lab_kb)

    print(f"Saved merged LAB KB: {LAB_KB_PATH}")
    print(f"Total chunks: {len(lab_kb)}")

    print("\nDONE")
    print("Next step:")
    print("python build_index.py")


if __name__ == "__main__":
    main()