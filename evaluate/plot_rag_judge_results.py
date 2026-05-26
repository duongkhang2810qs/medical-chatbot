from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

EVALUATE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EVALUATE_DIR / "rag_llm_judge_results.csv"
DEFAULT_OUT_DIR = EVALUATE_DIR / "charts"

SCORE_COLUMNS = [
    ("context_relevance_score", "Context relevance"),
    ("source_attribution_score", "Source attribution"),
    ("faithfulness_score", "Faithfulness"),
    ("medical_safety_score", "Medical safety"),
]


def parse_score(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_bar_chart(labels, values, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"][: len(labels)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, max(values + [1]) * 1.18)
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}" if isinstance(value, float) else str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_score_distribution(rows: list[dict[str, str]], out_dir: Path) -> None:
    labels = ["0", "0.5", "1"]
    x = range(len(labels))

    plt.figure(figsize=(11, 6))
    width = 0.2

    for idx, (col, name) in enumerate(SCORE_COLUMNS):
        counts = Counter(parse_score(row.get(col, "")) for row in rows)
        values = [counts.get(0.0, 0), counts.get(0.5, 0), counts.get(1.0, 0)]
        offsets = [pos + (idx - 1.5) * width for pos in x]
        plt.bar(offsets, values, width=width, label=name)

    plt.title("RAG judge score distribution")
    plt.xlabel("Score")
    plt.ylabel("Number of cases")
    plt.xticks(list(x), labels)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_distribution.png", dpi=180)
    plt.close()


def save_panel_summary(rows: list[dict[str, str]], out_dir: Path) -> None:
    panel_scores: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        panel = row.get("panel", "UNKNOWN") or "UNKNOWN"
        panel_scores[panel].append(parse_score(row.get("total_rag_4", "")))

    labels = sorted(panel_scores)
    values = [sum(panel_scores[label]) / len(panel_scores[label]) for label in labels]
    save_bar_chart(labels, values, "Average total RAG score by panel", "Average / 4", out_dir / "avg_total_by_panel.png")


def save_metric_average(rows: list[dict[str, str]], out_dir: Path) -> None:
    labels = [name for _, name in SCORE_COLUMNS]
    values = []

    for col, _name in SCORE_COLUMNS:
        scores = [parse_score(row.get(col, "")) for row in rows]
        values.append(sum(scores) / len(scores) if scores else 0.0)

    save_bar_chart(labels, values, "Average score by RAG criterion", "Average / 1", out_dir / "avg_by_criterion.png")


def save_total_histogram(rows: list[dict[str, str]], out_dir: Path) -> None:
    scores = [parse_score(row.get("total_rag_4", "")) for row in rows]
    bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]

    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=bins, color="#4C78A8", edgecolor="white")
    plt.title("Total RAG score histogram")
    plt.xlabel("Total score / 4")
    plt.ylabel("Number of cases")
    plt.xticks([0, 1, 2, 3, 4])
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "total_score_histogram.png", dpi=180)
    plt.close()


def write_summary_csv(rows: list[dict[str, str]], out_dir: Path) -> None:
    total_scores = [parse_score(row.get("total_rag_4", "")) for row in rows]
    summary_rows = [
        ("case_count", len(rows)),
        ("avg_total_rag_4", round(sum(total_scores) / len(total_scores), 3) if total_scores else 0),
        ("min_total_rag_4", min(total_scores) if total_scores else 0),
        ("max_total_rag_4", max(total_scores) if total_scores else 0),
    ]

    for col, name in SCORE_COLUMNS:
        scores = [parse_score(row.get(col, "")) for row in rows]
        summary_rows.append((f"avg_{col}", round(sum(scores) / len(scores), 3) if scores else 0))
        summary_rows.append((f"{col}_perfect_count", sum(1 for score in scores if score == 1.0)))
        summary_rows.append((f"{col}_zero_count", sum(1 for score in scores if score == 0.0)))

    with (out_dir / "rag_judge_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RAG judge result charts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input}")

    ensure_out_dir(args.out_dir)
    save_metric_average(rows, args.out_dir)
    save_score_distribution(rows, args.out_dir)
    save_panel_summary(rows, args.out_dir)
    save_total_histogram(rows, args.out_dir)
    write_summary_csv(rows, args.out_dir)

    print(f"Loaded rows: {len(rows)}")
    print(f"Charts written to: {args.out_dir}")


if __name__ == "__main__":
    main()
