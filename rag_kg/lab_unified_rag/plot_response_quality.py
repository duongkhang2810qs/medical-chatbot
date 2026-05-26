from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


EVAL_DIR = OUTPUT_DIR / "response_quality_eval"
SUMMARY_CSV = EVAL_DIR / "response_quality_summary.csv"


PANEL_LABEL_MAP = {
    "CBC": "Công thức máu",
    "Sinh hóa": "Sinh hóa",
    "BIOCHEM": "Sinh hóa",
    "Trung bình": "Trung bình",
}


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file summary: {path}\n"
            f"Hãy chạy eval_response_quality.py trước."
        )

    df = pd.read_csv(path)

    required_cols = [
        "Panel",
        "Số ca",
        "Đúng",
        "Đủ",
        "Căn cứ",
        "An toàn",
        "Dễ hiểu",
        "Điểm TB",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"File summary thiếu cột: {missing}\n"
            f"Các cột hiện có: {list(df.columns)}"
        )

    df["Panel"] = df["Panel"].map(lambda x: PANEL_LABEL_MAP.get(str(x), str(x)))

    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\nBẢNG TỔNG HỢP")
    print("=" * 100)
    print(df.to_string(index=False))


def add_bar_labels(ax, fontsize: int = 12, padding: int = 4) -> None:
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            fontsize=fontsize,
            fontweight="bold",
            padding=padding,
            rotation=0,
        )


def beautify_ax(ax, title: str, xlabel: str = "Nhóm xét nghiệm") -> None:
    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10)
    ax.set_ylabel("Điểm trung bình", fontsize=13, labelpad=10)

    ax.set_ylim(0, 5.35)

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_full_chart(df: pd.DataFrame) -> None:
    metrics = ["Đúng", "Đủ", "Căn cứ", "An toàn", "Dễ hiểu", "Điểm TB"]

    plot_df = df[df["Panel"].isin(["Công thức máu", "Sinh hóa", "Trung bình"])].copy()

    if plot_df.empty:
        print("Không có dữ liệu để vẽ biểu đồ.")
        return

    ax = plot_df.set_index("Panel")[metrics].plot(
        kind="bar",
        figsize=(15, 7),
        rot=0,
        width=0.82,
    )

    beautify_ax(
        ax,
        title="Tổng hợp đánh giá chất lượng phản hồi RAG + KG",
    )

    ax.legend(
        title="Tiêu chí",
        title_fontsize=12,
        fontsize=11,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
    )

    add_bar_labels(ax, fontsize=11, padding=3)

    plt.tight_layout()
    plt.show()


def plot_criteria_chart(df: pd.DataFrame) -> None:
    metrics = ["Đúng", "Đủ", "Căn cứ", "An toàn", "Dễ hiểu"]

    plot_df = df[df["Panel"].isin(["Công thức máu", "Sinh hóa"])].copy()

    if plot_df.empty:
        print("Không có dữ liệu Công thức máu/Sinh hóa để vẽ biểu đồ.")
        return

    ax = plot_df.set_index("Panel")[metrics].plot(
        kind="bar",
        figsize=(13, 6.5),
        rot=0,
        width=0.75,
    )

    beautify_ax(
        ax,
        title="Đánh giá chất lượng phản hồi theo tiêu chí",
    )

    ax.legend(
        title="Tiêu chí",
        title_fontsize=12,
        fontsize=11,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
    )

    add_bar_labels(ax, fontsize=12, padding=4)

    plt.tight_layout()
    plt.show()


def plot_overall_chart(df: pd.DataFrame) -> None:
    plot_df = df[df["Panel"].isin(["Công thức máu", "Sinh hóa", "Trung bình"])].copy()

    if plot_df.empty:
        print("Không có dữ liệu để vẽ biểu đồ điểm TB.")
        return

    ax = plot_df.plot(
        x="Panel",
        y="Điểm TB",
        kind="bar",
        legend=False,
        figsize=(9, 6),
        rot=0,
        width=0.55,
    )

    beautify_ax(
        ax,
        title="Điểm trung bình chất lượng phản hồi",
    )

    add_bar_labels(ax, fontsize=13, padding=5)

    plt.tight_layout()
    plt.show()


def plot_horizontal_summary(df: pd.DataFrame) -> None:
    """
    Biểu đồ ngang, dễ đưa vào báo cáo nếu biểu đồ cột bị chật.
    """
    metrics = ["Đúng", "Đủ", "Căn cứ", "An toàn", "Dễ hiểu", "Điểm TB"]

    plot_df = df[df["Panel"].isin(["Công thức máu", "Sinh hóa", "Trung bình"])].copy()

    if plot_df.empty:
        print("Không có dữ liệu để vẽ biểu đồ ngang.")
        return

    ax = plot_df.set_index("Panel")[metrics].plot(
        kind="barh",
        figsize=(13, 7),
        width=0.78,
    )

    ax.set_title(
        "Tổng hợp đánh giá chất lượng phản hồi",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("Điểm trung bình", fontsize=13, labelpad=10)
    ax.set_ylabel("Nhóm xét nghiệm", fontsize=13, labelpad=10)

    ax.set_xlim(0, 5.35)

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        title="Tiêu chí",
        title_fontsize=12,
        fontsize=11,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            fontsize=8,
            padding=3,
        )

    plt.tight_layout()
    plt.show()


def main(chart: str = "full") -> None:
    df = load_summary(SUMMARY_CSV)
    print_summary(df)

    if chart == "full":
        plot_full_chart(df)
    elif chart == "criteria":
        plot_criteria_chart(df)
    elif chart == "overall":
        plot_overall_chart(df)
    elif chart == "horizontal":
        plot_horizontal_summary(df)
    elif chart == "all":
        plot_full_chart(df)
        plot_criteria_chart(df)
        plot_overall_chart(df)
        plot_horizontal_summary(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chart",
        type=str,
        default="full",
        choices=["full", "criteria", "overall", "horizontal", "all"],
        help="Loại chart: full, criteria, overall, horizontal, all",
    )

    args = parser.parse_args()

    main(chart=args.chart)