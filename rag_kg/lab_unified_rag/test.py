# .\.venv\Scripts\Activate.ps1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

data = {
    "Nhóm xét nghiệm": ["Công thức máu", "Sinh hóa", "Trung bình"],
    "Đúng": [4.9091, 4.8571, 4.8966],
    "Đủ": [4.8636, 4.7143, 4.8276],
    "Căn cứ": [3.7273, 3.7143, 3.7241],
    "An toàn": [4.6364, 4.8571, 4.6897],
    "Dễ hiểu": [4.7727, 4.4286, 4.6897],
    "Điểm TB": [4.5818, 4.5143, 4.5655],
}

df = pd.DataFrame(data).set_index("Nhóm xét nghiệm")

# Bảng màu xanh dương - xanh ngọc - xanh lá, dịu mắt hơn
cmap = LinearSegmentedColormap.from_list(
    "blue_teal_green",
    ["#DCEEFF", "#A7D8F0", "#7FD6C2", "#8BD17C", "#CDEB7B"]
)

fig, ax = plt.subplots(figsize=(8.2, 3.0))

im = ax.imshow(
    df.values,
    aspect="auto",
    cmap=cmap,
    vmin=3.5,
    vmax=5.0
)

ax.set_xticks(np.arange(len(df.columns)))
ax.set_yticks(np.arange(len(df.index)))

ax.set_xticklabels(df.columns, fontsize=12)
ax.set_yticklabels(df.index, fontsize=12)

plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

for i in range(len(df.index)):
    for j in range(len(df.columns)):
        value = df.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#1F2933"
        )

ax.set_title(
    "Đánh giá chất lượng phản hồi RAG + KG",
    fontsize=15,
    fontweight="bold",
    pad=14
)

# Kẻ ô trắng nhẹ
ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
ax.tick_params(which="minor", bottom=False, left=False)

# Bỏ viền ngoài cho sạch hơn
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.ax.tick_params(labelsize=10)
cbar.set_label("Điểm", fontsize=11)

plt.tight_layout()
plt.savefig("rag_kg_quality_heatmap_blue_green.png", dpi=400, bbox_inches="tight")
plt.show()