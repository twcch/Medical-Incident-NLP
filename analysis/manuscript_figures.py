"""期刊論文第 3、4 章的五張圖（論文用的命名：LoRA-S＝第一輪的 v1，LoRA-L＝第二輪的 v2）。

    ms_fig1_framework.png       研究架構
    ms_fig2_discrimination.png  測試集 AUROC 與 macro-F1 的 95% 信賴區間
    ms_fig3_training.png        驗證損失與訓練損失（以累計更新步數對齊）
    ms_fig4_gain.png            累積增益曲線
    ms_fig5_inputs.png          不同輸入來源的 AUROC

樣式與 analysis/report_charts.py 相同（白底、藍 #2a78d6／橘 #eb6834），只在 macOS 驗證過。
在專案根目錄執行（先跑 analysis.idt_comparison 與 analysis.idt_inputs）：
    python -m analysis.manuscript_figures                 # → results/analysis/ms_fig*.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

ROOT = Path.cwd()
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/analysis"
for needed in ("idt_comparison.json", "idt_inputs.json"):
    if not (OUT / needed).is_file():
        sys.exit(f"找不到 {OUT / needed}；請先執行 python -m analysis.idt_comparison 與 analysis.idt_inputs。")
data = json.loads((OUT / "idt_comparison.json").read_text(encoding="utf-8"))
fields = json.loads((OUT / "idt_inputs.json").read_text(encoding="utf-8"))["fields"]
meta = {
    "LoRA-S": json.loads((ROOT / "models/idt_adapter/training_metadata.json").read_text(encoding="utf-8")),
    "LoRA-L": json.loads((ROOT / "models/idt_adapter_v2/training_metadata.json").read_text(encoding="utf-8")),
}
P = data["points"]

LATIN = "/Applications/Microsoft Word.app/Contents/Resources/DFonts/Century Gothic.ttf"
CJK_NAMES = ("PingFang.ttc", "STHeiti Medium.ttc", "Hiragino Sans GB.ttc", "Arial Unicode.ttf")
found = {Path(f).name: f for f in fm.findSystemFonts() if Path(f).name in CJK_NAMES}
if not found:
    sys.exit("找不到中文字型（PingFang／Heiti／Hiragino Sans GB／Arial Unicode），圖上的中文會變成方框。")
families = []
for path in (LATIN, next(found[name] for name in CJK_NAMES if name in found)):
    if Path(path).exists():
        fm.fontManager.addfont(path)
        families.append(fm.FontProperties(fname=path).get_name())
plt.rcParams.update({"font.family": families, "axes.unicode_minus": False, "savefig.dpi": 300, "figure.facecolor": "white"})

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style(ax, grid_axis="y"):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=MUTED, labelsize=9.5)


def dot(ax, value, interval, y, color=BLUE):
    ax.plot(interval, [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
    ax.plot([value], [y], marker="o", markersize=7, color=color, markeredgecolor="white", markeredgewidth=1.4, zorder=3)


# ---------------- 圖 1：研究架構 ----------------
fig, ax = plt.subplots(figsize=(6.2, 3.55))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")


def box(x, y, w, h, lines, fill="#f6f7f9", edge="#8a94a3", bold_first=True, size=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill, edgecolor=edge, linewidth=0.9))
    step = 3.5
    top = y + h / 2 + step * (len(lines) - 1) / 2
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - i * step, line, ha="center", va="center", fontsize=size + (0.5 if i == 0 and bold_first else 0),
                color=INK if i == 0 and bold_first else INK2)


def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "-|>", "color": "#5b6573", "linewidth": 1.0, "shrinkA": 0, "shrinkB": 0})


for x, text in ((16, "資料"), (51, "模型與機率評分"), (85, "選擇與評估")):
    ax.text(x, 58.2, text, ha="center", va="center", fontsize=9.2, color=MUTED)
box(2, 42, 28, 12, ["事件通報 3,288 筆", "個人 2,976／系統 312"])
box(2, 24, 28, 12, ["分層、分組切分（7：1：2）", "相同描述不跨子集"])
for i, (name, count) in enumerate((("訓練集", "2,302"), ("驗證集", "329"), ("測試集", "657"))):
    box(2 + i * 9.6, 5, 8.8, 12, [name, count], fill="#eef3fb", edge="#7d9cc9")
arrow(16, 42, 16, 36)
arrow(16, 24, 16, 17)
box(37, 40, 28, 14, ["語言模型微調", "Qwen3.5-9B＋LoRA", "兩種設定：LoRA-S、LoRA-L"], size=7.8)
box(37, 24, 28, 12, ["對照模型", "Zero-shot；TF-IDF＋邏輯迴歸"], size=7.8)
box(37, 5, 28, 12, ["機率評分", "P(系統)＝σ(z系統 − z個人)"], fill="#eef3fb", edge="#7d9cc9", size=7.8)
arrow(51, 40, 51, 36)
arrow(51, 24, 51, 17)
box(72, 40, 26, 14, ["模型與門檻的選擇（驗證集）", "checkpoint：log-loss 最低", "門檻 τ*：macro-F1 最高"], size=7.6)
box(72, 19.5, 26, 16.5, ["測試集評估", "AUROC、AP、log-loss", "macro-F1、precision、recall", "累積增益；bootstrap 95% CI"], size=7.6)
box(72, 3, 26, 12.5, ["資訊來源分析", "輸入欄位比較、標籤一致性", "錯誤重疊"], fill="#fdf1ea", edge="#d99a78", size=7.6)
arrow(85, 40, 85, 36)
arrow(85, 19.5, 85, 15.5)
LINE = {"color": "#5b6573", "linewidth": 1.0, "solid_capstyle": "butt"}
ax.plot([30.4, 33.7, 33.7], [11, 11, 47], **LINE)  # 資料 → 微調、對照模型、機率評分
for y in (47, 30, 11):
    arrow(33.7, y, 37, y)
ax.plot([65, 68.5, 68.5], [11, 11, 47], **LINE)  # 機率評分 → 模型與門檻的選擇
arrow(68.5, 47, 72, 47)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "ms_fig1_framework.png")
plt.close(fig)

# ---------------- 圖 2：AUROC 與 macro-F1 的信賴區間 ----------------
rows = [
    ("Zero-shot", "v1-zero-shot@thr"),
    ("LoRA-S，epoch 1（選定）", "v1-epoch-1@thr"),
    ("LoRA-S，epoch 2", "v1-epoch-2@thr"),
    ("LoRA-S，epoch 3", "v1-epoch-3@thr"),
    ("LoRA-L，epoch 1", "v2-epoch-1@thr"),
    ("LoRA-L，epoch 2（選定）", "v2-epoch-2@thr"),
    ("TF-IDF＋LR", "tfidf@thr"),
]
panels = [("AUROC", "auroc", (0.70, 1.0), [0.70, 0.80, 0.90, 1.00]), ("macro-F1（門檻 τ*）", "macro_f1", (0.55, 0.85), [0.60, 0.70, 0.80])]
fig, axes = plt.subplots(1, 2, figsize=(5.27, 2.95), sharey=True)
ys = list(range(len(rows)))[::-1]
for ax, (title, key, xlim, ticks) in zip(axes, panels, strict=True):
    style(ax, "x")
    reference = P["tfidf@thr"][key]
    ax.axvline(reference, color=BASE, linewidth=1.1, zorder=1)
    for y, (_, point) in zip(ys, rows, strict=True):
        dot(ax, P[point][key], P[point]["ci"][key], y)
        ax.text(xlim[1] + (xlim[1] - xlim[0]) * 0.015, y, f"{P[point][key]:.3f}", va="center", ha="left", fontsize=9, color=INK2)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=9)
    ax.axhline(-0.6, color=BASE, linewidth=0.9)
axes[0].set_yticks(ys)
axes[0].set_yticklabels([r[0] for r in rows], fontsize=8.8, color=INK2)
fig.subplots_adjust(left=0.335, right=0.918, top=0.875, bottom=0.095, wspace=0.30)
fig.savefig(OUT / "ms_fig2_discrimination.png")
plt.close(fig)

# ---------------- 圖 3：驗證損失與訓練損失 ----------------
series = tuple((name, color, f"{name}（每個 epoch {meta[name]['history'][0]['global_step']} 步）") for name, color in (("LoRA-S", BLUE), ("LoRA-L", ORANGE)))
fig, axes = plt.subplots(1, 2, figsize=(5.27, 2.75), sharey=True)
for ax, (field, title) in zip(axes, (("val_loss", "驗證損失（每個 epoch 結束時）"), ("train_loss", "訓練損失（每個 epoch 訓練中的平均）")), strict=True):
    style(ax)
    ax.axhline(0, color=BASE, linewidth=0.9)
    for name, color, label in series:
        steps = [h["global_step"] for h in meta[name]["history"]]
        values = [h[field] for h in meta[name]["history"]]
        ax.plot(steps, values, color=color, linewidth=2, solid_capstyle="round", marker="o", markersize=6.5, markeredgecolor="white", markeredgewidth=1.4, label=label, zorder=3)
        below = name == "LoRA-L"
        ax.annotate(f"{values[-1]:.3f}", (steps[-1], values[-1]), xytext=(0, -9) if below else (7, 0), textcoords="offset points",
                    va="top" if below else "center", ha="center" if below else "left", fontsize=9.5, color=INK2)
    if field == "val_loss":
        ax.text(120, 0.027, f"LoRA-L 共訓練 {meta['LoRA-L']['history'][-1]['global_step']} 步，\n少於 LoRA-S 的第一個 epoch（{meta['LoRA-S']['history'][0]['global_step']} 步）",
                va="center", ha="left", fontsize=8.2, color=MUTED, linespacing=1.35)
    ax.set_xlim(-80, 3080)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_xticklabels(["0", "1,000", "2,000", "3,000"])
    ax.set_ylim(0, 0.15)
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15"])
    ax.set_xlabel("累計更新步數", color=MUTED, fontsize=10)
    ax.set_title(title, loc="left", fontsize=9.6, color=INK, pad=8)
axes[1].legend(loc="upper right", frameon=False, fontsize=8.6, labelcolor=INK2, handlelength=1.6, borderaxespad=0.2)
fig.subplots_adjust(left=0.085, right=0.955, top=0.885, bottom=0.175, wspace=0.09)
fig.savefig(OUT / "ms_fig3_training.png")
plt.close(fig)

# ---------------- 圖 4：累積增益 ----------------
curves = data["gain_curve"]
fig, ax = plt.subplots(figsize=(5.27, 2.9))
style(ax)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.plot([0, 100], [0, 100], color=BASE, linewidth=1.1, zorder=1)
ax.text(71, 64, "隨機排序", fontsize=8.6, color=MUTED, rotation=31, ha="center", va="center")
for key, color, label, width in (("v1-zero-shot", GRAY, "Zero-shot", 1.6), ("tfidf", ORANGE, "TF-IDF＋LR", 2), ("v2-epoch-2", BLUE, "LoRA-L", 2)):
    ax.plot([f * 100 for f, _ in curves[key]], [r * 100 for _, r in curves[key]], color=color, linewidth=width, solid_capstyle="round", label=label, zorder=3)
mark = data["gain"]["v2-epoch-2"]["0.20"]
ax.plot([20], [mark["recall"] * 100], marker="o", markersize=7, color=BLUE, markeredgecolor="white", markeredgewidth=1.4, zorder=4)
ax.annotate(f"前 20% 的通報（{mark['reviewed']} 筆）\n涵蓋 {mark['recall'] * 100:.0f}% 的「系統」事件", (20, mark["recall"] * 100), xytext=(27, 52), textcoords="data",
            fontsize=9, color=INK2, va="center", ha="left", linespacing=1.35, arrowprops={"arrowstyle": "-", "color": BASE, "linewidth": 0.9, "shrinkB": 5})
ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.axhline(0, color=BASE, linewidth=0.9)
ax.set_xlabel("依 P(系統) 由高到低納入的通報比例", color=MUTED, fontsize=10)
ax.set_title("涵蓋的「系統」事件比例", loc="left", fontsize=10.5, color=INK, pad=9)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="lower right", frameon=False, fontsize=8.8, labelcolor=INK2, handlelength=1.6, borderaxespad=0.6)
fig.subplots_adjust(left=0.105, right=0.955, top=0.885, bottom=0.165)
fig.savefig(OUT / "ms_fig4_gain.png")
plt.close(fig)

# ---------------- 圖 5：輸入來源 ----------------
rows5 = [
    ("LoRA-L｜事件描述", P["v2-epoch-2@thr"]["auroc"], P["v2-epoch-2@thr"]["ci"]["auroc"], BLUE),
    ("TF-IDF＋LR｜事件描述", fields["description"]["auroc"], fields["description"]["ci"]["auroc"], BLUE),
    ("TF-IDF＋LR｜調查處理紀錄（去除流程用語）", fields["directive（去掉流程用語）"]["auroc"], fields["directive（去掉流程用語）"]["ci"]["auroc"], ORANGE),
    ("TF-IDF＋LR｜調查處理紀錄", fields["directive"]["auroc"], fields["directive"]["ci"]["auroc"], ORANGE),
    ("TF-IDF＋LR｜事件描述＋調查處理紀錄", fields["description+directive"]["auroc"], fields["description+directive"]["ci"]["auroc"], ORANGE),
]
fig, ax = plt.subplots(figsize=(5.27, 2.6))
style(ax, "x")
ys5 = list(range(len(rows5)))[::-1]
for y, (_, value, interval, color) in zip(ys5, rows5, strict=True):
    dot(ax, value, interval, y, color)
    ax.text(1.004, y, f"{value:.3f}", va="center", ha="left", fontsize=9, color=INK2)
ax.plot([], [], color=BLUE, linewidth=2, marker="o", markersize=6, markeredgecolor="white", label="輸入為事件描述")
ax.plot([], [], color=ORANGE, linewidth=2, marker="o", markersize=6, markeredgecolor="white", label="輸入含調查處理紀錄")
ax.set_xlim(0.86, 1.0)
ax.set_xticks([0.86, 0.90, 0.94, 0.98])
ax.set_xticklabels(["0.86", "0.90", "0.94", "0.98"])
ax.set_ylim(-0.6, len(rows5) - 0.4)
ax.set_yticks(ys5)
ax.set_yticklabels([r[0] for r in rows5], fontsize=8.4, color=INK2)
ax.axhline(-0.6, color=BASE, linewidth=0.9)
ax.set_title("測試集 AUROC（95% 信賴區間）", loc="left", fontsize=10.5, color=INK, pad=9)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False, fontsize=8.4, labelcolor=INK2, handlelength=1.6, columnspacing=1.6)
fig.subplots_adjust(left=0.50, right=0.915, top=0.875, bottom=0.215)
fig.savefig(OUT / "ms_fig5_inputs.png")
plt.close(fig)

for name in ("ms_fig1_framework", "ms_fig2_discrimination", "ms_fig3_training", "ms_fig4_gain", "ms_fig5_inputs"):
    print(OUT / f"{name}.png")
