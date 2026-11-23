"""實驗報告的四張圖：fig_ci（測試集 AUROC、macro-F1 的 95% 信賴區間）、fig_training（v1 與 v2 的驗證／訓練損失，
以累計更新步數對齊）、fig_screening（篩檢效益：依 P(系統) 排序複核的累積增益）、fig_fields（同一個 TF-IDF 模型換輸入欄位的 AUROC）。

樣式沿用 2026-10-18 報告的圖：白底、藍（#2a78d6）／橘（#eb6834）、細格線、直接標註端點。
英數字型用 Microsoft Office 內附的 Century Gothic（找不到就用 matplotlib 預設），中文用系統的
PingFang／Heiti 等字型（只在 macOS 驗證過）。

在專案根目錄執行（先跑 python -m analysis.idt_comparison）：
    python -m analysis.report_charts                      # → results/analysis/fig_ci.png、fig_training.png、fig_screening.png、fig_fields.png
    python -m analysis.report_charts <輸出目錄>            # idt_comparison.json 也從這個目錄讀
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path.cwd()
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/analysis"
DATA = OUT / "idt_comparison.json"
if not DATA.is_file():
    sys.exit(f"找不到 {DATA}；請先執行 python -m analysis.idt_comparison。")

LATIN = "/Applications/Microsoft Word.app/Contents/Resources/DFonts/Century Gothic.ttf"
CJK_NAMES = ("PingFang.ttc", "STHeiti Medium.ttc", "Hiragino Sans GB.ttc", "Arial Unicode.ttf")
found = {Path(f).name: f for f in fm.findSystemFonts() if Path(f).name in CJK_NAMES}
if not found:
    sys.exit("找不到中文字型（PingFang／Heiti／Hiragino Sans GB／Arial Unicode），圖上的中文會變成方框。")
CJK = next(found[name] for name in CJK_NAMES if name in found)
families = []
for path in (LATIN, CJK):
    if Path(path).exists():
        fm.fontManager.addfont(path)
        families.append(fm.FontProperties(fname=path).get_name())
plt.rcParams.update(
    {
        "font.family": families,
        "axes.unicode_minus": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style(ax, grid_axis="y"):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=MUTED, labelsize=9.5)


# ---------------- 圖 2：驗證損失與訓練損失，v1 與 v2 用累計更新步數對齊 ----------------
# 不用 epoch 當橫軸：v1 一個 epoch 是 864 步（6,906 筆），v2 是 288 步（2,302 筆），epoch 編號相同不代表訓練量相同。
# 訓練損失與驗證損失分開畫：前者是整個 epoch 訓練過程中的平均（dropout 開啟），後者在 epoch 結束時計算，不能直接相減。
meta = {
    "v1": json.loads((ROOT / "models/idt_adapter/training_metadata.json").read_text(encoding="utf-8")),
    "v2": json.loads((ROOT / "models/idt_adapter_v2/training_metadata.json").read_text(encoding="utf-8")),
}
series = (
    ("v1", BLUE, f"v1（每個 epoch {meta['v1']['history'][0]['global_step']} 步）"),
    ("v2", ORANGE, f"v2（每個 epoch {meta['v2']['history'][0]['global_step']} 步）"),
)
panels = (("val_loss", "驗證損失（每個 epoch 結束時）"), ("train_loss", "訓練損失（每個 epoch 訓練中的平均）"))
fig, axes = plt.subplots(1, 2, figsize=(5.27, 2.75), sharey=True)
for ax, (field, title) in zip(axes, panels, strict=True):
    style(ax)
    ax.axhline(0, color=BASE, linewidth=0.9)
    for run, color, label in series:
        history = meta[run]["history"]
        steps = [h["global_step"] for h in history]
        values = [h[field] for h in history]
        ax.plot(
            steps, values, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o", markersize=6.5, markeredgecolor="white", markeredgewidth=1.4, label=label, zorder=3,
        )
        # 只標最後一點；v2 停得早，數字放在點的下方，不壓到 v1 的線
        below = run == "v2"
        ax.annotate(
            f"{values[-1]:.3f}", (steps[-1], values[-1]), xytext=(0, -9) if below else (7, 0),
            textcoords="offset points", va="top" if below else "center", ha="center" if below else "left",
            fontsize=9.5, color=INK2,
        )
    if field == "val_loss":
        stop, first = meta["v2"]["history"][-1]["global_step"], meta["v1"]["history"][0]["global_step"]
        ax.text(
            120, 0.027, f"v2 只訓練到 {stop} 步，\n比 v1 的第一個 epoch（{first} 步）還少",
            va="center", ha="left", fontsize=8.2, color=MUTED, linespacing=1.35,
        )
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
fig.savefig(OUT / "fig_training.png")
plt.close(fig)

# ---------------- 圖 3：測試集 AUROC 與 macro-F1 的 95% 信賴區間 ----------------
data = json.loads(DATA.read_text(encoding="utf-8"))
P = data["points"]
rows = [
    ("zero-shot（不微調）", "v1-zero-shot@thr", "v1-zero-shot@thr"),
    ("v1 epoch-1（v1 選定）", "v1-epoch-1@thr", "v1-epoch-1@thr"),
    ("v1 epoch-2", "v1-epoch-2@thr", "v1-epoch-2@thr"),
    ("v1 epoch-3（上一輪採用）", "v1-epoch-3@thr", "v1-epoch-3@thr"),
    ("v2 epoch-1", "v2-epoch-1@thr", "v2-epoch-1@thr"),
    ("v2 epoch-2（v2 選定）", "v2-epoch-2@thr", "v2-epoch-2@thr"),
    ("TF-IDF＋邏輯迴歸", "tfidf@0.5", "tfidf@0.5"),
]
panels = [
    ("測試集 AUROC", "auroc", (0.70, 1.0), [0.70, 0.80, 0.90, 1.00], P["tfidf@0.5"]["auroc"], f"TF-IDF {P['tfidf@0.5']['auroc']:.3f}"),
    ("測試集 macro-F1", "macro_f1", (0.55, 0.85), [0.60, 0.70, 0.80], P["tfidf@0.5"]["macro_f1"], f"TF-IDF {P['tfidf@0.5']['macro_f1']:.3f}"),
]
fig, axes = plt.subplots(1, 2, figsize=(5.27, 2.95), sharey=True)
ys = list(range(len(rows)))[::-1]
for ax, (title, key, xlim, ticks, ref, ref_label) in zip(axes, panels, strict=True):
    style(ax, "x")
    ax.axvline(ref, color=BASE, linewidth=1.1, zorder=1)
    ax.text(
        ref, len(rows) - 0.52, ref_label, ha="center", va="center", fontsize=8.2, color=MUTED, zorder=4,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.6},
    )
    for y, (_, point_auroc, point_f1) in zip(ys, rows, strict=True):
        point = P[point_auroc if key == "auroc" else point_f1]
        lo, hi = point["ci"][key]
        ax.plot([lo, hi], [y, y], color=BLUE, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot([point[key]], [y], marker="o", markersize=7, color=BLUE, markeredgecolor="white", markeredgewidth=1.4, zorder=3)
        ax.text(xlim[1] + (xlim[1] - xlim[0]) * 0.015, y, f"{point[key]:.3f}", va="center", ha="left", fontsize=9, color=INK2)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=9)
    ax.axhline(-0.6, color=BASE, linewidth=0.9)
axes[0].set_yticks(ys)
axes[0].set_yticklabels([r[0] for r in rows], fontsize=8.8, color=INK2)
fig.subplots_adjust(left=0.335, right=0.918, top=0.875, bottom=0.095, wspace=0.30)
fig.savefig(OUT / "fig_ci.png")
plt.close(fig)
# ---------------- 圖 4：當成篩檢工具——依 P(系統) 由高到低複核，找到多少「系統」 ----------------
GRAY = "#898781"
curves = data["gain_curve"]
fig, ax = plt.subplots(figsize=(5.27, 2.9))
style(ax)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.plot([0, 100], [0, 100], color=BASE, linewidth=1.1, zorder=1)
ax.text(71, 64, "隨機複核", fontsize=8.6, color=MUTED, rotation=31, ha="center", va="center")
for key, color, label, width in (
    ("v1-zero-shot", GRAY, "zero-shot（不微調）", 1.6),
    ("tfidf", ORANGE, "TF-IDF＋邏輯迴歸", 2),
    ("v2-epoch-2", BLUE, "LoRA v2", 2),
):
    xs = [f * 100 for f, _ in curves[key]]
    ys = [r * 100 for _, r in curves[key]]
    ax.plot(xs, ys, color=color, linewidth=width, solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)
mark = data["gain"]["v2-epoch-2"]["0.20"]
ax.plot([20], [mark["recall"] * 100], marker="o", markersize=7, color=BLUE, markeredgecolor="white", markeredgewidth=1.4, zorder=4)
ax.annotate(
    f"複核前 20% 的通報（{mark['reviewed']} 筆）\n找到 {mark['recall'] * 100:.0f}% 的「系統」",
    (20, mark["recall"] * 100), xytext=(27, 52), textcoords="data", fontsize=9, color=INK2, va="center", ha="left",
    linespacing=1.35, arrowprops={"arrowstyle": "-", "color": BASE, "linewidth": 0.9, "shrinkB": 5},
)
ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.axhline(0, color=BASE, linewidth=0.9)
ax.set_xlabel("依 P(系統) 由高到低複核的通報比例", color=MUTED, fontsize=10)
ax.set_title("找到的「系統」事件比例（測試集 62 筆）", loc="left", fontsize=10.5, color=INK, pad=9)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="lower right", frameon=False, fontsize=8.8, labelcolor=INK2, handlelength=1.6, borderaxespad=0.6)
fig.subplots_adjust(left=0.105, right=0.965, top=0.885, bottom=0.165)
fig.savefig(OUT / "fig_screening.png")
plt.close(fig)

# ---------------- 圖 5：判定資訊在哪個欄位——同一個 TF-IDF 模型換輸入 ----------------
inputs_path = OUT / "idt_inputs.json"
if not inputs_path.is_file():
    sys.exit(f"找不到 {inputs_path}；請先執行 python -m analysis.idt_inputs。")
fields = json.loads(inputs_path.read_text(encoding="utf-8"))["fields"]
rows5 = [
    ("LoRA v2｜description", P["v2-epoch-2@thr"]["auroc"], P["v2-epoch-2@thr"]["ci"]["auroc"], BLUE),
    ("TF-IDF｜description", fields["description"]["auroc"], fields["description"]["ci"]["auroc"], BLUE),
    ("TF-IDF｜directive（去掉流程用語）", fields["directive（去掉流程用語）"]["auroc"], fields["directive（去掉流程用語）"]["ci"]["auroc"], ORANGE),
    ("TF-IDF｜directive", fields["directive"]["auroc"], fields["directive"]["ci"]["auroc"], ORANGE),
    ("TF-IDF｜description＋directive", fields["description+directive"]["auroc"], fields["description+directive"]["ci"]["auroc"], ORANGE),
]
fig, ax = plt.subplots(figsize=(5.27, 2.35))
style(ax, "x")
ys5 = list(range(len(rows5)))[::-1]
for y, (_, value, interval, color) in zip(ys5, rows5, strict=True):
    ax.plot(interval, [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
    ax.plot([value], [y], marker="o", markersize=7, color=color, markeredgecolor="white", markeredgewidth=1.4, zorder=3)
    ax.text(1.004, y, f"{value:.3f}", va="center", ha="left", fontsize=9, color=INK2)
ax.plot([], [], color=BLUE, linewidth=2, marker="o", markersize=6, markeredgecolor="white", label="只讀事件描述")
ax.plot([], [], color=ORANGE, linewidth=2, marker="o", markersize=6, markeredgecolor="white", label="讀到批示")
ax.set_xlim(0.86, 1.0)
ax.set_xticks([0.86, 0.90, 0.94, 0.98])
ax.set_xticklabels(["0.86", "0.90", "0.94", "0.98"])
ax.set_ylim(-0.6, len(rows5) - 0.4)
ax.set_yticks(ys5)
ax.set_yticklabels([r[0] for r in rows5], fontsize=8.8, color=INK2)
ax.axhline(-0.6, color=BASE, linewidth=0.9)
ax.set_title("測試集 AUROC（95% 信賴區間）", loc="left", fontsize=10.5, color=INK, pad=9)
ax.legend(loc="lower left", frameon=False, fontsize=8.6, labelcolor=INK2, handlelength=1.6, borderaxespad=0.5)
fig.subplots_adjust(left=0.44, right=0.905, top=0.86, bottom=0.115)
fig.savefig(OUT / "fig_fields.png")
plt.close(fig)

print("fonts:", families)
for name in ("fig_training.png", "fig_ci.png", "fig_screening.png", "fig_fields.png"):
    print(OUT / name)
