"""產生研究流程圖 docs/research_flow.svg（2026-11-22 版）；加 --png 另用 Google Chrome 轉出同名的 PNG。

版面與配色沿用 2026-10-18 的流程圖；①–⑨ 的編號不變，新增 ⑩⑪（步驟 05）與 TF-IDF 基準，
這一輪重跑的步驟標「本輪」。流程有變動時直接改這個檔案裡的卡片文字與座標。
文字寬度以 PingFang TC／Menlo 量測（只在 macOS 驗證過），超出卡片的行、互相重疊的卡片會列出來，
並讓程式以非零結束。等寬字（反引號包起來的片段）裡不要放中文，量出來的寬度會不準。

在專案根目錄執行：
    python -m analysis.research_flow                      # → docs/research_flow.svg
    python -m analysis.research_flow --png                # 另外產生 docs/research_flow.png
    python -m analysis.research_flow <輸出 svg 路徑> --png
"""

import glob
import html
import subprocess
import sys
from pathlib import Path

from PIL import ImageFont

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(_args[0]) if _args else Path.cwd() / "docs/research_flow.svg"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
_pingfang = sorted(
    glob.glob("/System/Library/AssetsV2/com_apple_MobileAsset_Font8/*/AssetData/PingFang.ttc")
    + glob.glob("/System/Library/Fonts/PingFang.ttc")
)
MENLO = "/System/Library/Fonts/Menlo.ttc"
if not _pingfang or not Path(MENLO).is_file():
    sys.exit("找不到 PingFang.ttc 或 Menlo.ttc，無法量文字寬度（這支腳本只在 macOS 驗證過）。")
PINGFANG = _pingfang[0]
FONT = "Microsoft JhengHei, Noto Sans TC, PingFang TC, sans-serif"
MONO = "Consolas, Menlo, Courier New, monospace"

W = 2000
INK, SUB, ARROW, SLATE = "#1f2937", "#64748b", "#334155", "#475569"
KIND = {  # 卡片類型 → (標題底色, 邊框色)
    "prog": ("#e2e8f0", "#475569"),
    "small": ("#e0e7ff", "#4338ca"),
    "lora": ("#ede9fe", "#7c3aed"),
    "judge": ("#fef3c7", "#b45309"),
    "eval": ("#d1fae5", "#047857"),
}
FILE_FILL, FILE_STROKE = "#f1f5f9", "#94a3b8"

_fonts = {}
out: list[str] = []
overflow: list[str] = []
boxes: list[tuple[str, float, float, float, float]] = []
_clip = 0


def font(size, weight=400, mono=False):
    key = (size, weight, mono)
    if key not in _fonts:
        if mono:
            _fonts[key] = ImageFont.truetype(MENLO, size, index=1 if weight >= 600 else 0)
        else:
            _fonts[key] = ImageFont.truetype(PINGFANG, size, index=10 if weight >= 600 else 6 if weight >= 500 else 2)
    return _fonts[key]


def segments(content):
    """以反引號切開：奇數段是等寬字。"""
    return [(part, i % 2 == 1) for i, part in enumerate(content.split("`")) if part]


def tw(text, size, weight=400, mono=False, mono_weight=400):
    if "`" not in text:
        return font(size, weight, mono).getlength(text)
    return sum(
        font(size - 1, mono_weight, True).getlength(part) if is_mono else font(size, weight).getlength(part)
        for part, is_mono in segments(text)
    )


def add(markup):
    out.append(markup)


def text(x, y, content, size=20, weight=400, fill=INK, anchor="start", mono=False, maxw=None, where="", mono_weight=400):
    width = tw(content, size, weight, mono, mono_weight)
    if maxw is not None and width > maxw:
        overflow.append(f"{where}：「{content}」寬 {width:.0f} > {maxw:.0f}")
    if mono and any(ord(ch) > 0x2E7F for ch in content):
        overflow.append(f"{where}：等寬字串含中文「{content}」，量不準")
    if "`" in content:
        body = "".join(
            f'<tspan font-family="{MONO}" font-size="{size - 1}" font-weight="{mono_weight}">{html.escape(part, quote=False)}</tspan>'
            if is_mono
            else html.escape(part, quote=False)
            for part, is_mono in segments(content)
        )
    else:
        body = html.escape(content, quote=False)
    add(
        f'<text x="{x:g}" y="{y:g}" font-family="{MONO if mono else FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{body}</text>'
    )


def arrow(points, dashed=False, head=True):
    d = "M" + " L".join(f"{x:g},{y:g}" for x, y in points)
    dash = ' stroke-dasharray="10 7"' if dashed else ""
    end = ' marker-end="url(#ah)"' if head else ""
    add(f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="2.5" stroke-linejoin="round"{dash}{end}/>')


def pill(right, cy, label, color, filled=False, mono=True):
    """靠右對齊的膠囊標籤；回傳它的左緣。"""
    width = tw(label, 17, 700, mono) + 28
    x = right - width
    add(
        f'<rect x="{x:g}" y="{cy - 14:g}" width="{width:g}" height="28" rx="14" '
        f'fill="{color if filled else "#ffffff"}" stroke="{color}" stroke-width="1.5"/>'
    )
    text(x + width / 2, cy + 6, label, 17, 700, "#ffffff" if filled else color, "middle", mono)
    return x


def card(x, y, w, kind, title, lines, num=None, module=None, script=None, rerun=False, chip=None, name=None):
    """流程卡片；回傳高度。chip＝(檔名, 說明) 是卡片底部的輸出檔。"""
    global _clip
    tint, stroke = KIND[kind]
    n = len(lines)
    h = 56 + 34 + (n - 1) * 29 + (24 + 40 + 14 if chip else 24)
    _clip += 1
    add(f'<clipPath id="cp{_clip}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14"/></clipPath>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#ffffff"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="56" fill="{tint}" clip-path="url(#cp{_clip})"/>')
    add(f'<line x1="{x}" y1="{y + 56}" x2="{x + w}" y2="{y + 56}" stroke="{stroke}" stroke-width="1.5"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" stroke="{stroke}" stroke-width="2.5"/>')
    tx = x + 20
    if num:
        add(f'<circle cx="{x + 34}" cy="{y + 28}" r="18" fill="{stroke}"/>')
        text(x + 34, y + 35, num, 20, 700, "#ffffff", "middle")
        tx = x + 64
    text(tx, y + 37, title, 26, 700)
    left = x + w - 14
    if rerun:
        left = pill(left, y + 28, "本輪", INK, filled=True, mono=False) - 8
    if script:
        left = pill(left, y + 28, script, stroke) - 8
    used = tx + tw(title, 26, 700)
    if module:
        mx = used + 22
        text(mx, y + 36, module, 17, 400, SUB, mono=True)
        used = mx + tw(module, 17, 400, True)
    if used > left - 6:
        overflow.append(f"{title}：標題列寬 {used:.0f} > {left - 6:.0f}")
    for i, line in enumerate(lines):
        text(x + 22, y + 56 + 34 + i * 29, line, 20, maxw=w - 44, where=title)
    if chip:
        cy = y + 56 + 34 + (n - 1) * 29 + 24
        add(
            f'<rect x="{x + 16}" y="{cy}" width="{w - 32}" height="40" rx="8" fill="{FILE_FILL}" '
            f'stroke="{FILE_STROKE}" stroke-width="1.5"/>'
        )
        text(x + 28, cy + 27, chip, 19, maxw=w - 56, where=f"{title} 輸出檔", mono_weight=700)
    boxes.append((name or title, x, y, w, h))
    return h


def file_card(x, y, w, name, line1, line2):
    h = 105
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{FILE_FILL}" stroke="{FILE_STROKE}" stroke-width="2"/>')
    text(x + 20, y + 33, name, 20, 700, mono=True)
    text(x + 20, y + 62, line1, 20, maxw=w - 40, where=name)
    text(x + 20, y + 89, line2, 19, 400, SUB, maxw=w - 40, where=name)
    boxes.append((name, x, y, w, h))
    return h


def label(x, y, content, anchor="start", size=19):
    """箭頭旁的說明文字，墊一塊白底避免壓到線。"""
    width = tw(content, size)
    left = x if anchor == "start" else x - width if anchor == "end" else x - width / 2
    add(f'<rect x="{left - 4:g}" y="{y - size:g}" width="{width + 8:g}" height="{size + 8:g}" fill="#ffffff"/>')
    text(x, y, content, size, 400, SLATE, anchor)
    boxes.append((f"label:{content}", left - 4, y - size, width + 8, size + 8))


# ============================== 標題區 ==============================
text(60, 78, "醫療事件 NLP 研究流程", 48, 700)
text(60, 122, "IDT 分類：人工標籤 → LoRA 微調 → 機率評分 → 驗證集挑 checkpoint 與門檻 → 測試集評估", 26, 500, maxw=1140, where="副標 1")
text(60, 158, "撰寫者情緒：模型標註 → 模型評審（沒有人工真值，不訓練、不計 accuracy）", 26, 500, maxw=1140, where="副標 2")
text(60, 196, "全程只有 description 進入模型；人工 IDT 標籤 y 一路保留；驗證與測試集不增生。", 21, 400, SUB, maxw=1140, where="副標 3")

add('<rect x="1230" y="36" width="710" height="190" rx="12" fill="#ffffff" stroke="#94a3b8" stroke-width="1.5"/>')
text(1248, 68, "執行方式（`python -m scripts.<名稱>`）", 20, 700, mono_weight=700)
for i, (script, steps, what) in enumerate(
    [
        ("01_prepare_data", "①②", "清洗、切分"),
        ("02_build_features", "③④", "增生、情緒標註"),
        ("03_train", "⑤", "IDT LoRA 訓練"),
        ("04_inference", "⑥–⑨", "推論、評估、評審"),
        ("05_score_idt", "⑩⑪", "機率評分、挑 checkpoint、調門檻"),
    ]
):
    y = 98 + i * 27
    text(1248, y, script, 18, 700, mono=True)
    text(1456, y, steps, 18)
    text(1524, y, what, 18, maxw=1940 - 16 - 1524, where="執行方式")

# ============================== 圖例 ==============================
lx, ly = 60, 255
for kind, name in (
    ("prog", "程式處理（不用模型）"),
    ("small", "Qwen3.5-2B 推論"),
    ("lora", "Qwen3.5-9B ＋ LoRA（訓練／推論／評分）"),
    ("judge", "Qwen3.8-27B 評審"),
    ("eval", "評估指標"),
):
    tint, stroke = KIND[kind]
    add(f'<rect x="{lx}" y="{ly - 13}" width="26" height="26" rx="6" fill="{tint}" stroke="{stroke}" stroke-width="2"/>')
    text(lx + 38, ly + 8, name, 19)
    lx += 38 + tw(name, 19) + 34
add(f'<rect x="{lx}" y="{ly - 13}" width="26" height="26" rx="6" fill="{FILE_FILL}" stroke="{FILE_STROKE}" stroke-width="2"/>')
text(lx + 38, ly + 8, "資料檔（JSON）", 19)
lx += 38 + tw("資料檔（JSON）", 19)
if lx > 1940:
    overflow.append(f"圖例第 1 列寬到 {lx:.0f}")
lx, ly = 60, 294
arrow([(lx, ly), (lx + 50, ly)])
text(lx + 62, ly + 8, "資料流", 19)
lx += 62 + tw("資料流", 19) + 34
arrow([(lx, ly), (lx + 50, ly)], dashed=True)
text(lx + 62, ly + 8, "只觀察、不更新權重", 19)
lx += 62 + tw("只觀察、不更新權重", 19) + 34
pw = tw("本輪", 17, 700) + 28
pill(lx + pw, ly, "本輪", INK, filled=True, mono=False)
text(lx + pw + 12, ly + 8, "2026-11-22 這一輪重跑的步驟；沒有標示的步驟沿用 2026-10-18 的結果", 19)

# ============================== ①② 與資料檔 ==============================
y1 = 334
h_src = card(
    60, y1, 600, "prog", "資料來源",
    [
        "`data/raw_data/data.csv`：3,288 列",
        "`idt_target`：人工 IDT 標籤 y（個人／系統）",
        "`description`：事件描述，唯一送進模型的文字",
        "`directive`：批示，保留在資料、不送模型",
    ],
    script="input", chip="`data.csv`　只讀，不修改",
)
h1 = card(
    780, y1, 1160, "prog", "讀取與清洗",
    [
        "NFKC 正規化 IDT 標籤並去空白",
        "依序剔除：標籤空白、不是「個人」或「系統」、描述為空、標籤＋描述＋批示完全重複",
        "每筆補上來源識別碼 `source_id`（檔名:Excel 列號），用於追蹤與切分稽核",
        "目前資料：3,288 列 → 3,288 筆（四種剔除原因皆為 0）",
    ],
    num="①", module="preprocessing.py", script="01", chip="`cleaned_data.json`　3,288 筆（個人 2,976／系統 312）",
)
assert h_src == h1
arrow([(660, y1 + h1 / 2), (780, y1 + h1 / 2)])

y2 = y1 + h1 + 62
h2 = card(
    480, y2, 1040, "prog", "依比例切分 7:1:2",
    [
        "依人工 IDT 標籤分層，讓稀少的「系統」在三個集合的比例相近",
        "描述文字相同的紀錄視為同一組、不拆開，避免同一段文字同時出現在訓練與測試",
        "隨機切分（不依時間先後）；`--split-seed 42`，與訓練的 `--seed` 分開",
        "先切分、再增生：改寫版本一定與其原文同在訓練集",
    ],
    num="②", module="split.py", script="01",
)
arrow([(1000, y1 + h1), (1000, y2)])
label(1014, y1 + h1 + 38, "清洗後 3,288 筆")

y3 = y2 + h2 + 66
LX, LW, CX, CW, RX, RW = 60, 550, 660, 680, 1390, 550
file_card(LX, y3, LW, "train_data.json", "訓練集 70%：2,302 筆（個人 2,083／系統 219）", "⑤ 與 TF-IDF 基準用原文；③④ 只做在這個集合")
file_card(CX, y3, CW, "val_data.json", "驗證集 10%：329 筆（個人 298／系統 31）", "不增生、不標註；用來挑 checkpoint 與門檻")
file_card(RX, y3, RW, "test_data.json", "測試集 20%：657 筆（個人 595／系統 62）", "不增生、不標註；只用於最後的推論與評估")
bend = y2 + h2 + 30
arrow([(620, y2 + h2), (620, bend), (LX + LW / 2, bend), (LX + LW / 2, y3)])
arrow([(1000, y2 + h2), (1000, y3)])
arrow([(1380, y2 + h2), (1380, bend), (RX + RW / 2, bend), (RX + RW / 2, y3)])
label(LX + LW / 2 + 14, bend - 9, "70%")
label(1014, y2 + h2 + 40, "10%")
label(RX + RW / 2 - 14, bend - 9, "20%", "end")

# ============================== 三條直欄 ==============================
yf = y3 + 105  # 資料檔的下緣
yl = yf + 84   # 直欄卡片的上緣

# ---- 中欄：IDT 主線（03 → 05）----
h5 = card(
    CX, yl, CW, "lora", "IDT LoRA 訓練（v2）",
    [
        "訓練資料：原始訓練集 2,302 筆（不用改寫）",
        "LoRA r=16、α=32，掛在文字模型的全部線性層",
        "200 個模組、4,011 萬個參數（v1：16 個、111 萬）",
        "提示＝完整 IDT 決策樹＋description，只輸出標籤",
        "2 epochs；梯度累積 8；學習率 2e-4；超長不截斷",
        "每個 epoch 結束：算驗證損失、存一份 checkpoint",
    ],
    num="⑤", module="training.py", script="03", rerun=True,
    chip="`models/idt_adapter_v2/`　epoch-1、epoch-2 兩份 adapter",
)
y10 = yl + h5 + 58
h10 = card(
    CX, y10, CW, "lora", "IDT 機率評分",
    [
        "不生成文字：讀回答第一個 token 的 logit",
        "P(系統)＝「系統」在「個人」「系統」兩個標籤中的機率佔比",
        "基礎模型只載入一次，依序掛上每個 epoch 的 adapter",
        "驗證集 329 筆、測試集 657 筆逐筆評分；可中斷續跑",
    ],
    num="⑩", module="scoring.py", script="05", rerun=True,
    chip="`scores_val.json`、`scores_test.json`　每筆的 P(系統)",
)
y11 = y10 + h10 + 58
h11 = card(
    CX, y11, CW, "eval", "評估與比較",
    [
        "驗證集：挑 log-loss 最低的 checkpoint",
        "驗證集：找 macro-F1 最高的門檻 thr*",
        "測試集：AUROC、AP；門檻 0.5 與 thr* 的混淆矩陣",
        "bootstrap 2,000 次：95% 信賴區間",
        "各設定減 TF-IDF 的配對差距；區間不含 0 才算有差",
    ],
    num="⑪", module="evaluation.py", script="05", rerun=True,
    chip="`idt_score_evaluation.json`　在 `results/idt_scores_v2/`",
)
arrow([(1000, yl + h5), (1000, y10)])
label(1014, yl + h5 + 36, "每個 epoch 的 adapter")
arrow([(1000, y10 + h10), (1000, y11)])
label(1014, y10 + h10 + 36, "每筆的 P(系統) 與人工標籤")

# ---- 資料檔 → 中欄 ----
arrow([(1000, yf), (1000, yl)], dashed=True)
label(1014, yf + 50, "每個 epoch 結束計算驗證損失")
tb = yf + 30
arrow([(LX + LW - 60, yf), (LX + LW - 60, tb), (CX + 110, tb), (CX + 110, yl)])
label(CX + 124, yf + 50, "原文＋人工 y", size=19)
bus = (CX + CW + RX) / 2  # 中欄與右欄之間的空隙
arrow([(CX + CW - 40, yf), (CX + CW - 40, tb), (bus, tb)], head=False)
arrow([(RX + 40, yf), (RX + 40, tb), (bus, tb)], head=False)
arrow([(bus, tb), (bus, y10 + h10 / 2), (CX + CW, y10 + h10 / 2)])
add(f'<circle cx="{bus:g}" cy="{tb:g}" r="5" fill="{ARROW}"/>')

# ---- 左欄：02 特徵工程（情緒支線）與 TF-IDF 基準 ----
h3 = card(
    LX, yl, LW, "small", "訓練集增生",
    [
        "Qwen3.5-2B，溫度 0.8；每筆 2 個改寫",
        "檢查：JSON 合法、非空、不與既有描述重複",
        "不合格就重試，每個改寫最多嘗試 4 次",
        "改寫沿用原文的人工標籤 y",
    ],
    num="③", module="augmentation.py", script="02",
    chip="`train_augmented.json`　6,906 筆（原文＋改寫）",
)
y4 = yl + h3 + 46
h4 = card(
    LX, y4, LW, "small", "撰寫者情緒標註",
    [
        "Qwen3.5-2B（不套 adapter），溫度 0",
        "十種情緒擇一；平鋪直敘標「中性」",
        "輸出無效就重試；3 次仍無效就中止",
        "emotion_target 是模型標註，不是人工真值",
    ],
    num="④", module="emotion.py", script="02",
    chip="`train_features.json`　6,906 筆（含情緒標註）",
)
arrow([(LX + LW / 2, yf), (LX + LW / 2, yl)])
label(LX + LW / 2 - 14, yf + 50, "只有訓練集", "end")
arrow([(LX + LW / 2, yl + h3), (LX + LW / 2, y4)])
text(LX + 4, y4 + h4 + 30, "v1（2026-10-18）的 ⑤ 以這 6,906 筆訓練；", 18, 400, SUB)
text(LX + 4, y4 + h4 + 55, "本輪的 v2 改用原始訓練集，不使用改寫。", 18, 400, SUB)

h_tf = 56 + 34 + 2 * 29 + 24
ytf = y11 + h11 / 2 - h_tf / 2
card(
    LX, ytf, LW, "prog", "TF-IDF 基準",
    [
        "字元 1–3 gram TF-IDF ＋ 邏輯迴歸",
        "C=10、類別平衡權重；只讀 description",
        "以原始訓練集 2,302 筆訓練，同樣輸出 P(系統)",
    ],
    module="baseline.py", script="05", rerun=True,
)
gy = ytf - 42 - 40
add(
    f'<rect x="{LX + 70}" y="{gy}" width="{LW - 140}" height="40" rx="8" fill="{FILE_FILL}" '
    f'stroke="{FILE_STROKE}" stroke-width="1.5" stroke-dasharray="6 5"/>'
)
text(LX + LW / 2, gy + 27, "`train_data.json`（同上，原文 2,302 筆）", 19, anchor="middle", mono_weight=700)
boxes.append(("ghost train", LX + 70, gy, LW - 140, 40))
arrow([(LX + LW / 2, gy + 40), (LX + LW / 2, ytf)])
arrow([(LX + LW, y11 + h11 / 2), (CX, y11 + h11 / 2)])

# ---- 右欄：04 推論與評估 ----
h6 = card(
    RX, yl, RW, "lora", "IDT 推論",
    [
        "Qwen3.5-9B ＋ models/idt_adapter（v1）",
        "貪婪解碼，只接受「個人」或「系統」",
        "等同固定用門檻 0.5；無效就重試，不猜標籤",
    ],
    num="⑥", module="inference.py", script="04",
    chip="`idt_pred`　每筆的 IDT 預測",
)
y7 = yl + h6 + 44
h7 = card(
    RX, y7, RW, "small", "情緒推論",
    [
        "Qwen3.5-2B（不套 adapter），溫度 0",
        "提示與 ④ 相同；每筆新增 emotion_pred",
        "預測先存檔，之後評審失敗也能重試",
    ],
    num="⑦", module="inference.py", script="04",
    chip="`inference_predictions.json`　逐筆預測",
)
y8 = y7 + h7 + 44
h8 = card(
    RX, y8, RW, "eval", "IDT 評估",
    [
        "比對 idt_pred 與人工 idt_target",
        "accuracy、macro-F1、混淆矩陣",
        "各類別 precision／recall／F1",
    ],
    num="⑧", module="evaluation.py", script="04",
)
y9 = y8 + h8 + 44
h9 = card(
    RX, y9, RW, "judge", "情緒評審",
    [
        "Qwen3.8-27B，溫度 0",
        "只看 description ＋ emotion_pred",
        "三面向各 1–5 分；總分＝平均，另附理由",
        "是品質分數，不是 accuracy",
    ],
    num="⑨", module="evaluation.py", script="04",
    chip="`inference_evaluation.json`　指標與評審彙整",
)
rc = RX + RW / 2
arrow([(rc, yf), (rc, yl)])
label(rc + 14, yf + 50, "測試集 657 筆")
for a, b in ((yl + h6, y7), (y7 + h7, y8), (y8 + h8, y9)):
    arrow([(rc, a), (rc, b)])
label(bus + 14, yf + 58, "驗證集＋測試集 → ⑩")
text(RX + 4, y9 + h9 + 30, "本輪沒有用 v2 重跑 04；v2 在門檻 0.5 的結果", 18, 400, SUB)
text(RX + 4, y9 + h9 + 55, "由 ⑪ 提供，與 ⑥ 的貪婪解碼等價。", 18, 400, SUB)

# ============================== 設計原則 ==============================
bottom = max(y11 + h11, y9 + h9 + 62, ytf + h_tf)
yp = bottom + 64
ph = 62 + 4 * 32 + 16
add(
    f'<rect x="60" y="{yp}" width="1880" height="{ph}" rx="14" fill="#fffbeb" stroke="#d97706" '
    f'stroke-width="2" stroke-dasharray="8 6"/>'
)
text(84, yp + 40, "設計原則與資料洩漏防護", 22, 700, "#92400e")
cols = [
    [
        "先切分、再增生：改寫版本必與原文同集合",
        "相同描述的紀錄不跨集合",
        "驗證／測試集不增生、不標註情緒",
        "模型只讀 description；批示不送模型",
    ],
    [
        "checkpoint 與門檻只用驗證集決定；測試集只做最後評估",
        "基準與 LoRA 用同一份訓練集、同一組 bootstrap 重抽樣比較",
        "每個呼叫模型的階段逐筆寫進度檔；模型或提示詞雜湊不同就拒絕續跑",
        "資料或切分變動後，須重跑 01 並重新訓練",
    ],
]
for c, items in enumerate(cols):
    bx = 84 + c * 760
    for i, item in enumerate(items):
        by = yp + 78 + i * 32
        text(bx, by, "•", 20, 700, "#92400e")
        text(bx + 24, by, item, 20, maxw=(736 if c == 0 else 1940 - 24 - bx - 24), where="設計原則")

# ============================== 模型載入順序 ==============================
ym = yp + ph + 58
text(60, ym, "模型一次只載入一個，用完釋放（①②⑧⑪ 與 TF-IDF 基準是純計算，不載入語言模型）", 22, 700)
steps = [
    ("small", "③ 增生", "Qwen3.5-2B"),
    ("small", "④ 情緒標註", "Qwen3.5-2B"),
    ("lora", "⑤ LoRA 訓練", "Qwen3.5-9B ＋ LoRA"),
    ("lora", "⑥ IDT 推論", "Qwen3.5-9B ＋ adapter"),
    ("small", "⑦ 情緒推論", "Qwen3.5-2B"),
    ("judge", "⑨ 評審", "Qwen3.8-27B"),
    ("lora", "⑩ 機率評分", "Qwen3.5-9B ＋ adapter"),
]
gap = 24
bw = (1880 - gap * (len(steps) - 1)) / len(steps)
for i, (kind, top, model) in enumerate(steps):
    tint, stroke = KIND[kind]
    bx = 60 + i * (bw + gap)
    add(f'<rect x="{bx:g}" y="{ym + 24}" width="{bw:g}" height="78" rx="12" fill="{tint}" stroke="{stroke}" stroke-width="2.5"/>')
    text(bx + bw / 2, ym + 56, top, 21, 700, anchor="middle", maxw=bw - 16, where="模型列")
    text(bx + bw / 2, ym + 86, model, 18, 400, stroke, "middle", maxw=bw - 12, where="模型列")
    if i < len(steps) - 1:
        arrow([(bx + bw + 2, ym + 63), (bx + bw + gap - 1, ym + 63)])
yn = ym + 24 + 78 + 46
text(60, yn, "數字為 2026-11-22 的資料快照（資料更新後會不同）；內容依 scripts/ 與 utils/ 的實作整理。", 18, 400, SUB)
H = int(yn + 40)

# ============================== 輸出 ==============================
head = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
    'aria-label="醫療事件 NLP 研究流程：資料清洗、7:1:2 切分、訓練集增生與情緒標註、IDT LoRA 訓練（v2）、'
    'IDT 機率評分、以驗證集挑 checkpoint 與門檻並與 TF-IDF 基準比較、測試集推論與情緒評審">'
    '<defs><marker id="ah" viewBox="0 0 12 10" refX="11" refY="5" markerWidth="12" markerHeight="10" '
    f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L12,5 L0,10 z" fill="{ARROW}"/></marker></defs>'
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>'
)
OUT.write_text(head + "".join(o for o in out if o) + "</svg>\n", encoding="utf-8")

# 檢查：卡片與標籤互不重疊、都在版面內
problems = list(overflow)
for i, (na, ax, ay, aw, ah) in enumerate(boxes):
    if ax < 40 or ax + aw > W - 40:
        problems.append(f"{na} 超出左右邊界（{ax:.0f}–{ax + aw:.0f}）")
    for nb, bx, by, bw_, bh in boxes[i + 1 :]:
        if ax < bx + bw_ and bx < ax + aw and ay < by + bh and by < ay + ah:
            problems.append(f"重疊：{na} ↔ {nb}")
print(f"{OUT}  {W}×{H}  卡片與標籤 {len(boxes)} 個")
for p in problems:
    print("  ✗", p)
if problems:
    sys.exit(1)
if "--png" in sys.argv:
    if not Path(CHROME).is_file():
        sys.exit(f"找不到 {CHROME}，無法轉 PNG；可用其他工具把 SVG 轉成 {W}×{H} 的 PNG。")
    png = OUT.with_suffix(".png")
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
         f"--window-size={W},{H}", f"--screenshot={png}", OUT.resolve().as_uri()],
        check=True, capture_output=True,
    )
    print(f"{png}")
