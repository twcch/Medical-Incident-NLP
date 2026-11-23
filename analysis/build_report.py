"""組出第二輪實驗報告 docs/experiment_report_<日期>.docx（IDT 機率評分、微調與可預測性分析）。

報告以「發現」為主軸：結構化摘要與發現一覽 → 研究問題與貢獻 → 資料與方法 → 依研究問題排列的結果 →
討論（意涵、優點、限制、投稿前要補的工作），作為之後撰寫期刊論文的底稿。

沿用上一份報告（--template，預設 docs/2026-10-18 的那一份）的樣式、頁首頁尾與版面，只重寫內文
（word/document.xml）與三張圖。表格與內文的數字由下列檔案代入，不手抄；內文的判斷（例如「差距
不顯著」）以 assert 綁在數字上，數字變了、敘述不再成立時會中止，要回來改文字：

    results/analysis/idt_comparison.json     python -m analysis.idt_comparison
    results/analysis/idt_inputs.json         python -m analysis.idt_inputs
    results/analysis/fig_*.png（四張圖）       python -m analysis.report_charts
    docs/research_flow.png                   python -m analysis.research_flow --png
    models/idt_adapter*/training_metadata.json

在專案根目錄執行：
    python -m analysis.build_report                       # → docs/experiment_report_2026-11-23.docx
    python -m analysis.build_report --date 2026-12-01 --out docs/experiment_report_2026-12-01.docx

這支腳本的內文是 2026-11-23 這一輪的報告；之後的報告可以複製它再改寫各節的文字。
產生後請用 Word 開啟檢查分頁（這裡只保證 XML 合乎格式）。
"""

import argparse
import json
import math
import re
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

ROOT = Path.cwd()
parser = argparse.ArgumentParser(description="組出第二輪實驗報告的 .docx")
parser.add_argument("--date", default="2026-11-23", help="報告日期，寫在副標題與頁首")
parser.add_argument("--template", type=Path, default=ROOT / "docs/2026-10-18/experiment_report_2026-10-18.docx", help="樣式來源：上一份報告")
parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results/analysis")
parser.add_argument("--flow", type=Path, default=ROOT / "docs/research_flow.png")
parser.add_argument("--out", type=Path, default=None, help="預設 docs/experiment_report_<date>.docx")
ARGS = parser.parse_args()
ANALYSIS, FLOW_PNG = ARGS.analysis_dir, ARGS.flow
OUT = ARGS.out or ROOT / f"docs/experiment_report_{ARGS.date}.docx"
for needed in (ARGS.template, ANALYSIS / "idt_comparison.json", ANALYSIS / "idt_inputs.json", *(ANALYSIS / f"fig_{name}.png" for name in ("ci", "training", "screening", "fields")), FLOW_PNG):
    if not needed.is_file():
        raise SystemExit(f"找不到 {needed}；產生方式見這個檔案開頭的說明。")
D = json.loads((ANALYSIS / "idt_comparison.json").read_text(encoding="utf-8"))
IN = json.loads((ANALYSIS / "idt_inputs.json").read_text(encoding="utf-8"))
P, CMP, OV = D["points"], D["comparisons"], D["overlap"]
CFG = {row["key"]: row for row in D["configs"]}
META = {
    "v1": json.loads((ROOT / "models/idt_adapter/training_metadata.json").read_text(encoding="utf-8")),
    "v2": json.loads((ROOT / "models/idt_adapter_v2/training_metadata.json").read_text(encoding="utf-8")),
}
REPORT_DATE = ARGS.date


# ------------------------------ 數字格式 ------------------------------
def minus(text):
    return text.replace("-", "−")


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x):
    return f"{x * 100:.1f}%"


def ci3(pair):
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def sdiff(x):
    return minus(f"{x:+.3f}")


def sci(pair):
    return minus(f"[{pair[0]:+.3f}, {pair[1]:+.3f}]")


def dci(item):
    return f"{sdiff(item['diff'])} {sci(item['ci'])}"


def significant(item):
    return not (item["ci"][0] <= 0 <= item["ci"][1])


def n(x):
    return f"{x:,}"


def pc(x):
    return f"{x * 100:.0f}%"


def pcci(pair):
    return f"[{pair[0] * 100:.0f}%, {pair[1] * 100:.0f}%]"


def dpp(item):
    """比例的差距改用百分點表示。"""
    return minus(f"{item['diff'] * 100:+.1f} 個百分點 [{item['ci'][0] * 100:+.1f}, {item['ci'][1] * 100:+.1f}]")


# ------------------------------ WordprocessingML ------------------------------
body: list[str] = []


def runs(text, size=None, color=None, bold=False):
    """`**粗體**` 標記與換行 → 一串 <w:r>。"""
    out = []
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        props = []
        if bold or i % 2 == 1:
            props.append("<w:b/><w:bCs/>")
        if color:
            props.append(f'<w:color w:val="{color}"/>')
        if size:
            props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
        rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
        for j, line in enumerate(part.split("\n")):
            if j:
                out.append(f"<w:r>{rpr}<w:br/></w:r>")
            if line:
                out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(line)}</w:t></w:r>')
    return "".join(out)


def title(text):
    body.append(
        '<w:p><w:pPr><w:spacing w:before="1200" w:after="120"/></w:pPr>'
        + runs(text, size=44, color="1F3864", bold=True)
        + "</w:p>"
    )


def subtitle(text):
    body.append('<w:p><w:pPr><w:spacing w:after="240"/></w:pPr>' + runs(text, size=20, color="52514E") + "</w:p>")


def h1(text):
    body.append(f'<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr>{runs(text)}</w:p>')


def h2(text):
    body.append(f'<w:p><w:pPr><w:pStyle w:val="2"/></w:pPr>{runs(text)}</w:p>')


def para(text, after=120, keep_next=False):
    keep = "<w:keepNext/>" if keep_next else ""
    body.append(f'<w:p><w:pPr>{keep}<w:spacing w:after="{after}"/></w:pPr>{runs(text)}</w:p>')


def label(text):
    body.append(f'<w:p><w:pPr><w:keepNext/><w:spacing w:after="60"/></w:pPr>{runs(text, bold=True)}</w:p>')


def bullet(text):
    body.append(
        '<w:p><w:pPr><w:pStyle w:val="a4"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>'
        f'<w:spacing w:after="80"/></w:pPr>{runs(text)}</w:p>'
    )


def caption(text):
    body.append(
        '<w:p><w:pPr><w:spacing w:before="60" w:after="240"/><w:jc w:val="center"/></w:pPr>'
        + runs(text, size=18, color="52514E")
        + "</w:p>"
    )


def note(text):
    """表格下方、圖表標題之前的小字說明。"""
    body.append(
        '<w:p><w:pPr><w:keepNext/><w:spacing w:before="60" w:after="0" w:line="260" w:lineRule="auto"/></w:pPr>'
        + runs(text, size=17, color="52514E")
        + "</w:p>"
    )


CELL_MARGIN = (
    '<w:tcMar><w:top w:w="50" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>'
    '<w:bottom w:w="50" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tcMar>'
)
ROW_EX = '<w:tblPrEx><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/></w:tblCellMar></w:tblPrEx>'


def table(headers, rows, widths, aligns=None, bold_rows=()):
    """aligns：每欄 'l'／'c'／'r'；bold_rows：要整列加粗的列索引。"""
    assert sum(widths) == 9000, sum(widths)
    assert all(len(row) == len(headers) for row in rows), [len(r) for r in rows]
    aligns = aligns or ["l"] * len(headers)
    jc = {"l": "", "c": '<w:jc w:val="center"/>', "r": '<w:jc w:val="right"/>'}
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [
        '<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:tblBorders>'
        + "".join(
            f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
            for side in ("top", "left", "bottom", "right", "insideH", "insideV")
        )
        + '</w:tblBorders><w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
        f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>"
    ]
    cells = "".join(
        f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="E8EEF7"/>'
        f'{CELL_MARGIN}<w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:keepNext/>'
        f'<w:spacing w:line="260" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{runs(h, size=18, bold=True)}</w:p></w:tc>'
        for h, w in zip(headers, widths, strict=True)
    )
    out.append(f"<w:tr>{ROW_EX}<w:trPr><w:tblHeader/></w:trPr>{cells}</w:tr>")
    for index, row in enumerate(rows):
        cells = "".join(
            f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{CELL_MARGIN}<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:keepNext/><w:spacing w:line="260" w:lineRule="auto"/>{jc[a]}</w:pPr>'
            f"{runs(str(cell), size=18, bold=index in bold_rows)}</w:p></w:tc>"
            for cell, w, a in zip(row, widths, aligns, strict=True)
        )
        out.append(f"<w:tr>{ROW_EX}<w:trPr><w:cantSplit/></w:trPr>{cells}</w:tr>")
    out.append("</w:tbl>")
    body.append("".join(out))


_pic = 0


def image(rid, path, width_emu):
    global _pic
    _pic += 1
    with Image.open(path) as im:
        w, h = im.size
    cx, cy = width_emu, round(width_emu * h / w)
    body.append(
        '<w:p><w:pPr><w:keepNext/><w:spacing w:before="120"/><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:noProof/></w:rPr>'
        f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{_pic}" name="圖片 {_pic}"/><wp:cNvGraphicFramePr>'
        '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
        '</wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr>'
        f'<pic:cNvPr id="{_pic}" name="圖片 {_pic}"/><pic:cNvPicPr><a:picLocks noChangeAspect="1" noChangeArrowheads="1"/>'
        f'</pic:cNvPicPr></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:srcRect/><a:stretch><a:fillRect/>'
        f'</a:stretch></pic:blipFill><pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/>'
        '</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        "</wp:inline></w:drawing></w:r></w:p>"
    )


# ------------------------------ 取值與核對 ------------------------------
SEL, V1, TF, OLD04 = P["v2-epoch-2@thr"], P["v1-epoch-1@thr"], P["tfidf@0.5"], P["04"]
SEL05 = P["v2-epoch-2@0.5"]
c_sel_04, c_sel05_04 = CMP["v2e2_thr_minus_04"], CMP["v2e2_05_minus_04"]
c_v2_v1, c_v2_tf, c_v1_tf = CMP["v2e2_thr_minus_v1e1_thr"], CMP["v2e2_thr_minus_tfidf_05"], CMP["v1e1_thr_minus_tfidf_05"]
c_v2e1_v1 = CMP["v2e1_thr_minus_v1e1_thr"]
h1v, h2v = META["v1"]["history"], META["v2"]["history"]
S1, S2 = [h["global_step"] for h in h1v], [h["global_step"] for h in h2v]
assert S2[-1] < S1[0] and h1v[1]["val_loss"] > h1v[0]["val_loss"] * 1.2  # v2 停在 v1 第一個 checkpoint 之前；v1 之後才變差
assert abs(h2v[1]["val_loss"] - h1v[0]["val_loss"]) < 0.002 and h2v[1]["train_loss"] < h1v[1]["train_loss"]
OC = D["overconfidence"]
assert D["selected"]["v2"]["config"] == "epoch-2" and D["selected"]["v1"]["config"] == "epoch-1"
assert (OLD04["tp"], OLD04["fp"]) == (28, 18)
assert OV["all3"] == len(D["all3_cases"]) and OV["all3"] - OV["all3_distinct_texts"] == 1
assert not significant(c_v2_v1["macro_f1"]) and not significant(c_v2_v1["auroc"])
assert not significant(c_v2_tf["macro_f1"]) and not significant(c_v2_tf["auroc"])
assert not significant(c_sel_04["macro_f1"]) and significant(c_sel_04["system_recall"]) and significant(c_sel_04["system_precision"])
assert all(not significant(c_sel05_04[k]) for k in ("macro_f1", "system_recall", "system_precision"))
assert significant(c_v2_tf["system_recall"]) and not significant(c_v2_tf["system_precision"])
assert significant(c_v2e1_v1["ap"])
low = [c for c in D["all3_cases"] if c["p_v2e2"] < 0.05]
ceil = {k: v["test_best_macro_f1"] for k, v in D["ceiling"].items()}
lora_keys = ["v1-epoch-1", "v1-epoch-2", "v1-epoch-3", "v2-epoch-1", "v2-epoch-2"]
aurocs = [CFG[k]["test"]["auroc"] for k in lora_keys]
sens = {round(r["threshold"], 3): r["test"] for r in D["sensitivity_v2e2"]}
band = [v["macro_f1"] for t, v in sens.items() if 0.05 <= t <= 0.3]
FI, CV = IN["fields"], IN["cross_validation"]
ND = IN["near_duplicates"]["0.80"]
AUG = IN["augmentation"]
LC = IN["learning_curve"]
assert FI["directive"]["macro_f1"] - FI["description"]["macro_f1"] > 0.05 and CV["directive"]["auroc"] > CV["description"]["auroc"]
trio = IN["same_text"]["conflicting"]
assert len(trio) == 1 and sorted(trio[0]["records"]) == [["data.csv:543", "系統"], ["data.csv:548", "個人"], ["data.csv:549", "個人"]]

ZS = P["v1-zero-shot@thr"]
c_v2_zs, c_v1_zs, c_zs_tf = CMP["v2e2_thr_minus_zs_thr"], CMP["v1e1_thr_minus_zs_thr"], CMP["zs_thr_minus_tfidf_05"]
assert all(significant(c_v2_zs[k]) for k in ("auroc", "macro_f1", "ap")) and significant(c_v1_zs["auroc"]) and significant(c_zs_tf["auroc"])
G, RN = D["gain"], D["review_needed"]
g10, g20, g30, g50 = (G["v2-epoch-2"][k] for k in ("0.10", "0.20", "0.30", "0.50"))
assert g20["ci"][0] > 0.6 and g30["recall"] > 0.9  # 篩檢效益的敘述
FD, FDC = FI["directive"], FI["directive（去掉流程用語）"]
assert all(item["minus_description"][k]["ci"][0] > 0 for item in (FD, FDC) for k in ("auroc", "macro_f1"))
RC = IN["reason_coverage"]
assert SEL["system_precision"] < 0.5 and SEL["system_recall"] > 0.66  # 5.1 節的敘述
d2, l2, l1 = META["v2"]["data"], META["v2"]["lora"], META["v1"]["lora"]
ratio = l2["trainable_params"] / l1["trainable_params"]
dg = D["diagnostics"]
mass = [dg[run][name]["label_mass_mean"] for run, name in (("v1", "epoch-1"), ("v1", "epoch-2"), ("v1", "epoch-3"), ("v2", "epoch-1"), ("v2", "epoch-2"))]
assert all(dg[run][name]["top_is_label_rate"] == 1.0 for run in dg for name in dg[run])
assert all(CFG[k]["ci"]["vs_baseline"]["auroc"][0] <= 0 <= CFG[k]["ci"]["vs_baseline"]["auroc"][1] for k in lora_keys)
assert min(CFG[k]["test"]["log_loss"] for k in ("v1-epoch-1", "v1-epoch-2", "v1-epoch-3")) == CFG["v1-epoch-1"]["test"]["log_loss"]

# ============================== 內容 ==============================
title("醫療事件 NLP 實驗結果報告")
subtitle(f"第二輪：IDT 歸因分類的機率評分、微調與可預測性分析　｜　{REPORT_DATE}")

# ------------------------------ 1. 摘要 ------------------------------
h1("1. 摘要")
para(
    "**目的**　醫療異常事件通報的 IDT（Incident Decision Tree）歸因，也就是判定事件屬於「個人」或「系統」，目前由人工完成。"
    "本研究檢驗：只憑通報當下的事件描述，微調後的大型語言模型能把這項判定做到什麼程度，以及限制表現的因素是什麼。"
)
para(
    "**方法**　3,288 筆通報（「系統」佔 9.5%）依標籤分層切成 7:1:2，描述相同的紀錄不跨集合。以 LoRA 微調 Qwen3.5-9B"
    "（v1、v2 兩種設定），不生成文字、直接讀取「個人」「系統」兩個標籤的機率作為 P(系統)；checkpoint 與決策門檻只由驗證集決定。"
    "對照組是不微調的同一個模型（zero-shot）與字元 n-gram 的 TF-IDF＋邏輯迴歸。測試集 657 筆（「系統」62 筆）以 bootstrap "
    "2,000 次估計 95% 信賴區間與配對差距。"
)
para(
    f"**結果**　微調把測試集 AUROC 由 {f3(ZS['auroc'])} 提高到 {f3(SEL['auroc'])}（差距 {dci(c_v2_zs['auroc'])}）。依模型分數排序，"
    f"複核前 20% 的通報就能找到 {pc(g20['recall'])} 的系統事件（95% 區間 {pcci(g20['ci'])}）；與第一輪的貪婪解碼相比，校準門檻後"
    f"系統事件的偵測率由 {pct(OLD04['system_recall'])} 提高到 {pct(SEL['system_recall'])}（{dpp(c_sel_04['system_recall'])}）。"
    f"可訓練參數相差 {ratio:.0f} 倍的兩種 LoRA 設定與 TF-IDF 基準，AUROC 沒有差別（{f3(V1['auroc'])}、{f3(SEL['auroc'])}、"
    f"{f3(TF['auroc'])}），三者漏掉的也是同一批案例。同一個 TF-IDF 模型改讀調查後的批示，AUROC 升到 {f3(FD['auroc'])}"
    f"（差距 {dci(FD['minus_description']['auroc'])}）。"
)
para(
    "**結論**　只憑通報當下的描述，模型已經能把系統事件有效地排到前面，適合作為優先複核的篩檢工具。各項證據顯示，進一步的提升"
    "主要受限於描述本身的資訊，而不是模型容量；判定所需的資訊多半出現在調查紀錄。後續應把「通報當下的篩檢」與「調查後的輔助判定」當成兩個任務。"
)
label("主要發現")
table(
    ["", "發現", "主要證據（測試集 657 筆，「系統」62 筆）"],
    [
        [
            "1",
            "微調明顯優於直接使用語言模型",
            f"AUROC 由 zero-shot 的 {f3(ZS['auroc'])} 提高到 {f3(SEL['auroc'])}，差距 {dci(c_v2_zs['auroc'])}；macro-F1 差距 "
            f"{dci(c_v2_zs['macro_f1'])}（4.1 節）",
        ],
        [
            "2",
            "可以當成篩檢工具，把系統事件排到前面",
            f"依 P(系統) 排序，複核前 20% 的通報找到 {pc(g20['recall'])} 的系統事件 {pcci(g20['ci'])}，前 30% 找到 {pc(g30['recall'])}；"
            "隨機複核只有 20% 與 30%（4.4 節）",
        ],
        [
            "3",
            "由驗證集決定門檻後，系統事件的偵測率大幅提高",
            f"「系統」recall 由第一輪的 {pct(OLD04['system_recall'])} 提高到 {pct(SEL['system_recall'])}（"
            f"{dpp(c_sel_04['system_recall'])}）；precision 由 {pct(OLD04['system_precision'])} 降為 {pct(SEL['system_precision'])}，"
            "macro-F1 沒有顯著變化（4.4 節）",
        ],
        [
            "4",
            "很小的 adapter、不用增生資料，就能達到同樣的表現",
            f"v1 只訓練 {l1['trainable_params'] / l1['total_params'] * 100:.3f}% 的參數；v2 不用改寫資料、只訓練 {S2[-1]} 個更新步。"
            f"兩者的 AUROC 是 {f3(V1['auroc'])} 與 {f3(SEL['auroc'])}，差距 {dci(c_v2_v1['auroc'])}（4.2 節）",
        ],
        [
            "5",
            "證據一致指向：限制表現的是輸入的資訊，不是模型容量",
            f"可訓練參數相差 {ratio:.0f} 倍的兩種 LoRA 與字元 n-gram 的 TF-IDF 分辨能力相同（v2 減 TF-IDF 的 AUROC "
            f"{dci(c_v2_tf['auroc'])}），三者漏掉同一批 {OV['all3']} 筆系統事件（4.3、4.5 節）",
        ],
        [
            "6",
            "判定所需的資訊多半在調查後的批示",
            f"同一個 TF-IDF 模型改讀批示，AUROC 由 {f3(FI['description']['auroc'])} 升到 {f3(FD['auroc'])}（差距 "
            f"{dci(FD['minus_description']['auroc'])}）；去掉流程用語後差距仍是 {dci(FDC['minus_description']['auroc'])}（4.6 節）",
        ],
        [
            "7",
            "人工標籤一致，上限不是標註雜訊造成的",
            f"描述相似度 0.8 以上的 {ND['pairs']} 對紀錄中，{ND['pairs'] - ND['label_conflicts']} 對標籤相同；唯一一組描述相同而標籤"
            "不同的紀錄，差別在批示記載的調查結果（4.6 節）",
        ],
    ],
    [450, 2850, 5700],
    ["c", "l", "l"],
)
caption("方括號是 95% bootstrap 信賴區間；差距的區間不含 0 表示顯著")

# ------------------------------ 2. 研究問題與貢獻 ------------------------------
h1("2. 研究問題與貢獻")
para(
    f"第一輪（2026-10-18）以貪婪解碼評估，IDT 的 macro-F1 是 {f3(OLD04['macro_f1'])}、「系統」recall 只有 "
    f"{pct(OLD04['system_recall'])}，而且分不出問題出在模型、訓練方式還是資料。本輪把它拆成五個可以用數據回答的問題："
)
bullet("**RQ1**　微調是否比直接使用語言模型（zero-shot）有效？")
bullet("**RQ2**　微調後的語言模型是否優於傳統的詞袋基準（TF-IDF＋邏輯迴歸）？")
bullet("**RQ3**　訓練設定（LoRA 的範圍與大小、資料增生、訓練量）是否影響表現？")
bullet("**RQ4**　把決策門檻交給驗證集決定，能不能提高「系統」的偵測率？當成篩檢工具的效益有多大？")
bullet("**RQ5**　限制表現的是什麼：模型、標籤品質，還是輸入的資訊？")
label("本輪的貢獻")
bullet(
    "**評估方法。**不生成文字、直接讀取標籤機率的評分方式，讓語言模型分類器可以計算 AUROC、AP 與 log-loss，並由驗證集決定 "
    "checkpoint 與門檻；搭配 bootstrap 的配對比較與 TF-IDF 強基準。過程中發現並處理了 bfloat16 下分數離散造成的同分問題（3.3 節）。"
)
bullet("**效益的量化。**微調相對於 zero-shot 的提升、門檻對偵測率的影響，以及當成篩檢工具時的累積增益，都附 95% 信賴區間。")
bullet(
    "**瓶頸的定位。**以三項互相獨立的證據（訓練設定、基準比較、共同的錯誤）說明限制來自輸入資訊，並以換輸入欄位的分析指出資訊所在。"
)
bullet("**可重現性。**資料切分、訓練、評分、評估，以及本報告的每個數字與圖表，都可以由版本庫中的程式重新產生（第 6 節）。")

# ------------------------------ 3. 資料與方法 ------------------------------
h1("3. 資料與方法")
h2("3.1 資料與切分")
para(
    "原始檔 data/raw_data/data.csv 共 3,288 筆，清洗後未剔除任何紀錄；「個人」2,976 筆（90.5%）、「系統」312 筆（9.5%），"
    "類別高度不平衡。依 IDT 標籤分層、以 seed 42 切成 7:1:2，與第一輪完全相同："
)
table(
    ["集合", "筆數", "個人", "系統", "系統佔比"],
    [
        ["train（train_data.json）", "2,302", "2,083", "219", "9.5%"],
        ["val（val_data.json）", "329", "298", "31", "9.4%"],
        ["test（test_data.json）", "657", "595", "62", "9.4%"],
        ["合計", "3,288", "2,976", "312", "9.5%"],
    ],
    [3000, 1500, 1500, 1500, 1500],
    ["l", "r", "r", "r", "r"],
    bold_rows=(3,),
)
caption("表 1　資料切分與類別分布")
para(
    "防止資訊洩漏的做法：描述相同的紀錄視為同一組、不跨集合（文字完全相同的描述有 "
    f"{D['duplicates']['groups']} 組、共 {D['duplicates']['records']} 筆，每一組都落在同一個集合）；先切分再增生；驗證集與測試集"
    "不增生；模型只讀事件描述（description），批示（directive）不送模型。checkpoint 與門檻只用驗證集決定，測試集只用來報告結果。"
)
h2("3.2 模型與訓練")
para(
    "基礎模型是 Qwen/Qwen3.5-9B。提示是完整的 IDT 決策樹（Q1 到 Q10）加上事件描述，模型只輸出「個人」或「系統」，"
    f"loss 只算標籤與結束 token（提示雜湊 {META['v2']['idt_prompt']}，兩版相同）。兩種 LoRA 設定如表 2：v1 是第一輪的設定，"
    "v2 把 LoRA 擴大到文字模型的全部線性層，並只用原始訓練集。"
)
table(
    ["項目", "LoRA v1（第一輪，2026-10-18）", "LoRA v2（本輪，2026-11-22）"],
    [
        ["訓練資料", "6,906 筆（原文 2,302＋改寫 4,604）", "2,302 筆（只有原文）"],
        [
            "LoRA 掛載範圍",
            f"q_proj／v_proj，{l1['matched_modules']} 個模組（32 層中使用標準注意力的 8 層）",
            f"注意力、線性注意力與 MLP 的全部線性層，{l2['matched_modules']} 個模組",
        ],
        ["rank／alpha", "8／16", "16／32"],
        [
            "可訓練參數",
            f"{n(l1['trainable_params'])}（{l1['trainable_params'] / l1['total_params'] * 100:.3f}%）",
            f"{n(l2['trainable_params'])}（{l2['trainable_params'] / l2['total_params'] * 100:.2f}%）",
        ],
        ["epochs／更新步", f"3／{n(h1v[-1]['global_step'])}", f"2／{h2v[-1]['global_step']}"],
        [
            "訓練時間",
            f"{META['v1']['duration_minutes'] / 60:.1f} 小時",
            f"{META['v2']['duration_minutes'] / 60:.1f} 小時（兩次的硬體不同，不能直接比較）",
        ],
        ["其餘設定", "相同：完整決策樹提示、dropout 0.05、lr 0.0002、梯度累積 8、warmup 0.03、max_length 4,096、seed 42", "同左"],
    ],
    [2000, 3500, 3500],
)
caption("表 2　兩種 LoRA 設定的對照（models/idt_adapter、models/idt_adapter_v2 的 training_metadata.json）")
para(
    "對照組有兩個。zero-shot：同一個基礎模型不掛 adapter，使用同一份提示。TF-IDF 基準：字元 1–3 gram 的 TF-IDF"
    "（min_df 2、sublinear tf）加邏輯迴歸（C=10、類別平衡權重），以原始訓練集 2,302 筆訓練，同樣只讀事件描述。"
    "v2 在 NVIDIA GB10（cuda、bfloat16）上訓練，評分在 Apple Silicon（mps、bfloat16）上執行。"
)
image("rId7", FLOW_PNG, 5699846)
caption("圖 1　研究流程（docs/research_flow.png；①–⑪ 為步驟編號，標示「本輪」的是這一輪重跑的步驟）")
h2("3.3 機率評分與挑選規則")
bullet(
    "**P(系統)。**第一輪的貪婪解碼等於固定在 P(系統) 超過 0.5 才判「系統」，看不出模型有多確定，也不能比較 checkpoint 或調整門檻。"
    "本輪不生成文字，用與訓練相同的提示，讀回答位置上「個人」「系統」兩個標籤第一個 token 的機率，在兩者之間正規化成 P(系統)。"
    f"兩個標籤的機率合計，zero-shot 平均 {dg['v1']['zero-shot']['label_mass_mean']:.3f}、微調後的五個 checkpoint 在 "
    f"{min(mass):.4f}–{max(mass):.4f} 之間，機率最高的 token 也全部是標籤，所以這個分數可以代表模型的判斷。"
)
bullet(
    "**挑選規則（事先定下）。**在訓練出的 checkpoint 中挑驗證集 log-loss 最低的；每個設定在驗證集找 macro-F1 最高的門檻"
    "（以下記為 thr*）。zero-shot 與 TF-IDF 是基準，不參與挑選。"
)
bullet(
    "**同分的處理。**bfloat16 下模型的 logit 是離散的，同一個 logit 差算出的 P(系統) 只差浮點誤差。評估時把 logit 相差不到 "
    f"{D['tie_tolerance_logit']:g} 的分數視為同分，門檻只會落在真正不同的分數之間，AUROC 與 AP 也把它們當成同分。"
)
bullet(
    "**指標。**排序能力與機率品質用 AUROC、AP（average precision）與 log-loss；分類結果用 accuracy、macro-F1 與「系統」的 "
    "precision／recall／F1，分別在門檻 0.5（等同貪婪解碼）與 thr* 計算；篩檢效益用依 P(系統) 由高到低複核時找到的「系統」比例。"
)
h2("3.4 統計方法")
bullet(
    "測試集 bootstrap 2,000 次（seed 42，所有設定共用同一組重抽樣），報告 95% 百分位信賴區間。兩個設定的比較用配對差距；"
    "區間不含 0 才視為有差別。"
)
bullet(
    "測試集有 62 筆「系統」、驗證集 31 筆。以本次資料，macro-F1 配對差距的區間半寬約 0.04–0.06，AUROC 約 0.01–0.05；"
    "比這更小的差距分不出來。"
)

# ------------------------------ 4. 結果 ------------------------------
h1("4. 結果")
h2("4.1 微調的效果（RQ1）")
para("表 3 列出七個設定在驗證集與測試集的 log-loss、AUROC 與 AP；圖 2 是測試集 AUROC 與 macro-F1 的 95% 信賴區間。")
NAMES = {
    "v1-zero-shot": "zero-shot（不掛 adapter）",
    "v1-epoch-1": "v1 epoch-1（v1 選定）",
    "v1-epoch-2": "v1 epoch-2",
    "v1-epoch-3": "v1 epoch-3（上一輪採用）",
    "v2-epoch-1": "v2 epoch-1",
    "v2-epoch-2": "v2 epoch-2（v2 選定）",
    "tfidf": "TF-IDF＋邏輯迴歸",
}
ORDER = list(NAMES)
table(
    ["設定", "val log-loss", "val AUROC", "val AP", "test log-loss", "test AUROC（95% 區間）", "test AP"],
    [
        [
            NAMES[k],
            f4(CFG[k]["val"]["log_loss"]),
            f3(CFG[k]["val"]["auroc"]),
            f3(CFG[k]["val"]["ap"]),
            f4(CFG[k]["test"]["log_loss"]),
            f"{f3(CFG[k]['test']['auroc'])} {ci3(CFG[k]['ci']['auroc'])}",
            f3(CFG[k]["test"]["ap"]),
        ]
        for k in ORDER
    ],
    [2250, 1000, 950, 800, 1050, 2150, 800],
    ["l", "r", "r", "r", "r", "r", "r"],
    bold_rows=(ORDER.index("v2-epoch-2"),),
)
caption("表 3　各設定在驗證集與測試集的 log-loss、AUROC 與 AP（results/idt_scores、results/idt_scores_v2）")
image("rId8", ANALYSIS / "fig_ci.png", 5619750)
caption(
    "圖 2　測試集 AUROC 與 macro-F1 的 95% 信賴區間（bootstrap 2,000 次）。macro-F1 用各設定在驗證集調出的門檻；"
    "TF-IDF 用門檻 0.5（見表 6）。直線是 TF-IDF 的數值"
)
bullet(
    f"**微調顯著優於 zero-shot。**選定的 v2 與 zero-shot 相比，AUROC 差距 {dci(c_v2_zs['auroc'])}、macro-F1 差距 "
    f"{dci(c_v2_zs['macro_f1'])}、AP 差距 {dci(c_v2_zs['ap'])}，區間都不含 0。v1 的結果相同（AUROC 差距 {dci(c_v1_zs['auroc'])}）。"
    f"五個微調後的 checkpoint 測試集 AUROC 都在 {min(aurocs):.3f}–{max(aurocs):.3f}。"
)
bullet(
    f"**只把決策樹寫進提示並不夠。**zero-shot 使用完整的 IDT 決策樹提示，AUROC 只有 {f3(ZS['auroc'])}，還顯著低於 TF-IDF"
    f"（差距 {dci(c_zs_tf['auroc'])}）。這個任務需要用有標籤的資料調整模型。"
)
vl = [CFG[k]["val"]["log_loss"] for k in ("v1-epoch-1", "v1-epoch-2", "v1-epoch-3")]
va = [CFG[k]["val"]["auroc"] for k in ("v1-epoch-1", "v1-epoch-2", "v1-epoch-3")]
bullet(
    f"**以驗證集 log-loss 挑 checkpoint，可以避開過度自信的 epoch。**v1 的三個 epoch，驗證集 AUROC 都在 {min(va):.3f}–{max(va):.3f}，log-loss 卻由 "
    f"{vl[0]:.3f} 惡化到 {vl[2]:.3f}：後面的 epoch 沒有排得更差，而是更過度自信。測試集 62 筆「系統」中，P(系統) 低於 0.01 的有 "
    f"{OC['v1-epoch-1/test']['pos_lt_0.01']}、{OC['v1-epoch-2/test']['pos_lt_0.01']}、{OC['v1-epoch-3/test']['pos_lt_0.01']} 筆"
    "（epoch-1 到 epoch-3）。依規則選出的 v1 epoch-1，也是測試集 log-loss 最低的 checkpoint；第一輪採用的卻是 epoch-3。"
)
bullet(
    f"v2 兩個 checkpoint 的驗證集 log-loss 是 {f4(CFG['v2-epoch-1']['val']['log_loss'])} 與 {f4(CFG['v2-epoch-2']['val']['log_loss'])}，"
    "依規則選 epoch-2；兩者只差 0.001，遠小於抽樣誤差。"
)

h2("4.2 訓練設定與成本（RQ3）")
table(
    ["版本", "epoch", "累計更新步", "train_loss", "val_loss", "val_label_match", "時間（小時）"],
    [
        [run, str(h["epoch"]), n(h["global_step"]), f4(h["train_loss"]), f4(h["val_loss"]), f4(h["val_label_match"]), f"{h['seconds'] / 3600:.2f}"]
        for run, history in (("v1", h1v), ("v2", h2v))
        for h in history
    ],
    [1000, 900, 1400, 1400, 1400, 1700, 1200],
    ["c", "c", "r", "r", "r", "r", "r"],
)
caption("表 4　每個 epoch 的訓練與驗證指標")
image("rId9", ANALYSIS / "fig_training.png", 5619750)
caption(
    "圖 3　驗證損失（左）與訓練損失（右），橫軸是累計更新步數（每步 8 筆）。每個點是一個 epoch 結束："
    f"v1 每個 epoch {S1[0]} 步，v2 每個 epoch {S2[0]} 步"
)
bullet(
    f"**兩種設定的表現相同。**選定的 v2 epoch-2 與 v1 epoch-1 相比，macro-F1 差距 {dci(c_v2_v1['macro_f1'])}、AUROC 差距 "
    f"{dci(c_v2_v1['auroc'])}，都不顯著；兩者給 657 筆的分數排名高度一致（Spearman 相關 "
    f"{D['spearman']['v2-epoch-2|v1-epoch-1']:.2f}）。"
)
bullet(
    f"**小的 adapter 就夠。**v1 只在 {l1['matched_modules']} 個模組上訓練 {n(l1['trainable_params'])} 個參數"
    f"（{l1['trainable_params'] / l1['total_params'] * 100:.3f}%）；v2 擴大到 {l2['matched_modules']} 個模組、{n(l2['trainable_params'])} "
    f"個參數（約 {ratio:.0f} 倍），沒有帶來額外的提升。"
)
bullet(
    f"**不需要資料增生，訓練量也可以更少。**v2 只用原始訓練集 {n(d2['train_records'])} 筆、共 {S2[-1]} 個更新步"
    f"（{n(d2['train_records'] * len(h2v))} 筆次）；v1 的 epoch-1 用含改寫的 {n(META['v1']['data']['train_records'])} 筆、{S1[0]} 步。"
    f"以 TF-IDF 估計，訓練資料加入改寫後 macro-F1 也沒有提高（{f3(AUG['originals']['macro_f1'])} 對 "
    f"{f3(AUG['originals+paraphrases']['macro_f1'])}）。之後的實驗可以省去增生這一步。"
)
bullet(
    f"驗證損失：v2 的兩個 checkpoint（{S2[0]}、{S2[1]} 步）與 v1 的第一個 checkpoint（{S1[0]} 步）都在 0.074 左右；v1 到 "
    f"{n(S1[1])} 步、{n(S1[2])} 步才上升到 {f3(h1v[1]['val_loss'])}、{f3(h1v[2]['val_loss'])}。v2 沒有訓練到 v1 開始變差的範圍，"
    "所以只能說 v2 停在驗證損失還沒變差的地方，不能說 v2 的設定解決了過擬合。圖 3 的訓練損失是每個 epoch 訓練過程中的平均"
    "（dropout 開啟），不能與驗證損失直接相減。"
)
bullet(
    "v1 只匹配到 16 個模組的原因：Qwen3.5-9B 的 32 層中，只有 8 層使用名為 q_proj／v_proj 的注意力投影，其餘 24 層是線性注意力，"
    "模組名稱不同。v2 的 200 個模組＝24 層 × 6 個＋8 層 × 7 個。"
)

h2("4.3 與 TF-IDF 基準的比較（RQ2）")
def cmp_cell(item):
    if item is None:
        return "—"
    text = f"{sdiff(item['diff'])}\n{sci(item['ci'])}"
    return f"**{text}**" if significant(item) else text


def cmp_row(name, key):
    c = CMP[key]
    return [name, cmp_cell(c["macro_f1"]), cmp_cell(c["system_recall"]), cmp_cell(c["system_precision"]), cmp_cell(c["auroc"])]

table(
    ["比較（前者 − 後者）", "macro-F1", "系統 recall", "系統 precision", "AUROC"],
    [
        cmp_row("LoRA v2 − zero-shot", "v2e2_thr_minus_zs_thr"),
        cmp_row("LoRA v2 − LoRA v1", "v2e2_thr_minus_v1e1_thr"),
        cmp_row("LoRA v2 − TF-IDF", "v2e2_thr_minus_tfidf_05"),
        cmp_row("LoRA v1 − TF-IDF", "v1e1_thr_minus_tfidf_05"),
        cmp_row("zero-shot − TF-IDF", "zs_thr_minus_tfidf_05"),
        cmp_row("LoRA v2（thr*）− 第一輪 04 的輸出", "v2e2_thr_minus_04"),
        cmp_row("LoRA v2（0.5）− 第一輪 04 的輸出", "v2e2_05_minus_04"),
    ],
    [2600, 1600, 1600, 1600, 1600],
    ["l", "c", "c", "c", "c"],
)
note(
    "每格上方是差距、下方是 95% 配對 bootstrap 區間；粗體表示區間不含 0。LoRA 與 zero-shot 用驗證集調出的門檻（thr*），"
    "TF-IDF 用門檻 0.5（它在測試集較好的操作點，見表 6）。04 只有判定結果、沒有機率，與它比較的 AUROC 改用 05 對同一個 "
    "adapter（v1 epoch-3）的評分計算。"
)
caption("表 5　各設定之間的配對比較（測試集）")
bullet(
    f"**微調後的語言模型與 TF-IDF 的分辨能力相同。**v2 減 TF-IDF 的 AUROC 差距 {dci(c_v2_tf['auroc'])}、macro-F1 差距 "
    f"{dci(c_v2_tf['macro_f1'])}；v1 的結果相同。區間都包含 0。"
)
bullet(
    f"**三項證據都指向同一件事：瓶頸不在模型。**（一）可訓練參數相差 {ratio:.0f} 倍的兩種 LoRA 表現相同（4.2 節）；"
    "（二）9B 的語言模型與字元 n-gram 的線性模型表現相同；（三）三個模型漏掉的是同一批案例（4.5 節）。"
    "如果限制來自模型容量或訓練方式，這三者不會落在同一個水準。"
)
bullet(
    f"LoRA 的「系統」recall 顯著高於 TF-IDF（v2 {sdiff(c_v2_tf['system_recall']['diff'])}、v1 "
    f"{sdiff(c_v1_tf['system_recall']['diff'])}），是因為 LoRA 在較低的門檻操作，precision 同時較低；AUROC 相同，表示整體的"
    "分辨能力沒有差別。"
)
e = D["ensemble"]["v2e2+tfidf"]
bullet(
    f"兩類模型用到的訊號大致重疊：v2 與 TF-IDF 的分數排名相關是 {D['spearman']['v2-epoch-2|tfidf']:.2f}；把兩者的 P(系統) 取平均，"
    f"測試集 AUROC 是 {f3(e['test_auroc'])}，與 v2 的差距 {sdiff(e['diff_vs_v2e2'])} {sci(e['ci'])} 不顯著。"
)

h2("4.4 門檻與篩檢效益（RQ4）")
def cls_row(name, point, thr_label):
    m = P[point]
    return [
        name,
        thr_label,
        f3(m["accuracy"]),
        f"{f3(m['macro_f1'])} {ci3(m['ci']['macro_f1'])}",
        f3(m["system_precision"]),
        f3(m["system_recall"]),
        f3(m["system_f1"]),
        str(m["tp"]),
        str(m["fp"]),
    ]


rows6 = [cls_row("第一輪 04（v1 epoch-3）", "04", "貪婪解碼")]
SHORT = {
    "v1-zero-shot": "zero-shot",
    "v1-epoch-1": "v1 epoch-1（選定）",
    "v1-epoch-2": "v1 epoch-2",
    "v1-epoch-3": "v1 epoch-3",
    "v2-epoch-1": "v2 epoch-1",
    "v2-epoch-2": "v2 epoch-2（選定）",
    "tfidf": "TF-IDF",
}
for k in ORDER:
    rows6.append(cls_row(SHORT[k], f"{k}@0.5", "0.5"))
    (tn, fp), (fn, tp) = CFG[k]["test_at_0.5"]["confusion_matrix"]["matrix"]
    assert (P[f"{k}@0.5"]["tp"], P[f"{k}@0.5"]["fp"]) == (tp, fp), k
    thr_point = f"{k}@thr"
    rows6.append(cls_row(SHORT[k], thr_point, f"thr* {P[thr_point]['threshold']:.3f}"))
sel_index = next(i for i, r in enumerate(rows6) if r[0] == SHORT["v2-epoch-2"] and r[1].startswith("thr*"))
table(
    ["設定", "門檻", "accuracy", "macro-F1（95% 區間）", "系統 precision", "系統 recall", "系統 F1", "TP", "FP"],
    rows6,
    [1800, 950, 950, 1850, 1000, 750, 650, 525, 525],
    ["l", "c", "r", "r", "r", "r", "r", "r", "r"],
    bold_rows=(sel_index,),
)
caption(
    "表 6　測試集在門檻 0.5 與 thr* 的分類結果（thr* 是各設定在驗證集 macro-F1 最高的門檻；"
    "TP 是判為「系統」且正確的筆數，FP 是誤報的筆數；測試集共 62 筆「系統」）"
)
table(
    ["人工 ＼ 預測", "個人", "系統", "合計", "recall"],
    [
        ["個人", str(SEL["tn"]), str(SEL["fp"]), "595", pct(SEL["tn"] / 595)],
        ["系統", str(SEL["fn"]), str(SEL["tp"]), "62", pct(SEL["tp"] / 62)],
        ["合計", str(SEL["tn"] + SEL["fn"]), str(SEL["fp"] + SEL["tp"]), "657", ""],
        ["precision", pct(SEL["tn"] / (SEL["tn"] + SEL["fn"])), pct(SEL["tp"] / (SEL["tp"] + SEL["fp"])), "", ""],
    ],
    [2200, 1700, 1700, 1700, 1700],
    ["l", "r", "r", "r", "r"],
)
caption(f"表 7　選定設定（v2 epoch-2、門檻 {SEL['threshold']:.3f}）的混淆矩陣（列＝人工標籤、欄＝模型預測）")
bullet(
    f"**系統事件的偵測率提高。**與第一輪 04 的實際輸出相比，選定設定多找到 {SEL['tp'] - OLD04['tp']} 筆「系統」"
    f"（{OLD04['tp']}→{SEL['tp']}），recall 提高 {dpp(c_sel_04['system_recall'])}，顯著。代價是誤報由 {OLD04['fp']} 筆增為 "
    f"{SEL['fp']} 筆（precision {dpp(c_sel_04['system_precision'])}）；macro-F1 的差距 {dci(c_sel_04['macro_f1'])} 不顯著。"
    f"accuracy 由 {pct(OLD04['accuracy'])} 降到 {pct(SEL['accuracy'])}，低於全判「個人」的 90.6%：類別這麼不平衡時，"
    "accuracy 不適合當主要指標。"
)
bullet(
    f"**提高來自決策規則，不是模型。**同樣用門檻 0.5 時，v2 與第一輪的差距都不顯著（macro-F1 {dci(c_sel05_04['macro_f1'])}）。"
    "第一輪的 recall 偏低，主要是因為貪婪解碼把門檻固定在 0.5。"
)
bullet(
    f"**門檻是可以依需求選擇的取捨。**以 v2 epoch-2 為例，門檻由 0.05 調到 0.30，測試集 macro-F1 都在 {min(band):.3f}–"
    f"{max(band):.3f}，但找到的「系統」由 {sens[0.05]['tp']} 筆降到 {sens[0.3]['tp']} 筆、誤報由 {sens[0.05]['fp']} 筆降到 "
    f"{sens[0.3]['fp']} 筆。門檻該設在哪裡，取決於漏掉一筆系統問題與多一筆誤報各自的代價。即使直接在測試集上挑門檻"
    f"（不能採用，只當作上限），五個 checkpoint 的 macro-F1 也只有 {min(ceil[k] for k in lora_keys):.3f}–"
    f"{max(ceil[k] for k in lora_keys):.3f}。"
)
rows8 = []
for key in ("0.10", "0.20", "0.30", "0.50"):
    a, b, z = G["v2-epoch-2"][key], G["tfidf"][key], G["v1-zero-shot"][key]
    share = f"{float(key) * 100:.0f}%"
    rows8.append([f"前 {share}", str(a["reviewed"]), f"{pc(a['recall'])} {pcci(a['ci'])}", f"{pc(b['recall'])} {pcci(b['ci'])}", pc(z["recall"]), share])
table(
    ["複核的通報", "筆數", "LoRA v2 找到的系統", "TF-IDF 找到的系統", "zero-shot", "隨機複核"],
    rows8,
    [1300, 900, 2300, 2300, 1100, 1100],
    ["l", "r", "r", "r", "r", "r"],
)
caption("表 8　篩檢效益：依 P(系統) 由高到低複核，找到的「系統」佔全部 62 筆的比例（方括號是 95% 信賴區間；同分的通報按比例計入）")
image("rId101", ANALYSIS / "fig_screening.png", 5619750)
caption("圖 4　篩檢效益的累積曲線：橫軸是依 P(系統) 由高到低複核的通報比例，縱軸是找到的「系統」事件比例")
missed_half = round(62 * (1 - g50["recall"]))
bullet(
    f"**把通報依 P(系統) 排序，系統事件集中在前面。**複核前 20% 的通報（{g20['reviewed']} 筆）找到 {pc(g20['recall'])} 的系統事件"
    f" {pcci(g20['ci'])}，是隨機複核的 {g20['recall'] / 0.2:.1f} 倍；前 30% 找到 {pc(g30['recall'])}，前 50% 找到 {pc(g50['recall'])}，"
    f"排在後半的通報裡只有 {missed_half} 筆系統事件。"
)
bullet(
    f"**複核量可以大幅減少。**在測試集上，要找到 80% 的系統事件，只需要複核 {RN['v2-epoch-2']['0.80'] * 100:.0f}% 的通報；找到 90% 需要 "
    f"{RN['v2-epoch-2']['0.90'] * 100:.0f}%。zero-shot 分別需要 {RN['v1-zero-shot']['0.80'] * 100:.0f}% 與 "
    f"{RN['v1-zero-shot']['0.90'] * 100:.0f}%。"
)
bullet(
    f"TF-IDF 的篩檢效益與 LoRA 相同（前 20% 找到 {pc(G['tfidf']['0.20']['recall'])} {pcci(G['tfidf']['0.20']['ci'])}），"
    "與 4.3 節的結論一致：在只讀描述的設定下，簡單的模型就能提供同樣的排序。"
)
bullet(
    f"驗證集只有 31 筆「系統」，調出的門檻有抽樣誤差：v2 epoch-2 在驗證集的 macro-F1 是 "
    f"{CFG['v2-epoch-2']['threshold']['val_macro_f1']:.3f}，到測試集是 {f3(SEL['macro_f1'])}；TF-IDF 調出的門檻"
    f"（{P['tfidf@thr']['threshold']:.3f}）在測試集反而比 0.5 差（{f3(P['tfidf@thr']['macro_f1'])} 對 {f3(TF['macro_f1'])}），"
    "所以本報告與 TF-IDF 比較時一律用它較好的 0.5。篩檢效益（表 8）不依賴門檻，沒有這個問題。"
)
c_e3_thr, c_e3_05 = CMP["v1e3_thr_minus_tfidf_thr"]["macro_f1"], CMP["v1e3_thr_minus_tfidf_05"]["macro_f1"]
assert significant(c_e3_thr) and not significant(c_e3_05)
bullet(
    f"checkpoint 依事先定的規則報告 epoch-2。v2 epoch-1（macro-F1 {f3(P['v2-epoch-1@thr']['macro_f1'])}）與 v1 epoch-3"
    f"（{f3(P['v1-epoch-3@thr']['macro_f1'])}）在測試集的數字較高，但不是規則選出的，看了測試集再改選等於用測試集挑模型。"
    f"05 的輸出檔是拿各設定與 TF-IDF 的 thr* 比較，其中只有 v1 epoch-3 的 macro-F1 差距區間不含 0（{dci(c_e3_thr)}）；"
    f"改與門檻 0.5 的 TF-IDF 比較後是 {dci(c_e3_05)}，區間包含 0。"
)

h2("4.5 三個模型共同的錯誤（RQ2、RQ5）")
ms = OV["missed"]
bullet(
    f"測試集 62 筆「系統」中，v1 epoch-1（thr*）漏 {ms['v1e1']} 筆、v2 epoch-2（thr*）漏 {ms['v2e2']} 筆、TF-IDF（0.5）漏 "
    f"{ms['tfidf']} 筆。三者都漏的有 {OV['all3']} 筆（表 9），至少被一個模型找到的有 {OV['any_of_3_hit']} 筆。"
)
bullet(
    f"兩版 LoRA 漏掉的案例高度重疊：兩者都漏 {OV['v1e1_and_v2e2']} 筆；v2 找回 {OV['v2e2_new_hit_vs_v1e1']} 筆 v1 漏掉的，"
    f"同時漏掉 {OV['v2e2_new_miss_vs_v1e1']} 筆 v1 找到的。v1 與 TF-IDF 都漏的 {OV['v1e1_and_tfidf']} 筆中，v2 只找回 "
    f"{OV['old16_recovered_by_v2e2']} 筆。"
)
bullet(
    f"這 {OV['all3']} 筆大多不是落在門檻邊緣：v2 給的 P(系統) 有 {len(low)} 筆低於 0.05，最低是 "
    f"{min(c['p_v2e2'] for c in D['all3_cases']):.4f}。模型不是不確定，而是有把握地判成「個人」，這與「描述裡沒有判定依據」"
    "的解釋相符（4.6 節）。"
)
bullet(f"驗證集 31 筆「系統」中，三者都漏的有 {OV['val']['all3']} 筆。這些案例是下一步逐筆複核的對象。")
EXCERPT = {
    "data.csv:450": "兒子到護理站表示病人掙脫約束帶，將左側PCN管路拔除",
    "data.csv:2895": "發現此住民跌坐在輪椅旁的地面上，無任何喊叫情形，自訴因想要練習站立",
    "data.csv:3103": "發現ND管路滑出約54公分(原固定於81公分)，告知專師後追蹤CXR確定位置",
    "data.csv:1471": "下唇紗布蓋住保護避免持續咬住嘴唇，於12:40探視病人時發現紗布已不見",
    "data.csv:2428": "兩人協力將剛煮好的大鍋茶葉蛋魯汁，蓋上鍋蓋後兩人從廚房拿到冰箱冰存",
    "data.csv:1440": "此時才發現病房廁所外天花板冷氣出風口漏水且地上濕濕的",
    "data.csv:1642": "病人左鼠蹊部中心靜脈導管固定於外露3公分處",
    "data.csv:1820": "發現病童左手抓管路，身上4.5號氣切已滑脫，血氧約80%",
    "data.csv:2194": "病患之前陸續皆有就診，當日看診完畢，同樣表示需要診斷書",
    "data.csv:2804": "22病房約18:05表示要交班轉床，交班表示病人中午有發燒",
    "data.csv:1497": "化學治療給藥時發現藥品與化療手寫單劑量不符合",
    "data.csv:2093": "發現此檢驗單與檢體的姓名不相符，致電到病房，詢問主護表示，姓名貼錯標籤",
}
rows9 = []


def small(x):
    return "<0.001" if x < 0.0005 else f3(x)


for c in D["all3_cases"]:
    if c["key"] == "data.csv:1475":
        same = next(o for o in D["all3_cases"] if o["key"] == "data.csv:2093")
        assert c["description"] == same["description"]
        excerpt = "描述與 data.csv:2093 完全相同"
    else:
        excerpt = EXCERPT[c["key"]]
        assert excerpt in c["description"], c["key"]
        excerpt = ("" if c["description"].startswith(excerpt) else "…") + excerpt + "…"
    rows9.append([c["key"], small(c["p_v1e1"]), small(c["p_v2e2"]), small(c["p_tfidf"]), excerpt])
assert len(rows9) == OV["all3"]
table(
    ["source_id", "v1 epoch-1", "v2 epoch-2", "TF-IDF", "描述（節錄）"],
    rows9,
    [1350, 800, 800, 800, 5250],
    ["l", "r", "r", "r", "l"],
)
caption(
    f"表 9　三個模型都漏判的 {OV['all3']} 筆「系統」與各模型給的 P(系統)（依 v2 的分數由低到高；門檻分別是 "
    f"{V1['threshold']:.3f}、{SEL['threshold']:.3f}、0.5）"
)

h2("4.6 判定所需的資訊在哪裡（RQ5）")
CUE, EV = IN["directive_cues"], IN["event_types"]
para(
    "為了解釋三個模型為什麼停在同一個水準、漏掉同一批案例，另外用同一個 TF-IDF＋邏輯迴歸模型，只換輸入欄位來比較"
    "（表 10、圖 5；analysis/idt_inputs.py）。這個分析只用 TF-IDF，目的是估計各欄位的資訊量。"
)
rows10 = []
for name, key in (
    ("description（模型實際讀到的）", "description"),
    ("directive（批示），去掉流程用語", "directive（去掉流程用語）"),
    ("directive（批示）", "directive"),
    ("description＋directive", "description+directive"),
):
    item = FI[key]
    gap = item.get("minus_description")
    cv = CV.get(key)
    rows10.append([
        name, f3(item["macro_f1"]), f3(item["system_recall"]), f"{f3(item['auroc'])} {ci3(item['ci']['auroc'])}",
        dci(gap["auroc"]) if gap else "—", f3(cv["auroc"]) if cv else "—",
    ])
table(
    ["輸入欄位", "macro-F1", "系統 recall", "AUROC（95% 區間）", "AUROC 減 description", "5-fold AUROC"],
    rows10,
    [2350, 900, 900, 1850, 2100, 900],
    ["l", "r", "r", "r", "r", "r"],
)
caption(
    "表 10　同一個 TF-IDF＋邏輯迴歸模型換輸入欄位的結果（測試集、門檻 0.5；5-fold 是全部 3,288 筆的分組交叉驗證，"
    "相同描述同組，三個亂數種子的平均）"
)
image("rId102", ANALYSIS / "fig_fields.png", 5619750)
caption("圖 5　輸入欄位與測試集 AUROC：只讀事件描述的兩個模型落在同一個水準，讀到批示後明顯提高")
gd, gc = FD["minus_description"], FDC["minus_description"]
bullet(
    f"**讀到批示，分辨能力顯著提高。**同一個模型由讀事件描述改成讀批示，測試集 AUROC 由 {f3(FI['description']['auroc'])} 升到 "
    f"{f3(FD['auroc'])}（差距 {dci(gd['auroc'])}），macro-F1 由 {f3(FI['description']['macro_f1'])} 升到 {f3(FD['macro_f1'])}"
    f"（差距 {dci(gd['macro_f1'])}）；全部資料的交叉驗證方向一致（{f3(CV['description']['auroc'])} 對 {f3(CV['directive']['auroc'])}）。"
)
bullet(
    f"**提高不只是來自流程用語。**批示裡有與標籤相關的流程用語：「敬會」出現在 {pct(CUE['敬會']['system_rate'])} 的「系統」批示與 "
    f"{pct(CUE['敬會']['person_rate'])} 的「個人」批示，「PDCA」是 {pct(CUE['PDCA']['system_rate'])} 對 {pct(CUE['PDCA']['person_rate'])}。"
    f"把這些用語與會辦單位名稱去掉後，AUROC 仍有 {f3(FDC['auroc'])}（與描述的差距 {dci(gc['auroc'])}），差距依然顯著。"
)
bullet(
    "**相同的描述可以對應不同的標籤。**data.csv:543、548、549 的描述一字不差（手術通知單的處置代碼與術前診斷位置不同，"
    "未造成傷害），543 標「系統」，548 與 549 標「個人」。三筆批示記載的原因各不相同，543 寫的是系統沒有雙側的處置代碼選項。"
    "標籤反映的是調查結果，從描述看不出來。"
)
bullet(
    f"**標籤是一致的。**描述相似度 0.8 以上的 {ND['pairs']} 對紀錄中，標籤相同的有 {ND['pairs'] - ND['label_conflicts']} 對"
    f"（{(ND['pairs'] - ND['label_conflicts']) / ND['pairs'] * 100:.0f}%）；全部資料中描述完全相同而標籤不同的只有上面這一組。"
    "第一輪報告把這類案例解讀為「人工標籤不一致」，應修正為：標註是一致的，但描述本身常常不足以判斷。"
)
bullet(
    f"批示中寫有「原因說明」的紀錄佔 {pct(RC['all'])}（「系統」{pct(RC['system'])}、「個人」{pct(RC['person'])}）。"
    "若要讓模型使用調查資訊，需要先界定可用的內容（例如只取原因說明），並排除流程用語。"
)
bullet(
    f"只看描述時，模型能利用的主要是事件類型的基準比例。以關鍵字粗分，「系統」的比例在跌倒是 {pct(EV['跌倒']['system_rate'])}、"
    f"自拔管路 {pct(EV['自拔管路']['system_rate'])}，在檢驗／檢體是 {pct(EV['檢驗／檢體']['system_rate'])}、設備／資訊 "
    f"{pct(EV['設備／資訊']['system_rate'])}、傳送 {pct(EV['傳送']['system_rate'])}。"
)

# ------------------------------ 5. 討論 ------------------------------
h1("5. 討論")
h2("5.1 發現的意涵")
bullet(
    "**通報當下：適合做優先複核，不適合自動判定。**只憑事件描述，模型能把大部分系統事件排到前面（前 20% 涵蓋八成以上），"
    f"可以用來決定哪些通報先看、哪些需要追問。但要找到三分之二的系統事件時，判為「系統」的通報只有不到一半是對的"
    f"（選定設定的 precision 是 {pct(SEL['system_precision'])}），不足以取代人工判定。"
)
bullet(
    "**調查之後：判定所需的資訊才出現。**IDT 決策樹中關於作業規範是否適當（Q6）、替代測試（Q7）與教育訓練、督導（Q8）的題目，"
    "多半要有調查結果才能回答；批示的分析也顯示資訊在這裡。後續研究宜把任務分成兩個：通報當下的篩檢，以及調查後的輔助判定。"
)
bullet(
    "**模型的選擇。**在只讀描述的設定下，字元 n-gram 的線性模型就能達到與 9B 語言模型相同的排序，成本低得多。"
    "語言模型是否在需要理解調查敘述的第二個任務上有優勢，是下一步要驗證的問題。"
)
bullet(
    "**對評估方法的啟示。**第一輪只看貪婪解碼的結果時，看不出模型的排序能力其實已經不差，也看不出問題在門檻；"
    "改用機率評分、加上強基準與信賴區間之後，才能分辨「模型不好」「門檻不對」與「資訊不夠」。"
)
h2("5.2 本研究的優點")
bullet("資料切分以相同描述為單位分組，並確認沒有描述跨集合；驗證集與測試集不含增生資料。")
bullet("checkpoint 與門檻的挑選規則事先定下、只用驗證集；測試集只用於報告，報告時不因測試集的數字改選設定。")
bullet("同時有不微調的對照（zero-shot）與傳統的強基準（TF-IDF），可以分別回答「微調有沒有用」與「語言模型有沒有必要」。")
bullet("所有比較都用同一組 bootstrap 重抽樣的配對差距，報告信賴區間，不只比較點估計。")
bullet("以機率評分取代生成文字，並處理了 bfloat16 下分數離散造成的同分；評分逐筆存檔，評估可以不載模型重算。")
bullet("負面的結果（語言模型沒有超過 TF-IDF）有後續分析解釋原因，並指出下一步的方向，而不是停在沒有進步。")
h2("5.3 第一輪留下的問題與本輪的回答")
table(
    ["第一輪報告提出的問題", "本輪的回答"],
    [
        [
            "類別不平衡，預測「系統」偏保守（46 筆，實際 62 筆）",
            f"原因是貪婪解碼把門檻固定在 0.5。由驗證集決定門檻後，recall 由 {pct(OLD04['system_recall'])} 提高到 "
            f"{pct(SEL['system_recall'])}；當成篩檢工具時，前 20% 的通報涵蓋 {pc(g20['recall'])} 的系統事件",
        ],
        [
            "過擬合，最終 adapter 卻是最後一個 epoch",
            "以驗證集 log-loss 挑 checkpoint：v1 選出 epoch-1（也是測試集 log-loss 最低者）；v2 的兩個 checkpoint 驗證損失持平，選出 epoch-2",
        ],
        [
            "LoRA 只匹配 16 個模組",
            f"原因是模型有 24 層使用線性注意力。v2 擴大到 200 個模組、可訓練參數約 {ratio:.0f} 倍，表現相同：小的 adapter 已經足夠",
        ],
        [
            "沒有不微調的對照",
            f"zero-shot 的 AUROC 是 {f3(ZS['auroc'])}，微調後 {f3(SEL['auroc'])}，差距顯著；同時加入 TF-IDF 基準（{f3(TF['auroc'])}）",
        ],
        [
            "增生資料對 IDT 的淨效益未知",
            "不用增生的 v2 與 v1 表現相同；以 TF-IDF 估計，加入改寫也沒有提高。之後可以省去增生"
            "（v2 同時改了 LoRA 範圍，個別的影響無法拆開）",
        ],
        [
            "人工標籤不一致",
            f"標籤是一致的：近似描述的 {ND['pairs']} 對中只有 {ND['label_conflicts']} 對標籤不同；差別來自批示記載的調查結果",
        ],
        [
            "val_label_match 看不出少數類的學習情形",
            "改用驗證集的 log-loss、AUROC、AP 與 macro-F1",
        ],
    ],
    [3300, 5700],
)
caption("表 11　第一輪報告 6.1、6.3 節提出的問題與本輪的回答")
h2("5.4 限制")
distinct = list(D["distinct_scores_test"].values())
bullet(
    "樣本小：測試集只有 62 筆「系統」，macro-F1 的 95% 信賴區間寬約 ±0.05；驗證集 31 筆，挑出的門檻有抽樣誤差。"
    "「沒有差別」的結論，是指差距小於這個解析度。"
)
bullet(
    f"v1 與 v2 之間同時改了訓練資料、LoRA 範圍、rank 與訓練步數（v2 共 {S2[-1]} 步，v1 的第一個 epoch 就有 {S1[0]} 步），"
    "只能比較整體，不能拆出各項的貢獻，也不能判斷 v2 訓練更久會不會過擬合；兩版都只訓練一個亂數種子。"
)
bullet("資料來自單一來源，還沒有在其他時期或其他機構的通報上驗證；門檻與篩檢效益在新資料上可能不同。")
bullet(
    "輸入欄位的分析（4.6 節）只用 TF-IDF，是資訊量的估計，不是語言模型讀批示的實驗；批示寫於調查之後，"
    "其中哪些內容可以合理地當成輸入，還需要界定。"
)
bullet(
    f"分數是離散的：bfloat16 下，測試集 657 筆在每個語言模型設定下只有 {min(distinct)}–{max(distinct)} 個不同的 P(系統)。"
    "評估時已合併同分，但分數檔裡的浮點誤差仍在；之後可以改成直接由兩個標籤的 logit 差計算 P(系統)。評分在 MPS、訓練在 CUDA "
    "上執行，數值差異會翻轉接近門檻的個案（v1 epoch-3 在 05 的重現與 04 的輸出有 1／657 筆不同）。"
)
bullet("情緒標註與評審本輪沒有重跑，限制同第一輪報告的 6.2 節。")
h2("5.5 投稿前建議補強的工作")
bullet(
    f"**逐筆複核共同漏判的案例**（測試集 {OV['all3']} 筆、驗證集 {OV['val']['all3']} 筆），分成：描述裡有線索但模型沒找到、"
    "線索只在批示、標註有疑問。這能把「資訊不在描述裡」由統計上的推論變成直接的證據。"
)
bullet("**多個亂數種子與分組交叉驗證**：以 train＋val 做分組 5-fold、每個設定訓練數次，報告變異；測試集留到最後只用一次。")
bullet(
    "**讓語言模型讀調查資訊的實驗**：以批示中的原因說明為輸入（排除流程用語），比較語言模型與 TF-IDF，"
    "回答語言模型在第二個任務上有沒有優勢。"
)
bullet("**篩檢效益的外部驗證**：以另一段時間的通報驗證表 8 的數字，並與實際的複核流程對照，決定操作點。")
bullet(
    f"**更多「系統」案例**：以 TF-IDF 估計的學習曲線，訓練資料由一半增加到全部時 macro-F1 由 {f3(LC['0.50']['macro_f1'])} 升到 "
    f"{f3(LC['1.00']['macro_f1'])}，還沒有飽和。"
)
bullet("**標註一致性的直接量測**：由第二位標註者重標一部分案例並計算一致性，補足 4.6 節的間接證據。")
bullet("**情緒部分是否納入論文**：若納入，需要人工抽樣複核（第一輪報告 6.2 節）。")

# ------------------------------ 6. 可重現性與執行紀錄 ------------------------------
h1("6. 可重現性與執行紀錄")
bullet(
    "**重新產生本報告。**在專案根目錄依序執行 python -m analysis.idt_comparison、analysis.idt_inputs、analysis.report_charts、"
    "analysis.research_flow --png、analysis.build_report。報告裡的數字都由 results/ 下的分數檔與評估檔代入；"
    "內文的判斷以程式中的檢查綁在數字上，數字改變、敘述不再成立時會中止。"
)
bullet(
    "**訓練與評分的指令**見 README 的「正式執行」。模型權重（*.safetensors、*.pt）單檔超過 GitHub 的 100 MB 上限，不進版本庫；"
    "v2 adapter 的設定檔與 training_metadata.json 在 models/idt_adapter_v2/，逐筆分數與評估結果在 results/idt_scores_v2/，"
    "重算評估不需要權重。"
)
bullet(
    f"**執行環境與時間。**v2 訓練在 vast.ai 的 NVIDIA GB10 上執行，共 {META['v2']['duration_minutes']} 分鐘。05 的評分在 "
    "Apple Silicon（mps、bfloat16）上執行：v2 兩個 checkpoint 共 1,972 筆，147.9 分鐘；v1 的三個 checkpoint 與 zero-shot 於 "
    "2026-11-20 評分，v2 於 2026-11-22 評分，兩次的資料與提示相同。"
)
bullet(
    "**GB10 上載入模型的問題。**transformers 載入的權重仍對應在硬碟的檔案上，直接 model.to(\"cuda\") 在這台機器上超過十分鐘"
    "沒有完成；改成每個權重先複製到記憶體再搬上 GPU 後，17.5 GiB 在 5 秒內搬完（utils/llm.py 的 move_to_device）。"
)
bullet(
    "**同分處理的修正（2026-11-23）。**修正前，05 把只差浮點誤差的分數當成不同的值，驗證集挑出的門檻可能落在同分的紀錄之間。"
    "修正後重算評估（只讀分數檔）：選出的 checkpoint、log-loss 與門檻 0.5 的結果不變；thr* 的測試集 macro-F1 變動在 0.006 以內、"
    "AUROC 在 0.001 以內、AP 在 0.008 以內。本報告的數字都是修正後的結果。"
)
ag = {k: CFG[k]["agreement_with_reference"] for k in ("v1-epoch-3", "v2-epoch-1", "v2-epoch-2")}
bullet(
    f"**與 04 的核對。**v1 epoch-3 在 05 門檻 0.5 的判斷與 04 的貪婪解碼有 {ag['v1-epoch-3']['matched']}／657 筆一致，"
    f"確認機率評分與生成的結果相符。v2 的兩個 checkpoint 與 04 一致的筆數是 {ag['v2-epoch-1']['matched']} 與 "
    f"{ag['v2-epoch-2']['matched']}，反映的是兩版模型的差異。"
)

# ------------------------------ 7. 未重跑 ------------------------------
h1("7. 本輪未重跑的部分")
bullet(
    "02 特徵工程（③ 增生、④ 訓練集情緒標註）與 04 的情緒推論、情緒評審（⑦⑨）沒有重跑，結果與第一輪報告的第 3 節、"
    "5.2 節相同：測試集的情緒預測以「中性」為主（58.6%），評審平均總分 4.20（中性 5.00、非中性 3.07）。"
)
bullet(
    f"04 的 IDT 貪婪解碼與評估（⑥⑧）沒有用 v2 重跑。v2 在門檻 0.5 的結果與貪婪解碼等價：macro-F1 {f3(SEL05['macro_f1'])}、"
    f"「系統」recall {pct(SEL05['system_recall'])}、precision {pct(SEL05['system_precision'])}（表 6）。"
)
bullet("第一輪的報告與當時的流程圖在 docs/2026-10-18/。")


# ============================== 組檔 ==============================
scratch = Path(tempfile.mkdtemp(prefix="report_"))
work = scratch / "pkg"
with zipfile.ZipFile(ARGS.template) as template:
    template.extractall(work)
old_doc = (work / "word/document.xml").read_text(encoding="utf-8")
root_open = old_doc[: old_doc.index("<w:body>")]
sect = re.findall(r"<w:sectPr.*?</w:sectPr>", old_doc, flags=re.S)[-1]
sect = re.sub(r' w:rsid\w*="[^"]*"', "", sect)
document = root_open + "<w:body>" + "".join(body) + sect + "</w:body></w:document>"
(work / "word/document.xml").write_text(document, encoding="utf-8")
MEDIA = {  # 關聯 id → (封裝內的檔名, 來源)；rId7–9 是樣式來源原有的三張圖，其餘是新增的
    "rId7": ("image1.png", FLOW_PNG),
    "rId8": ("image2.png", ANALYSIS / "fig_ci.png"),
    "rId9": ("image3.png", ANALYSIS / "fig_training.png"),
    "rId101": ("image4.png", ANALYSIS / "fig_screening.png"),
    "rId102": ("image5.png", ANALYSIS / "fig_fields.png"),
}
rels_path = work / "word/_rels/document.xml.rels"
rels_xml = rels_path.read_text(encoding="utf-8")
for rid, (name, source) in MEDIA.items():
    shutil.copyfile(source, work / "word/media" / name)
    if f'Id="{rid}"' not in rels_xml:
        rels_xml = rels_xml.replace(
            "</Relationships>",
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{name}"/></Relationships>',
        )
rels_path.write_text(rels_xml, encoding="utf-8")

# 頁首的日期
header = (work / "word/header2.xml").read_text(encoding="utf-8")
assert header.count(" 2026-10-18") == 1, "樣式來源的頁首不是預期的格式"
(work / "word/header2.xml").write_text(header.replace(" 2026-10-18", f" {REPORT_DATE}"), encoding="utf-8")

# 拿掉 Word 增益集面板（與報告內容無關）
shutil.rmtree(work / "word/webextensions")
types = (work / "[Content_Types].xml").read_text(encoding="utf-8")
types, removed = re.subn(r'<Override PartName="/word/webextensions/[^"]+" ContentType="[^"]+"/>', "", types)
assert removed == 3
(work / "[Content_Types].xml").write_text(types, encoding="utf-8")
rels = (work / "_rels/.rels").read_text(encoding="utf-8")
rels, removed = re.subn(r'<Relationship Id="rId2" Type="[^"]*webextensiontaskpanes" Target="[^"]+"/>', "", rels)
assert removed == 1
(work / "_rels/.rels").write_text(rels, encoding="utf-8")

now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
(work / "docProps/core.xml").write_text(
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    "<dc:title>醫療事件 NLP 實驗結果報告（第二輪）</dc:title><dc:creator>Chih-Chien Hsieh</dc:creator>"
    "<dc:description>Medical-Incident-NLP：IDT 機率評分與 v2 重訓的第二輪實驗結果</dc:description>"
    "<cp:lastModifiedBy>Chih-Chien Hsieh</cp:lastModifiedBy><cp:revision>1</cp:revision>"
    f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
    f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>',
    encoding="utf-8",
)
(work / "docProps/app.xml").write_text(
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
    'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Template>Normal.dotm</Template>'
    "<Application>Microsoft Office Word</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop>"
    "<SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion></Properties>",
    encoding="utf-8",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()
files = sorted(p for p in work.rglob("*") if p.is_file() and p.name != ".DS_Store")
files.sort(key=lambda p: (p.name != "[Content_Types].xml", str(p.relative_to(work))))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for path in files:
        z.write(path, str(path.relative_to(work)))
shutil.rmtree(scratch)
print(f"{OUT}（{OUT.stat().st_size:,} bytes；表格 {document.count('<w:tbl>')}、圖 {_pic}）")
