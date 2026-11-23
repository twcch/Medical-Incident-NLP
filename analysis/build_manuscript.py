"""組出期刊論文的第 3 章（研究方法）與第 4 章（實驗結果）：docs/manuscript_ch3_ch4_<日期>.docx。

內容來自 2026-11-23 的實驗結果，但以論文的寫法呈現：正式的段落、編號公式、三線表（表題在上）、
圖題在下；不出現腳本名稱、輪次或內部代號。論文中的命名：LoRA-S＝第一輪的 v1（只調整注意力的
q／v 投影），LoRA-L＝第二輪的 v2（全部線性層）；「調查處理紀錄」＝資料中的批示（directive）。
【　】標出的是需要作者補上的資訊（機構、期間、倫理審查等），程式不會自行編造。

版面採投稿稿件的慣用格式：12 pt、兩倍行高、Times New Roman＋新細明體、黑色標題。行內的數學符號用一般文字
（斜體與上下標），編號公式用 Word 的公式物件；表格每一格都會檢查放不放得進欄寬。

數字由下列檔案代入，敘述以 assert 綁在數字上：
    results/analysis/idt_comparison.json、idt_inputs.json     python -m analysis.idt_comparison／idt_inputs
    results/analysis/ms_fig*.png                              python -m analysis.manuscript_figures
    models/idt_adapter*/training_metadata.json、data/processed_data/*.json

在專案根目錄執行：
    python -m analysis.build_manuscript                   # → docs/manuscript_ch3_ch4_2026-11-23.docx
產生後請用 Word 開啟檢查分頁與公式。
"""

import argparse
import json
import re
import shutil
import statistics
import tempfile
import unicodedata
import zipfile
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageFont

ROOT = Path.cwd()
parser = argparse.ArgumentParser(description="組出期刊論文第 3、4 章的 .docx")
parser.add_argument("--date", default="2026-11-23", help="寫在頁首的日期")
parser.add_argument("--template", type=Path, default=ROOT / "docs/2026-10-18/experiment_report_2026-10-18.docx", help="樣式來源")
parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results/analysis")
parser.add_argument("--out", type=Path, default=None, help="預設 docs/manuscript_ch3_ch4_<date>.docx")
ARGS = parser.parse_args()
ANALYSIS = ARGS.analysis_dir
OUT = ARGS.out or ROOT / f"docs/manuscript_ch3_ch4_{ARGS.date}.docx"
FIGURES = [ANALYSIS / f"ms_fig{i}_{name}.png" for i, name in enumerate(("framework", "discrimination", "training", "gain", "inputs"), 1)]
for needed in (ARGS.template, ANALYSIS / "idt_comparison.json", ANALYSIS / "idt_inputs.json", *FIGURES):
    if not needed.is_file():
        raise SystemExit(f"找不到 {needed}；產生方式見這個檔案開頭的說明。")
D = json.loads((ANALYSIS / "idt_comparison.json").read_text(encoding="utf-8"))
IN = json.loads((ANALYSIS / "idt_inputs.json").read_text(encoding="utf-8"))
P, CMP, OV, G = D["points"], D["comparisons"], D["overlap"], D["gain"]
CFG = {row["key"]: row for row in D["configs"]}
MS, ML = (json.loads((ROOT / f"models/{name}/training_metadata.json").read_text(encoding="utf-8")) for name in ("idt_adapter", "idt_adapter_v2"))
SETS = {name: json.loads((ROOT / f"data/processed_data/{name}_data.json").read_text(encoding="utf-8")) for name in ("train", "val", "test")}
EVERYTHING = [record for rows in SETS.values() for record in rows]


# ------------------------------ 數字格式 ------------------------------
def minus(text):
    return text.replace("-", "−")


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x):
    return f"{x * 100:.1f}%"


def pc(x):
    return f"{x * 100:.0f}%"


def ci3(pair):
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def pcci(pair):
    return f"[{pair[0] * 100:.0f}%, {pair[1] * 100:.0f}%]"


def signed(x):
    """帶正負號的三位小數；四捨五入後為 0 時不帶符號。"""
    text = f"{x:+.3f}"
    return minus(text[1:] if float(text) == 0 else text)


def dci(item):
    """差異與其 95% 信賴區間，帶正負號。"""
    return f"{signed(item['diff'])} [{signed(item['ci'][0])}, {signed(item['ci'][1])}]"


def increase(item):
    """方向已由文字說明的增加量：不帶正負號，區間必須完全大於 0。"""
    assert item["ci"][0] > 0, item
    return f"{item['diff']:.3f} [{item['ci'][0]:.3f}, {item['ci'][1]:.3f}]"


def pp_change(item):
    """以百分點表示的變化量（取絕對值），方向由文字說明；區間必須不含 0。"""
    assert significant(item), item
    lower, upper = sorted(abs(v) * 100 for v in item["ci"])
    return f"{abs(item['diff']) * 100:.1f} 個百分點 [{lower:.1f}, {upper:.1f}]"


def significant(item):
    return not (item["ci"][0] <= 0 <= item["ci"][1])


def n(x):
    return f"{x:,}"


def length_quartiles(records, field="description"):
    """文字長度（字元數）的第 1 四分位數、中位數、第 3 四分位數；只計入有文字的通報。"""
    lengths = sorted(len(r.get(field) or "") for r in records if (r.get(field) or "").strip())
    q1, _, q3 = statistics.quantiles(lengths, n=4, method="inclusive")
    return q1, statistics.median(lengths), q3


def length_summary(records, field="description"):
    """表格用：中位數（第 1–第 3 四分位數）。"""
    q1, median, q3 = length_quartiles(records, field)
    return f"{median:.0f}（{q1:.0f}–{q3:.0f}）"


def length_text(records, field="description"):
    """內文用：100 字（四分位距 50–196 字）。"""
    q1, median, q3 = length_quartiles(records, field)
    return f"{median:.0f} 字（四分位距 {q1:.0f}–{q3:.0f} 字）"


# ------------------------------ WordprocessingML：論文樣式 ------------------------------
BODY = 24  # 內文 12 pt
TABLE = 18  # 表格內文 9 pt
CELL_PADDING_PT = 11  # 儲存格左右邊界各 4 pt，再留 3 pt 的餘裕
body: list[str] = []
_counter = {"eq": 0, "pic": 0}


@lru_cache
def _times(bold):
    path = Path("/System/Library/Fonts/Supplemental") / ("Times New Roman Bold.ttf" if bold else "Times New Roman.ttf")
    return ImageFont.truetype(str(path), 200) if path.is_file() else None


def text_width(text, points, bold=False):
    """估計一行文字在 Word 裡的寬度（pt）：全形字元算一個字寬，其餘以 Times New Roman 量測；沒有這個字型時回傳 None。"""
    font = _times(bold)
    if font is None:
        return None
    wide = sum(unicodedata.east_asian_width(ch) in "WF" for ch in text)
    narrow = "".join(ch for ch in text if unicodedata.east_asian_width(ch) not in "WF")
    return (wide + font.getlength(narrow) / 200) * points


def runs(text, size=BODY, color=None, bold=False):
    """文字 → 一串 <w:r>。支援 `**粗體**`、換行與 `$…$` 行內數學；τ 與 τ* 一律排成數學符號。"""
    out = []
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        props, strong = [], bold or i % 2 == 1
        if strong:
            props.append("<w:b/><w:bCs/>")
        if color:
            props.append(f'<w:color w:val="{color}"/>')
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
        rpr = f"<w:rPr>{''.join(props)}</w:rPr>"
        for k, piece in enumerate(part.split("$")):
            if k % 2 == 1:
                out.append(inline_math(piece, size, color, strong))
                continue
            for token in re.split(r"(τ\*?)", piece):
                if token in ("τ", "τ*"):
                    out.append(inline_math("τ^*" if token == "τ*" else "τ", size, color, strong))
                    continue
                for j, line in enumerate(token.split("\n")):
                    if j:
                        out.append(f"<w:r>{rpr}<w:br/></w:r>")
                    if line:
                        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(line)}</w:t></w:r>')
    return "".join(out)


def h1(text):
    body.append(f'<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr>{runs(text, size=28)}</w:p>')


def h2(text):
    body.append(f'<w:p><w:pPr><w:pStyle w:val="2"/></w:pPr>{runs(text, size=24)}</w:p>')


def para(text, indent=True):
    """內文段落：左右對齊、首行縮排兩字、兩倍行高（投稿稿件的慣用格式）。"""
    ind = '<w:ind w:firstLineChars="200" w:firstLine="480"/>' if indent else ""
    body.append(
        f'<w:p><w:pPr><w:spacing w:after="0" w:line="480" w:lineRule="auto"/>{ind}<w:jc w:val="both"/></w:pPr>{runs(text)}</w:p>'
    )


def m_r(text, plain=False, size=BODY, color=None):
    style = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    ink = f'<w:color w:val="{color}"/>' if color else ""
    return (
        f'<m:r>{style}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>{ink}<w:sz w:val="{size}"/>'
        f'<w:szCs w:val="{size}"/></w:rPr><m:t xml:space="preserve">{escape(text)}</m:t></m:r>'
    )


def m_sub(base, sub):
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"


def m_hat(base):
    return '<m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e>' + base + "</m:e></m:acc>"


def m_frac(numerator, denominator):
    return f"<m:f><m:num>{numerator}</m:num><m:den>{denominator}</m:den></m:f>"


def m_sum(lower, upper, expression):
    """求和符號；upper 為 None 時不顯示上限。"""
    hide = "" if upper is not None else '<m:supHide m:val="1"/>'
    return (
        f'<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/>{hide}</m:naryPr>'
        f"<m:sub>{lower}</m:sub><m:sup>{upper or ''}</m:sup><m:e>{expression}</m:e></m:nary>"
    )


LETTERS = "A-Za-zΑ-Ωα-ω"


def inline_math(source, size=BODY, color=None, bold=False):
    """`x_i`、`z_{sys}`、`τ^*` 這類行內數學 → 一般文字的 <w:r>：變數斜體，數字與運算符號正體，上下標用 vertAlign。

    行內符號不用公式物件，在 Word 以外的檢視器也能正確顯示；編號公式才用 OMML。
    """

    def run(text, italic, align=None):
        props = '<w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
        props += "<w:b/><w:bCs/>" if bold else ""
        props += "<w:i/><w:iCs/>" if italic else ""
        props += f'<w:color w:val="{color}"/>' if color else ""
        props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        props += f'<w:vertAlign w:val="{align}"/>' if align else ""
        return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'

    def upright_or_italic(text):
        pieces = re.findall(f"[{LETTERS}]+|[^{LETTERS}]+", text)
        return "".join(run(piece, bool(re.match(f"[{LETTERS}]", piece))) for piece in pieces)

    out, position = [], 0
    for match in re.finditer(r"([_^])(?:\{([^}]*)\}|(.))", source):
        out.append(upright_or_italic(source[position : match.start()]))
        script = match.group(2) if match.group(2) is not None else match.group(3)
        italic = len(script) == 1 and bool(re.match(f"[{LETTERS}]", script))
        out.append(run(script, italic, "subscript" if match.group(1) == "_" else "superscript"))
        position = match.end()
    out.append(upright_or_italic(source[position:]))
    return "".join(out)


def equation(*parts):
    """置中的編號公式；回傳公式編號。"""
    _counter["eq"] += 1
    number = _counter["eq"]
    math = "".join(parts)
    body.append(
        '<w:p><w:pPr><w:tabs><w:tab w:val="center" w:pos="4513"/><w:tab w:val="right" w:pos="9026"/></w:tabs>'
        '<w:spacing w:before="120" w:after="120" w:line="480" w:lineRule="auto"/></w:pPr><w:r><w:tab/></w:r>'
        f"<m:oMath>{math}</m:oMath><w:r><w:tab/></w:r>" + runs(f"({number})") + "</w:p>"
    )
    return number


RULE = '<w:{side} w:val="single" w:sz="{size}" w:space="0" w:color="000000"/>'
CELL_MARGIN = (
    '<w:tcMar><w:top w:w="45" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
    '<w:bottom w:w="45" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tcMar>'
)


def table(number, caption_text, headers, rows, widths, aligns=None, note=None, bold_rows=(), rule_before=(), wrap=()):
    """三線表：表題在上，上下粗線、表頭下細線，沒有直線。

    rule_before 是上方要加細線的列索引；wrap 是允許自動換行的欄索引，其餘欄位（含表頭）每一行都必須放得進欄寬，
    需要換行的地方以 "\n" 明確指定。
    """
    assert sum(widths) == 9000, sum(widths)
    assert all(len(row) == len(headers) for row in rows), [len(r) for r in rows]
    for r, row in enumerate([headers, *rows]):
        for c, (content, width) in enumerate(zip(row, widths, strict=True)):
            for line in str(content).replace("**", "").replace("$", "").split("\n"):
                needed = text_width(line, TABLE / 2, bold=r - 1 in bold_rows or "**" in str(content))
                fits = needed is None or needed + CELL_PADDING_PT <= width / 20
                where = f"表 {number} 第 {r} 列第 {c + 1} 欄" if r else f"表 {number} 表頭第 {c + 1} 欄"
                assert fits or c in wrap, f"{where}放不下「{line}」：需要 {needed + CELL_PADDING_PT:.0f} pt，欄寬 {width / 20:.0f} pt"
    aligns = aligns or ["l"] * len(headers)
    jc = {"l": "", "c": '<w:jc w:val="center"/>', "r": '<w:jc w:val="right"/>'}
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    body.append(
        '<w:p><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="80" w:line="300" w:lineRule="auto"/></w:pPr>'
        + runs(f"表 {number}", size=20, bold=True)
        + runs(f"　{caption_text}", size=20)
        + "</w:p>"
    )
    out = [
        '<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>'
        + RULE.format(side="top", size=12)
        + '<w:left w:val="nil"/>'
        + RULE.format(side="bottom", size=12)
        + '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders><w:tblLayout w:type="fixed"/>'
        '<w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
        f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>"
    ]

    def row_xml(cells, header=False, bold=False, top_rule=False):
        borders = ""
        if header:
            borders = f"<w:tcBorders>{RULE.format(side='bottom', size=6)}</w:tcBorders>"
        elif top_rule:
            borders = f"<w:tcBorders>{RULE.format(side='top', size=4)}</w:tcBorders>"
        xml = "".join(
            f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{borders}{CELL_MARGIN}<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:keepNext/><w:spacing w:line="260" w:lineRule="auto"/>{jc["c"] if header else jc[a]}</w:pPr>'
            f"{runs(str(cell), size=TABLE, bold=bold)}</w:p></w:tc>"
            for cell, w, a in zip(cells, widths, aligns, strict=True)
        )
        props = "<w:trPr><w:tblHeader/></w:trPr>" if header else "<w:trPr><w:cantSplit/></w:trPr>"
        return f"<w:tr>{props}{xml}</w:tr>"

    out.append(row_xml(headers, header=True))
    out.extend(row_xml(row, bold=i in bold_rows, top_rule=i in rule_before) for i, row in enumerate(rows))
    out.append("</w:tbl>")
    body.append("".join(out))
    if note:
        body.append(
            '<w:p><w:pPr><w:spacing w:before="60" w:after="200" w:line="260" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
            + runs(note, size=17, color="3A3A3A")
            + "</w:p>"
        )
    else:
        body.append('<w:p><w:pPr><w:spacing w:after="120" w:line="120" w:lineRule="exact"/></w:pPr></w:p>')


def figure(number, caption_text, width_emu=5619750):
    """圖在上、圖題在下；圖檔依編號取 FIGURES。"""
    _counter["pic"] += 1
    pic = _counter["pic"]
    with Image.open(FIGURES[number - 1]) as im:
        w, h = im.size
    cx, cy = width_emu, round(width_emu * h / w)
    body.append(
        '<w:p><w:pPr><w:keepNext/><w:spacing w:before="200"/><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:noProof/></w:rPr>'
        f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{pic}" name="圖 {number}"/><wp:cNvGraphicFramePr>'
        '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
        '</wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr>'
        f'<pic:cNvPr id="{pic}" name="圖 {number}"/><pic:cNvPicPr><a:picLocks noChangeAspect="1" noChangeArrowheads="1"/>'
        f'</pic:cNvPicPr></pic:nvPicPr><pic:blipFill><a:blip r:embed="rIdFig{number}"/><a:srcRect/><a:stretch><a:fillRect/>'
        f'</a:stretch></pic:blipFill><pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/>'
        '</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        "</wp:inline></w:drawing></w:r></w:p>"
    )
    body.append(
        '<w:p><w:pPr><w:spacing w:before="60" w:after="240" w:line="300" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
        + runs(f"圖 {number}", size=20, bold=True)
        + runs(f"　{caption_text}", size=20)
        + "</w:p>"
    )


# ------------------------------ 取值與核對 ------------------------------
LS, LL, ZS = P["v1-epoch-1@thr"], P["v2-epoch-2@thr"], P["v1-zero-shot@thr"]
TFT, TF5, GREEDY = P["tfidf@thr"], P["tfidf@0.5"], P["04"]
assert D["selected"]["v1"]["config"] == "epoch-1" and D["selected"]["v2"]["config"] == "epoch-2"
c_ll_zs, c_ls_zs = CMP["v2e2_thr_minus_zs_thr"], CMP["v1e1_thr_minus_zs_thr"]
c_ll_ls = CMP["v2e2_thr_minus_v1e1_thr"]
c_ll_tf, c_ls_tf, c_zs_tf = CMP["v2e2_thr_minus_tfidf_thr"], CMP["v1e1_thr_minus_tfidf_thr"], CMP["zs_thr_minus_tfidf_thr"]
c_ll_tf5 = CMP["v2e2_thr_minus_tfidf_05"]
c_ll_greedy, c_ll5_greedy = CMP["v2e2_thr_minus_04"], CMP["v2e2_05_minus_04"]
assert all(significant(c[k]) for c in (c_ll_zs, c_ls_zs) for k in ("auroc", "ap", "macro_f1"))
assert significant(c_zs_tf["auroc"]) and c_zs_tf["auroc"]["diff"] < 0
assert all(not significant(c[k]) for c in (c_ll_tf, c_ls_tf, c_ll_ls, c_ll_tf5) for k in ("auroc", "ap", "macro_f1"))
assert significant(c_ll_greedy["system_recall"]) and significant(c_ll_greedy["system_precision"]) and not significant(c_ll_greedy["macro_f1"])
assert all(not significant(c_ll5_greedy[k]) for k in ("macro_f1", "system_recall", "system_precision"))
HS, HL = MS["history"], ML["history"]
SS, SL = [h["global_step"] for h in HS], [h["global_step"] for h in HL]
assert SL[-1] < SS[0] and HS[1]["val_loss"] > HS[0]["val_loss"] * 1.2
lora_keys = ["v1-epoch-1", "v1-epoch-2", "v1-epoch-3", "v2-epoch-1", "v2-epoch-2"]
aurocs = [P[k + "@thr"]["auroc"] for k in lora_keys]
OC = D["overconfidence"]
g10, g20, g30, g50 = (G["v2-epoch-2"][k] for k in ("0.10", "0.20", "0.30", "0.50"))
RN = D["review_needed"]
FI, CV = IN["fields"], IN["cross_validation"]
FD, FDC = FI["directive"], FI["directive（去掉流程用語）"]
assert all(item["minus_description"][k]["ci"][0] > 0 for item in (FD, FDC) for k in ("auroc", "macro_f1"))
ND, CUE, RC = IN["near_duplicates"]["0.80"], IN["directive_cues"], IN["reason_coverage"]
assert len(IN["same_text"]["conflicting"]) == 1
ratio = ML["lora"]["trainable_params"] / MS["lora"]["trainable_params"]
low = [c for c in D["all3_cases"] if c["p_v2e2"] < 0.05]
ceil = {k: v["test_best_macro_f1"] for k, v in D["ceiling"].items()}
sens = {round(r["threshold"], 3): r["test"] for r in D["sensitivity_v2e2"]}
band = [v["macro_f1"] for t, v in sens.items() if 0.05 <= t <= 0.3]
dg = D["diagnostics"]
mass = [dg[run][name]["label_mass_mean"] for run, name in (("v1", "epoch-1"), ("v1", "epoch-2"), ("v1", "epoch-3"), ("v2", "epoch-1"), ("v2", "epoch-2"))]
assert all(dg[run][name]["top_is_label_rate"] == 1.0 for run in dg for name in dg[run])
SYSTEM = [r for r in EVERYTHING if r["idt_target"] == "系統"]
PERSON = [r for r in EVERYTHING if r["idt_target"] == "個人"]
assert len(EVERYTHING) == 3288 and len(SYSTEM) == 312
with_notes = sum(bool((r.get("directive") or "").strip()) for r in EVERYTHING)
N_SYS = sum(r["idt_target"] == "系統" for r in SETS["test"])
e = D["ensemble"]["v2e2+tfidf"]

# ============================== 第 3 章　研究方法 ==============================
h1("3　研究方法")
h2("3.1　研究設計")
para(
    "本研究為回溯性的預測模型開發與驗證研究，目的在於檢驗：僅依據醫療異常事件通報當下的事件描述，能否判定該事件在 "
    "Incident Decision Tree（IDT）架構下的歸因類別（「個人」或「系統」）。我們以低秩調適（low-rank adaptation, LoRA）微調"
    "一個開放權重的大型語言模型，並與未經微調的同一模型，以及以字元 n-gram 為特徵的線性分類器比較。所有模型都輸出事件屬於"
    "「系統」的機率；模型的檢查點（checkpoint）與決策門檻只由驗證集決定，測試集不參與任何選擇。除了模型之間的比較，"
    "我們另外設計資訊來源分析，檢驗模型表現是否受限於事件描述所含的資訊。整體流程如圖 1。"
)
figure(
    1,
    "研究架構。事件通報經分層、分組切分為訓練集、驗證集與測試集；微調的語言模型（LoRA-S、LoRA-L）與兩個對照模型都輸出 "
    "P(系統)；模型檢查點與決策門檻由驗證集決定，測試集用於最終評估與資訊來源分析。",
    width_emu=5699846,
)

h2("3.2　資料來源與標註")
para(
    "資料取自【醫療機構名稱】病人安全通報系統於【通報期間】收錄的醫療異常事件通報。每筆通報包含三項本研究使用的內容。"
    "其一為事件描述，由通報者在通報當下以自由文字撰寫，是本研究所有分類模型的唯一輸入。其二為調查處理紀錄，是事件會辦、"
    "調查與處理過程中由相關單位陸續填寫的文字，常包含原因說明與改善措施；這項內容在通報當下尚不存在，僅用於 3.11 節的"
    "資訊來源分析。其三為 IDT 歸因標籤，由【標註者及其資歷】依 IDT 的判定流程標記為「個人」或「系統」，作為參考標準。"
)
para(
    "IDT 以一系列依序進行的檢驗區分個人責任與系統因素，包括行為是否蓄意、是否受健康或能力因素影響、是否偏離既有規範"
    "以及規範本身是否適當可行、其他具備相同資格的人員在相同情境下是否會有相同行為（替代測試），以及教育訓練與督導是否有"
    f"缺失（Meadows 等，2005）。本研究納入具有有效歸因標籤的通報共 {n(len(EVERYTHING))} 筆，其中「個人」{n(len(PERSON))} 筆"
    f"（{pct(len(PERSON) / len(EVERYTHING))}）、「系統」{len(SYSTEM)} 筆（{pct(len(SYSTEM) / len(EVERYTHING))}）"
    "【原始通報總數與納入、排除流程】。【倫理審查核准字號與去識別化程序】"
)

h2("3.3　資料切分")
para(
    "通報以 7：1：2 的比例隨機切分為訓練集、驗證集與測試集，並依歸因標籤分層，使三個子集的類別比例一致。為避免相同文字"
    "同時出現在訓練與評估資料而高估表現，事件描述完全相同的通報視為同一組，整組分配到同一個子集；資料中共有 "
    f"{D['duplicates']['groups']} 組、{D['duplicates']['records']} 筆這樣的通報，切分後沒有任何一組跨越子集。"
    "驗證集與測試集維持原始資料，不做任何形式的資料增生。"
)

h2("3.4　任務定義")
para(
    "令第 $i$ 筆通報的事件描述為 $x_i$，歸因標籤為 $y_i ∈ {0, 1}$，其中 1 代表「系統」、0 代表「個人」。分類模型輸出該通報屬於"
    "「系統」的機率 $s_i = P(y_i = 1 | x_i)$，並在給定決策門檻 τ 時作出判定："
)
equation(m_sub(m_hat(m_r("y")), m_r("i")), m_r(" = I("), m_sub(m_r("s"), m_r("i")), m_r(" ≥ τ)"))
para(
    "其中 $I(·)$ 為指示函數。由於「系統」事件僅佔約一成，而漏判系統因素與誤報的代價並不對等，本研究同時評估不依賴門檻的"
    "排序能力與特定門檻下的分類表現（3.9 節）。",
    indent=False,
)

h2("3.5　語言模型與微調")
tok = ML["data"]["token_lengths"]["train"]
para(
    "基礎模型為 Qwen3.5-9B（約 94 億個參數）【模型出處與授權】，其 32 層解碼層中有 8 層採用標準自注意力、24 層採用線性注意力。模型的輸入是"
    "一則使用者訊息，依序包含任務說明、完整的 IDT 決策樹（十個依序判斷的問題及其判定規則）與事件描述，並要求模型只輸出"
    "「個人」或「系統」。輸入套用模型的對話樣板，並關閉其推理（thinking）模式。包含決策樹在內，原始訓練集的輸入序列"
    f"平均長度為 {round(tok['mean']):,} 個 token（最長 {n(tok['max'])}），皆未超過 {n(ML['training']['max_length'])} 個 token 的長度上限。"
)
para("微調採監督式學習。令答案序列（標籤與結束符號）為 $a_1, …, a_T$，訓練目標是只在答案位置上計算的平均交叉熵：")
equation(
    m_r("ℒ", plain=True), m_r("(θ) = −"), m_frac(m_r("1"), m_r("T")),
    m_sum(m_r("t = 1"), m_r("T"), m_r("log ", plain=True) + m_sub(m_r("p"), m_r("θ")) + m_r("(") + m_sub(m_r("a"), m_r("t")) + m_r(" | x, ") + m_sub(m_r("a"), m_r("<t")) + m_r(")")),
)
para(
    "其中 $x$ 為包含提示的輸入序列，$p_θ$ 為模型對下一個 token 的機率分布，$θ$ 為可訓練參數。我們以 LoRA（Hu 等，2022）進行"
    "參數高效微調：凍結預訓練權重 $W$，僅學習低秩增量，",
    indent=False,
)
equation(m_r("W′ = W + "), m_frac(m_r("α"), m_r("r")), m_r(" BA"))
para(
    "其中 $B$ 與 $A$ 為秩 $r$ 的可訓練矩陣，$α$ 為縮放係數。為了解調適範圍與訓練資料對表現的影響，我們比較兩種設定（表 1）。"
    f"LoRA-S 只調整 8 層標準自注意力中的 query 與 value 投影，可訓練參數佔全部參數的 "
    f"{MS['lora']['trainable_params'] / MS['lora']['total_params'] * 100:.3f}%；其訓練資料除原始訓練集外，另加入以較小的語言模型"
    "（Qwen3.5-2B，取樣溫度 0.8）為每筆事件描述產生的兩則改寫，改寫沿用原通報的標籤。LoRA-L 則調整全部 32 層中注意力、"
    f"線性注意力與前饋網路的所有線性投影，可訓練參數約為 LoRA-S 的 {ratio:.0f} 倍，並只使用原始訓練集。兩種設定在調適範圍、"
    "秩與訓練資料上同時不同，本研究將它們視為兩個整體設定加以比較，不分別估計各項因素的影響。兩種設定都在每個 epoch "
    "結束時保存檢查點，供 3.8 節的模型選擇使用。",
    indent=False,
)
table(
    1,
    "兩種 LoRA 微調設定",
    ["項目", "LoRA-S", "LoRA-L"],
    [
        ["調適的模組", f"8 層標準自注意力的 query、value 投影（{MS['lora']['matched_modules']} 個模組）",
         f"全部 32 層的注意力、線性注意力與前饋網路之線性投影（{ML['lora']['matched_modules']} 個模組）"],
        ["秩 $r$／縮放係數 $α$", f"{MS['training']['lora_rank']}／{MS['training']['lora_alpha']}", f"{ML['training']['lora_rank']}／{ML['training']['lora_alpha']}"],
        ["可訓練參數", f"{n(MS['lora']['trainable_params'])}（{MS['lora']['trainable_params'] / MS['lora']['total_params'] * 100:.3f}%）",
         f"{n(ML['lora']['trainable_params'])}（{ML['lora']['trainable_params'] / ML['lora']['total_params'] * 100:.2f}%）"],
        ["訓練資料", f"{n(MS['data']['train_records'])} 筆（原始 2,302 筆＋改寫 4,604 筆）", f"{n(ML['data']['train_records'])} 筆（原始訓練集）"],
        ["訓練 epoch／更新步數", f"{MS['training']['epochs']}／{n(SS[-1])}", f"{ML['training']['epochs']}／{SL[-1]}"],
        ["共同設定", "AdamW（無權重衰減），學習率 $2×10^{−4}$，前 3% 步數線性暖身後線性衰減；批次大小 1、梯度累積 8 步；梯度裁剪 1.0；LoRA dropout 0.05；bfloat16；亂數種子 42", "同左"],
    ],
    [2000, 3500, 3500],
    wrap=(1, 2),
    note="百分比為可訓練參數佔模型全部參數的比例。",
)

h2("3.6　機率評分")
para(
    "生成式語言模型常見的作法是讓模型生成標籤文字（貪婪解碼）。這種作法只提供判定結果，等同將決策門檻固定，無法評估模型的"
    "排序能力，也無法依驗證資料選擇檢查點或調整門檻。本研究改為直接讀取模型在答案位置的下一個 token 分布：令 $z_{sys}$ 與 $z_{ind}$ "
    "分別為「系統」與「個人」兩個標籤第一個 token 的 logit（兩個標籤的第一個 token 不同），定義"
)
z_sys, z_ind = m_sub(m_r("z"), m_r("sys", plain=True)), m_sub(m_r("z"), m_r("ind", plain=True))
equation(
    m_r("s = "),
    m_frac(m_r("exp(", plain=True) + z_sys + m_r(")"), m_r("exp(", plain=True) + z_sys + m_r(") + ") + m_r("exp(", plain=True) + z_ind + m_r(")")),
    m_r(" = σ("), z_sys, m_r(" − "), z_ind, m_r(")"),
)
para(
    "其中 $σ$ 為 logistic 函數。這個分數是在兩個標籤之間正規化的機率；當兩個標籤佔據絕大部分的機率質量時，以 0.5 為門檻的判定"
    f"與貪婪解碼一致。在本研究的資料中，兩個標籤的機率合計平均為 {dg['v1']['zero-shot']['label_mass_mean']:.3f}（未微調）與 "
    f"{min(mass):.4f}–{max(mass):.4f}（微調後），且機率最高的 token 均為兩個標籤之一。",
    indent=False,
)
para(
    "模型以 bfloat16 精度運算，logit 只取離散的數值，因此許多通報的分數在數值上相同。為避免浮點誤差使相同的分數被視為不同，"
    "評估時將 logit 相差小於 $10^{−4}$ 的分數視為同分；同分的通報在計算排序指標與決定門檻時一律同等處理。"
)

h2("3.7　比較模型")
para(
    "我們設置兩個對照模型。第一個是未微調的基礎模型（以下稱 zero-shot），使用與微調相同的提示與機率評分方式，用以衡量微調的"
    "貢獻。第二個是詞彙式基準模型（以下稱 TF-IDF＋LR）：以字元 1 至 3 元組的 TF-IDF（最小文件頻率 2、次線性詞頻）為特徵，"
    "訓練 L2 正則化的邏輯迴歸（C = 10，類別權重與類別頻率成反比）（Salton 與 Buckley，1988；Pedregosa 等，2011），訓練資料為"
    "原始訓練集，輸入同樣只有事件描述。此外，為了與生成式的慣用流程對照，我們另報告以 LoRA-S 最後一個 epoch 的檢查點進行"
    "貪婪解碼的結果（以下稱生成式判定）。"
)

h2("3.8　模型選擇與決策門檻")
para(
    "模型選擇只使用驗證集。對每一種微調設定，選擇驗證集 log-loss 最低的檢查點；對每一個模型（包含兩個對照模型），以驗證集 "
    "macro-F1 最高者作為決策門檻 τ*，候選門檻為相鄰相異分數的中點，數個門檻的 macro-F1 相同時取最接近 0.5 者。為了與生成式判定"
    "比較，各模型在固定門檻 0.5 的結果也一併報告。測試集不參與任何檢查點或門檻的選擇。為了解門檻對結果的影響，我們另外報告"
    "測試集表現隨門檻的變化，以及直接以測試集選擇門檻時 macro-F1 所能達到的數值；後者只作為參考上限，不用於任何選擇。"
)

h2("3.9　評估指標")
para(
    "排序能力以接收者操作特徵曲線下面積（AUROC）與平均精確率（average precision, AP）衡量；後者在類別不平衡時對少數類別"
    "較為敏感（Saito 與 Rehmsmeier，2015）。機率的品質以 log-loss 衡量："
)
equation(
    m_r("LogLoss", plain=True), m_r(" = −"), m_frac(m_r("1"), m_r("N")),
    m_sum(m_r("i = 1"), m_r("N"), m_r("[") + m_sub(m_r("y"), m_r("i")) + m_r("log ", plain=True) + m_sub(m_r("s"), m_r("i")) + m_r(" + (1 − ") + m_sub(m_r("y"), m_r("i")) + m_r(") ") + m_r("log", plain=True) + m_r("(1 − ") + m_sub(m_r("s"), m_r("i")) + m_r(")]")),
)
para(
    "其中 $N$ 為通報數；log-loss 愈低，表示機率愈接近實際的標籤。我們另以 P(系統) 低於 0.01 的「系統」事件數，描述模型以"
    "高信心誤判的程度。給定門檻下的分類表現，以「系統」類別的精確率（precision, $P$）、召回率（recall, $R$）與 F1，以及兩個"
    "類別 F1 的平均（macro-F1）衡量：",
    indent=False,
)
f1_c = m_sub(m_r("F1", plain=True), m_r("c"))
equation(
    f1_c, m_r(" = "), m_frac(m_r("2") + m_sub(m_r("P"), m_r("c")) + m_sub(m_r("R"), m_r("c")), m_sub(m_r("P"), m_r("c")) + m_r(" + ") + m_sub(m_r("R"), m_r("c"))),
    m_r("，　"), m_r("macro-F1", plain=True), m_r(" = "),
    m_frac(m_sub(m_r("F1", plain=True), m_r("sys", plain=True)) + m_r(" + ") + m_sub(m_r("F1", plain=True), m_r("ind", plain=True)), m_r("2")),
)
para(
    "其中 $c$ 表示類別。由於多數類別佔九成，準確率（accuracy）僅列為參考。最後，為評估模型作為篩檢工具的效益，我們計算"
    "累積增益：將通報依 $s$ 由高到低排序，取前 $k$ 比例的通報集合 $T_k$，",
    indent=False,
)
equation(
    m_r("G(k) = "), m_frac(m_r("1"), m_sub(m_r("N"), m_r("sys", plain=True))),
    m_sum(m_r("i ∈ ") + m_sub(m_r("T"), m_r("k")), None, m_sub(m_r("y"), m_r("i"))),
)
para(
    "即前 $k$ 比例的通報所涵蓋的「系統」事件比例，其中 $N_{sys}$ 為「系統」事件的總數；分數相同而跨越界線的通報按比例計入。"
    "我們並報告涵蓋 80% 與 90% 的「系統」事件所需檢視的通報比例。",
    indent=False,
)

h2("3.10　統計分析")
para(
    "所有指標的不確定性以無母數拔靴法（bootstrap）估計（Efron 與 Tibshirani，1993）：對測試集的通報重複抽樣 2,000 次"
    "（抽後放回），以百分位法計算 95% 信賴區間。兩個模型的比較採配對設計，在相同的重抽樣樣本上計算兩者指標之差，差值的 "
    "95% 信賴區間不包含 0 視為達統計顯著。重抽樣時模型的檢查點與決策門檻維持不變，因此信賴區間反映的是測試樣本的抽樣變異，"
    "不包含模型訓練與門檻選擇的變異。本研究以選定的微調模型與 TF-IDF＋LR 在測試集 AUROC 上的差異為主要比較；其餘比較為"
    "次要分析，未做多重比較校正。下文以方括號表示 95% 信賴區間。"
)

h2("3.11　資訊來源分析")
para(
    "為檢驗分類表現是否受限於事件描述所含的資訊，我們進行三項分析。第一，輸入替換：以相同的 TF-IDF＋LR 模型與相同的資料切分，"
    "分別以事件描述、調查處理紀錄、兩者串接，以及去除流程用語（會辦與品質改善流程的固定用語及會辦單位名稱）後的調查處理紀錄"
    "作為輸入，比較測試集的表現；另以全部資料進行分組五折交叉驗證（相同描述同組，三個亂數種子）確認結果的穩定性。使用同一個"
    "模型而只改變輸入，是為了將資訊來源的影響與模型能力的影響分開。第二，標註一致性：以字元 2 至 3 元組的 TF-IDF 餘弦相似度"
    "找出事件描述高度相似（相似度 ≥ 0.8）的通報配對，計算其中標籤相同的比例，用以評估標註雜訊是否構成表現的上限。第三，"
    "錯誤重疊：比較各模型在測試集漏判的「系統」事件，檢視不同模型的錯誤是否集中在相同的通報；並以 Spearman 等級相關衡量"
    "不同模型分數排序的一致程度，以及將語言模型與 TF-IDF＋LR 的 P(系統) 取算術平均後的 AUROC，檢視兩者的訊息是否互補。"
    "此外，我們統計調查處理紀錄中載有「原因說明」段落的通報比例，用以描述調查後的文字所含的歸因資訊。"
)

h2("3.12　實作環境")
para(
    "模型微調在 NVIDIA DGX Spark（GB10 Grace Blackwell 系統單晶片，20 核心 Arm 處理器與 128 GB 統一記憶體）上進行，使用 "
    "PyTorch 2.15、Transformers 5.24 與 PEFT 0.22，並啟用梯度檢查點以降低記憶體用量。LoRA-L 的訓練共 "
    f"{SL[-1]} 個更新步，耗時 {ML['duration_minutes'] / 60:.1f} 小時；LoRA-S 共 {n(SS[-1])} 個更新步，耗時 "
    f"{MS['duration_minutes'] / 60:.1f} 小時【請確認 LoRA-S 的訓練硬體】。機率評分在 Apple Silicon 處理器上以相同的 bfloat16 "
    "精度執行。TF-IDF＋LR 以 "
    "scikit-learn 1.9 實作。所有隨機程序均固定亂數種子（42）。【程式碼與資料的取得方式】"
)

# ============================== 第 4 章　實驗結果 ==============================
h1("4　實驗結果")
h2("4.1　資料集特性")
share = [sum(r["idt_target"] == "系統" for r in SETS[name]) / len(SETS[name]) for name in ("train", "val", "test")]
assert length_quartiles(SYSTEM)[1] > length_quartiles(PERSON)[1]
assert length_quartiles(SYSTEM, "directive")[1] > length_quartiles(PERSON, "directive")[1]
para(
    f"表 2 彙整資料集的特性。三個子集的「系統」事件比例一致（{pct(min(share))}–{pct(max(share))}）。事件描述長度的中位數為 "
    f"{length_text(EVERYTHING)}；「系統」事件的描述較長，中位數為 {length_text(SYSTEM)}，「個人」事件為 {length_text(PERSON)}。"
    f"有調查處理紀錄的通報佔 {pct(with_notes / len(EVERYTHING))}；「系統」事件的調查處理紀錄同樣較長，中位數為 "
    f"{length_text(SYSTEM, 'directive')}，「個人」事件為 {length_text(PERSON, 'directive')}。"
)


def data_row(name, records, show_system=True):
    system = sum(r["idt_target"] == "系統" for r in records)
    return [name, n(len(records)), f"{system}（{pct(system / len(records))}）" if show_system else "—",
            length_summary(records), length_summary(records, "directive")]


table(
    2,
    "資料集特性",
    ["子集／類別", "通報數", "「系統」筆數（%）", "事件描述長度（字）", "調查處理紀錄長度（字）"],
    [
        data_row("訓練集", SETS["train"]), data_row("驗證集", SETS["val"]), data_row("測試集", SETS["test"]),
        data_row("全部", EVERYTHING), data_row("「個人」事件", PERSON, False), data_row("「系統」事件", SYSTEM, False),
    ],
    [1800, 1100, 1900, 1900, 2300],
    ["l", "r", "r", "r", "r"],
    note="長度以字元數計，表示為中位數（第 1–第 3 四分位數）；調查處理紀錄的長度只計入有該項紀錄的通報。",
    rule_before=(3, 4),
)

h2("4.2　微調對分辨能力的影響")
MODELS = [
    ("Zero-shot", "—", "v1-zero-shot"), ("LoRA-S", "epoch 1†", "v1-epoch-1"), ("LoRA-S", "epoch 2", "v1-epoch-2"),
    ("LoRA-S", "epoch 3", "v1-epoch-3"), ("LoRA-L", "epoch 1", "v2-epoch-1"), ("LoRA-L", "epoch 2†", "v2-epoch-2"),
    ("TF-IDF＋LR", "—", "tfidf"),
]
vl = [CFG[k]["val"]["log_loss"] for k in ("v1-epoch-1", "v1-epoch-2", "v1-epoch-3")]
va = [CFG[k]["val"]["auroc"] for k in ("v1-epoch-1", "v1-epoch-2", "v1-epoch-3")]
oc_s = [OC[f"v1-epoch-{i}/test"]["pos_lt_0.01"] for i in (1, 2, 3)]
oc_l = [OC[f"v2-epoch-{i}/test"]["pos_lt_0.01"] for i in (1, 2)]
assert vl[0] < vl[1] < vl[2] and oc_s[0] < oc_s[1] < oc_s[2] and max(va) == va[0]
assert min(CFG[k]["test"]["log_loss"] for k in ("v1-epoch-1", "v1-epoch-2", "v1-epoch-3")) == CFG["v1-epoch-1"]["test"]["log_loss"]
para(
    f"表 3 與圖 2 呈現各模型在測試集的排序能力與機率品質。未微調的 zero-shot 模型 AUROC 為 {f3(ZS['auroc'])} "
    f"{ci3(ZS['ci']['auroc'])}；微調後，五個檢查點的 AUROC 介於 {min(aurocs):.3f}–{max(aurocs):.3f}。依驗證集 log-loss "
    "選出的檢查點為 LoRA-S 的第 1 個 epoch 與 LoRA-L 的第 2 個 epoch。相對於 zero-shot，兩者的 AUROC 分別提高 "
    f"{increase(c_ls_zs['auroc'])} 與 {increase(c_ll_zs['auroc'])}，AP 分別提高 {increase(c_ls_zs['ap'])} 與 "
    f"{increase(c_ll_zs['ap'])}（表 4）。相對地，僅在提示中提供完整決策樹而未經微調的模型，其 AUROC 顯著低於 TF-IDF＋LR"
    f"（差異 {dci(c_zs_tf['auroc'])}）。"
)
para(
    f"LoRA-S 的三個檢查點顯示，延長訓練並未再提升排序能力（驗證集 AUROC 介於 {min(va):.3f}–{max(va):.3f}），機率品質則逐步惡化："
    f"驗證集 log-loss 由 {f4(vl[0])} 上升至 {f4(vl[1])} 與 {f4(vl[2])}，測試集 {N_SYS} 筆「系統」事件中 P(系統) 低於 0.01 者由 "
    f"{oc_s[0]} 筆增加為 {oc_s[1]} 筆與 {oc_s[2]} 筆，顯示模型趨於過度自信。以驗證集 log-loss 選出的第 1 個 epoch，同時也是"
    f"測試集 log-loss 最低的檢查點。LoRA-L 兩個檢查點的驗證集 log-loss 相近（{f4(CFG['v2-epoch-1']['val']['log_loss'])} 與 "
    f"{f4(CFG['v2-epoch-2']['val']['log_loss'])}），P(系統) 低於 0.01 的「系統」事件分別為 {oc_l[0]} 筆與 {oc_l[1]} 筆。"
)
table(
    3,
    "各模型在測試集的排序能力與機率品質",
    ["模型", "檢查點", "驗證集\nlog-loss", "測試集\nlog-loss", "AUROC（95% CI）", "AP（95% CI）"],
    [
        [name, epoch, f4(CFG[key]["val"]["log_loss"]), f4(CFG[key]["test"]["log_loss"]),
         f"{f3(P[key + '@thr']['auroc'])} {ci3(P[key + '@thr']['ci']['auroc'])}", f"{f3(P[key + '@thr']['ap'])} {ci3(P[key + '@thr']['ci']['ap'])}"]
        for name, epoch, key in MODELS
    ],
    [1400, 1100, 1300, 1300, 1950, 1950],
    ["l", "l", "r", "r", "r", "r"],
    note="† 依驗證集 log-loss 選定的檢查點。AP：average precision；CI：信賴區間（bootstrap 2,000 次）。",
    rule_before=(1, 4, 6),
)
figure(
    2,
    "各模型在測試集的 AUROC（左）與 macro-F1（右）及其 95% 信賴區間。macro-F1 以各模型在驗證集決定的門檻 τ* 計算；"
    "垂直線為 TF-IDF＋LR 的數值。",
)


def cell(item):
    """表 4 的一格：上列點估計、下列信賴區間；區間不含 0 時粗體。"""
    text = f"{signed(item['diff'])}\n[{signed(item['ci'][0])}, {signed(item['ci'][1])}]"
    return f"**{text}**" if significant(item) else text


def diff_row(name, comparison):
    return [name, cell(comparison["auroc"]), cell(comparison["ap"]), cell(comparison["macro_f1"]), cell(comparison["system_recall"])]


table(
    4,
    "模型之間的配對差異（測試集）",
    ["比較（前者 − 後者）", "ΔAUROC", "ΔAP", "Δmacro-F1", "Δ召回率"],
    [
        diff_row("LoRA-S − Zero-shot", c_ls_zs), diff_row("LoRA-L − Zero-shot", c_ll_zs), diff_row("LoRA-L − LoRA-S", c_ll_ls),
        diff_row("LoRA-S − TF-IDF＋LR", c_ls_tf), diff_row("LoRA-L − TF-IDF＋LR", c_ll_tf), diff_row("Zero-shot − TF-IDF＋LR", c_zs_tf),
    ],
    [2400, 1650, 1650, 1650, 1650],
    ["l", "c", "c", "c", "c"],
    note="每格上列為差異的點估計，下列為 95% 配對 bootstrap 信賴區間；粗體表示區間不包含 0。LoRA-S 與 LoRA-L 為選定的檢查點；"
    "macro-F1 與「系統」召回率以各模型的 τ* 計算。",
    rule_before=(3,),
)

h2("4.3　與詞彙式基準模型的比較")
bound = max(abs(v) for c in (c_ll_tf, c_ls_tf) for v in c["auroc"]["ci"])
flag_ls, flag_tf = LS["tp"] + LS["fp"], TFT["tp"] + TFT["fp"]
ensemble_gap = {"diff": e["diff_vs_v2e2"], "ci": e["ci"]}
assert bound < 0.03 and TF5["macro_f1"] > TFT["macro_f1"] and not significant(ensemble_gap)
assert significant(c_ls_tf["system_recall"]) and not significant(c_ll_tf["system_recall"]) and flag_ls > flag_tf
para(
    "在主要比較上，選定的微調模型並未優於 TF-IDF＋LR。LoRA-L 與 TF-IDF＋LR 的 AUROC 分別為 "
    f"{f3(LL['auroc'])} 與 {f3(TFT['auroc'])}，差異為 {dci(c_ll_tf['auroc'])}；LoRA-S 與 TF-IDF＋LR 的差異為 "
    f"{dci(c_ls_tf['auroc'])}。兩個信賴區間都落在 $±0.03$ 之內，顯示兩類模型的排序能力即使有差距也相當有限。AP 與 macro-F1 "
    f"的差異同樣未達統計顯著（LoRA-L：AP {dci(c_ll_tf['ap'])}，macro-F1 {dci(c_ll_tf['macro_f1'])}；表 4）。TF-IDF＋LR "
    "在固定門檻 0.5 的測試集 macro-F1 高於 τ* 的結果（表 5）；改以前者為對照時，結論不變（LoRA-L 的 macro-F1 差異 "
    f"{dci(c_ll_tf5['macro_f1'])}）。"
)
para(
    "表 4 中，微調模型與 TF-IDF＋LR 之間唯一達統計顯著的差異，是 LoRA-S 在 τ* 下較高的召回率"
    f"（{dci(c_ls_tf['system_recall'])}）。LoRA-S 在該門檻判為「系統」的通報也較多（{flag_ls} 筆，TF-IDF＋LR 為 {flag_tf} 筆），"
    "這項差異反映的是操作點的不同；如前所述，兩者的排序能力並無可辨別的差距。"
)
para(
    f"LoRA-L 與 TF-IDF＋LR 的分數呈正相關（Spearman 相關係數 {D['spearman']['v2-epoch-2|tfidf']:.2f}）。將兩者的 P(系統) "
    f"取平均後，測試集 AUROC 為 {f3(e['test_auroc'])}，相對於 LoRA-L 的差異 {dci(ensemble_gap)} 未達統計顯著。"
)

h2("4.4　調適範圍、訓練資料與訓練動態")
first_val = [HL[0]["val_loss"], HL[1]["val_loss"], HS[0]["val_loss"]]
assert max(first_val) - min(first_val) < 0.001 and HS[0]["train_loss"] > HS[1]["train_loss"] > HS[2]["train_loss"]
para(
    f"LoRA-L 的可訓練參數約為 LoRA-S 的 {ratio:.0f} 倍，且未使用改寫資料；兩者選定的檢查點在測試集的表現相當："
    f"AUROC 差異 {dci(c_ll_ls['auroc'])}，macro-F1 差異 {dci(c_ll_ls['macro_f1'])}（表 4）。兩個模型對 {len(SETS['test'])} "
    f"筆測試通報的分數排序高度一致（Spearman 相關係數 {D['spearman']['v2-epoch-2|v1-epoch-1']:.2f}）。就訓練量而言，LoRA-L 經過 "
    f"{SL[-1]} 個更新步（{n(ML['data']['train_records'] * len(HL))} 筆次）即達到與 LoRA-S 第一個檢查點（{SS[0]} 個更新步，"
    f"{n(MS['data']['train_records'])} 筆次）相當的表現，且不需事先產生改寫資料。"
)
para(
    "圖 3 以累計更新步數對齊兩種設定的訓練過程。圖中的損失為式 (2) 的數值，即每個答案 token 的平均交叉熵，與表 3 以標籤機率"
    f"計算的 log-loss 尺度不同。LoRA-L 的兩個檢查點（{SL[0]} 與 {SL[1]} 步）與 LoRA-S 的第一個檢查點（{SS[0]} 步）驗證損失相近"
    f"（{min(first_val):.3f}–{max(first_val):.3f}）；LoRA-S 在 {n(SS[1])} 與 {n(SS[2])} 步時上升至 {f3(HS[1]['val_loss'])} 與 "
    f"{f3(HS[2]['val_loss'])}，同時訓練損失持續下降，呈現過度擬合的型態。LoRA-L 的總訓練步數少於 LoRA-S 的第一個 epoch，"
    "因此本研究的資料無法判斷 LoRA-L 在更長的訓練下是否會出現相同的現象。"
)
figure(
    3,
    "兩種微調設定的驗證損失（左）與訓練損失（右）。橫軸為累計更新步數，每個點代表一個 epoch 結束。訓練損失為各 epoch "
    "訓練過程中的平均值，驗證損失於 epoch 結束時計算，兩者不宜直接相減。",
)

h2("4.5　決策門檻與分類表現")
agree = CFG["v1-epoch-3"]["agreement_with_reference"]
LS5, LL5 = P["v1-epoch-1@0.5"], P["v2-epoch-2@0.5"]
person_test = sum(r["idt_target"] == "個人" for r in SETS["test"])
assert agree["compared"] == len(SETS["test"]) and agree["matched"] >= agree["compared"] - 3
assert c_ll_greedy["system_recall"]["diff"] > 0 > c_ll_greedy["system_precision"]["diff"]
assert max(LS5["system_recall"], LL5["system_recall"]) < GREEDY["system_recall"] < min(LS["system_recall"], LL["system_recall"])
assert LL["accuracy"] < person_test / len(SETS["test"])
para(
    f"表 5 列出各模型在固定門檻 0.5 與驗證集門檻 τ* 下的分類表現。生成式判定的 macro-F1 為 {f3(GREEDY['macro_f1'])}，"
    f"「系統」召回率為 {pct(GREEDY['system_recall'])}、精確率為 {pct(GREEDY['system_precision'])}。以同一個檢查點比較時，"
    f"機率評分在門檻 0.5 的判定與生成式判定在 {agree['compared']} 筆測試通報中有 {agree['matched']} 筆相同，"
    "可見生成式判定相當於將門檻固定於 0.5。"
)
para(
    f"選定的 LoRA-L 在 τ* = {LL['threshold']:.3f} 時，召回率為 {pct(LL['system_recall'])}（{N_SYS} 筆中找出 {LL['tp']} 筆），"
    f"精確率為 {pct(LL['system_precision'])}（誤報 {LL['fp']} 筆），macro-F1 為 {f3(LL['macro_f1'])} {ci3(LL['ci']['macro_f1'])}。"
    f"與生成式判定相比，召回率提高 {pp_change(c_ll_greedy['system_recall'])}，精確率降低 "
    f"{pp_change(c_ll_greedy['system_precision'])}，macro-F1 的差異 {dci(c_ll_greedy['macro_f1'])} 未達統計顯著。"
    f"召回率的提升來自決策門檻：在門檻 0.5 時，LoRA-L 與 LoRA-S 的召回率分別為 {pct(LL5['system_recall'])} 與 "
    f"{pct(LS5['system_recall'])}，並未高於生成式判定（LoRA-L 與生成式判定的 macro-F1 差異 {dci(c_ll5_greedy['macro_f1'])}）；"
    f"門檻降至 τ* 後才提高至 {pct(LL['system_recall'])} 與 {pct(LS['system_recall'])}。"
)


def class_row(name, rule, point):
    m = P[point]
    return [name, rule, f"{f3(m['macro_f1'])} {ci3(m['ci']['macro_f1'])}", f3(m["system_precision"]), f3(m["system_recall"]),
            f3(m["system_f1"]), f3(m["accuracy"]), f"{m['tp']}／{m['fp']}"]


rows5 = [class_row("生成式判定", "貪婪解碼", "04")]
for name, key in (("Zero-shot", "v1-zero-shot"), ("LoRA-S（epoch 1）", "v1-epoch-1"), ("LoRA-L（epoch 2）", "v2-epoch-2"), ("TF-IDF＋LR", "tfidf")):
    rows5.append(class_row(name, "τ = 0.5", f"{key}@0.5"))
    rows5.append(class_row("", f"τ* = {P[key + '@thr']['threshold']:.3f}", f"{key}@thr"))
table(
    5,
    "各模型在測試集的分類表現",
    ["模型", "決策規則", "macro-F1（95% CI）", "精確率", "召回率", "F1", "準確率", "TP／FP"],
    rows5,
    [1800, 1250, 1950, 800, 800, 650, 800, 950],
    ["l", "l", "r", "r", "r", "r", "r", "r"],
    note="精確率、召回率與 F1 為「系統」類別的數值。生成式判定：以 LoRA-S 最後一個 epoch 的檢查點進行貪婪解碼。τ*：驗證集 "
    f"macro-F1 最高的門檻。TP／FP：判為「系統」且正確／錯誤的通報數（測試集共有 {N_SYS} 筆「系統」事件）。",
    rule_before=(1, 3, 5, 7),
)
para(
    f"門檻在相當大的範圍內對 macro-F1 的影響有限：LoRA-L 的門檻由 0.05 調整至 0.30 時，測試集 macro-F1 介於 {min(band):.3f}–"
    f"{max(band):.3f}，但找出的「系統」事件由 {sens[0.05]['tp']} 筆降為 {sens[0.3]['tp']} 筆，誤報由 {sens[0.05]['fp']} 筆降為 "
    f"{sens[0.3]['fp']} 筆；門檻的選擇主要決定召回率與誤報之間的取捨。即使直接以測試集選擇門檻，五個微調檢查點的 macro-F1 "
    f"上限也僅為 {min(ceil[k] for k in lora_keys):.3f}–{max(ceil[k] for k in lora_keys):.3f}。由於類別不平衡，LoRA-L 在 τ* "
    f"的準確率（{pct(LL['accuracy'])}）低於將所有通報判為「個人」時的 {pct(person_test / len(SETS['test']))}。此外，驗證集僅有 "
    f"{sum(r['idt_target'] == '系統' for r in SETS['val'])} 筆「系統」事件，所選門檻帶有抽樣誤差：LoRA-L 的 macro-F1 在驗證集為 "
    f"{CFG['v2-epoch-2']['threshold']['val_macro_f1']:.3f}，在測試集為 {f3(LL['macro_f1'])}；TF-IDF＋LR 的 τ* 在測試集的 "
    f"macro-F1（{f3(TFT['macro_f1'])}）低於門檻 0.5 的結果（{f3(TF5['macro_f1'])}）。"
)

h2("4.6　作為篩檢工具的效益")
remaining = N_SYS - g50["captured"]
assert abs(remaining - round(remaining)) < 1e-9
para(
    f"依 P(系統) 由高到低排序時，「系統」事件集中於排序的前段（表 6、圖 4）。以 LoRA-L 而言，前 20% 的通報（{g20['reviewed']} 筆）"
    f"涵蓋 {pc(g20['recall'])} {pcci(g20['ci'])} 的「系統」事件，為隨機排序的 {g20['recall'] / 0.2:.1f} 倍；前 30% 涵蓋 "
    f"{pc(g30['recall'])}，前 50% 涵蓋 {pc(g50['recall'])}，排序後半的通報中僅餘 {round(remaining)} 筆「系統」事件。"
    f"要涵蓋 80% 與 90% 的「系統」事件，分別需要檢視 {RN['v2-epoch-2']['0.80'] * 100:.0f}% 與 "
    f"{RN['v2-epoch-2']['0.90'] * 100:.0f}% 的通報；zero-shot 模型則需要 {RN['v1-zero-shot']['0.80'] * 100:.0f}% 與 "
    f"{RN['v1-zero-shot']['0.90'] * 100:.0f}%。TF-IDF＋LR 的累積增益與 LoRA-L 相近（前 20% 涵蓋 "
    f"{pc(G['tfidf']['0.20']['recall'])} {pcci(G['tfidf']['0.20']['ci'])}）。"
)
rows6 = []
for key in ("0.10", "0.20", "0.30", "0.50"):
    a, b, z = G["v2-epoch-2"][key], G["tfidf"][key], G["v1-zero-shot"][key]
    share_text = f"{float(key) * 100:.0f}%"
    rows6.append([f"前 {share_text}", str(a["reviewed"]), f"{pc(a['recall'])} {pcci(a['ci'])}", f"{pc(b['recall'])} {pcci(b['ci'])}", pc(z["recall"]), share_text])
table(
    6,
    "累積增益：依 P(系統) 排序後，前 $k$ 比例的通報所涵蓋的「系統」事件比例（測試集）",
    ["納入的通報", "通報數", "LoRA-L（95% CI）", "TF-IDF＋LR（95% CI）", "Zero-shot", "隨機排序"],
    rows6,
    [1300, 1000, 2250, 2250, 1100, 1100],
    ["l", "r", "r", "r", "r", "r"],
    note="分數相同而跨越界線的通報按比例計入。",
)
figure(4, "累積增益曲線。橫軸為依 P(系統) 由高到低納入的通報比例，縱軸為所涵蓋的「系統」事件比例；對角線為隨機排序。")

h2("4.7　錯誤分析")
ms = OV["missed"]
para(
    f"測試集的 {N_SYS} 筆「系統」事件中，LoRA-S 與 LoRA-L（τ*）分別漏判 {ms['v1e1']} 與 {ms['v2e2']} 筆，TF-IDF＋LR（門檻 0.5）"
    f"漏判 {ms['tfidf']} 筆。三個模型皆漏判者有 {OV['all3']} 筆，至少被一個模型找出者有 {OV['any_of_3_hit']} 筆。兩個微調模型的"
    f"漏判高度重疊（共同漏判 {OV['v1e1_and_v2e2']} 筆）。在三者皆漏判的 {OV['all3']} 筆中，有 {len(low)} 筆被 LoRA-L 賦予低於 "
    f"0.05 的 P(系統)（最低為 {min(c['p_v2e2'] for c in D['all3_cases']):.4f}），亦即模型並非處於不確定的狀態，而是以高信心"
    f"判為「個人」。驗證集的 {sum(r['idt_target'] == '系統' for r in SETS['val'])} 筆「系統」事件中，三個模型皆漏判者有 "
    f"{OV['val']['all3']} 筆。"
)

h2("4.8　判定資訊的來源")
gd, gc = FD["minus_description"], FDC["minus_description"]
conflict = IN["same_text"]["conflicting"][0]
assert sorted(label for _, label in conflict["records"]) == ["個人", "個人", "系統"] and "手術通知單" in conflict["description"]
para(
    f"表 7 與圖 5 為輸入替換分析的結果。同一個 TF-IDF＋LR 模型以調查處理紀錄取代事件描述作為輸入時，測試集 AUROC 由 "
    f"{f3(FI['description']['auroc'])} 提高至 {f3(FD['auroc'])}（差異 {dci(gd['auroc'])}），macro-F1 由 "
    f"{f3(FI['description']['macro_f1'])} 提高至 {f3(FD['macro_f1'])}（差異 {dci(gd['macro_f1'])}）；分組五折交叉驗證的結果"
    f"方向一致（AUROC {f3(CV['description']['auroc'])} 對 {f3(CV['directive']['auroc'])}）。將事件描述與調查處理紀錄串接，"
    "並未優於單獨使用調查處理紀錄。"
)
para(
    f"調查處理紀錄中含有與標籤相關的流程用語，例如「敬會」一詞出現於 {pct(CUE['敬會']['system_rate'])} 的「系統」事件與 "
    f"{pct(CUE['敬會']['person_rate'])} 的「個人」事件。去除這類用語與會辦單位名稱後，AUROC 仍為 {f3(FDC['auroc'])}，"
    f"與事件描述的差異 {dci(gc['auroc'])} 仍達統計顯著，顯示增益並非僅來自流程用語。"
)
para(
    f"在標註一致性方面，事件描述高度相似的 {ND['pairs']} 對通報中，{ND['pairs'] - ND['label_conflicts']} 對"
    f"（{(ND['pairs'] - ND['label_conflicts']) / ND['pairs'] * 100:.0f}%）的標籤相同。全部資料中僅有一組事件描述完全相同而標籤"
    "不同的通報：三筆皆記載手術通知單的處置代碼與術前診斷位置不符，其中一筆標為「系統」、兩筆標為「個人」，而三筆的"
    f"調查處理紀錄記載了不同的原因。此外，調查處理紀錄中載有「原因說明」段落的通報佔 {pct(RC['all'])}（「系統」事件 "
    f"{pct(RC['system'])}，「個人」事件 {pct(RC['person'])}）。"
)
rows7 = []
for name, key in (("事件描述", "description"), ("調查處理紀錄\n（去除流程用語）", "directive（去掉流程用語）"), ("調查處理紀錄", "directive"), ("事件描述＋\n調查處理紀錄", "description+directive")):
    item, gap, cv = FI[key], FI[key].get("minus_description"), CV.get(key)
    rows7.append([name, f"{f3(item['auroc'])} {ci3(item['ci']['auroc'])}", increase(gap["auroc"]) if gap else "—",
                  f"{f3(item['macro_f1'])} {ci3(item['ci']['macro_f1'])}", f3(item["system_recall"]), f3(cv["auroc"]) if cv else "—"])
table(
    7,
    "以相同的 TF-IDF＋LR 模型替換輸入來源的結果",
    ["輸入", "AUROC（95% CI）", "ΔAUROC（95% CI）", "macro-F1（95% CI）", "召回率", "五折\nAUROC"],
    rows7,
    [1750, 1800, 1900, 1850, 800, 900],
    ["l", "r", "r", "r", "r", "r"],
    note=f"除最後一欄外，均為測試集的結果；macro-F1 與「系統」召回率以門檻 0.5 計算，ΔAUROC 為相對於事件描述的配對差異。"
    f"五折 AUROC 為全部 {n(len(EVERYTHING))} 筆的分組交叉驗證（相同描述同組，三個亂數種子的平均）。",
)
figure(5, "不同輸入來源下的測試集 AUROC 及其 95% 信賴區間。以事件描述為輸入的語言模型與詞彙式模型位於相近的水準；輸入包含調查處理紀錄時明顯提高。")

# ------------------------------ 參考文獻 ------------------------------
h2("參考文獻【僅列第 3、4 章引用者；書目資料請核對後併入全文】")
for reference in (
    "Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
    "Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. International Conference on Learning Representations.",
    "Meadows, S., Baker, K., & Butler, J. (2005). The Incident Decision Tree: Guidelines for action following patient safety incidents. In Advances in Patient Safety: From Research to Implementation (Vol. 4). Agency for Healthcare Research and Quality.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
    "Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513–523.",
):
    body.append(
        '<w:p><w:pPr><w:spacing w:after="120" w:line="360" w:lineRule="auto"/><w:ind w:left="480" w:hanging="480"/></w:pPr>'
        + runs(reference, size=22)
        + "</w:p>"
    )

# ============================== 組檔 ==============================
scratch = Path(tempfile.mkdtemp(prefix="manuscript_"))
work = scratch / "pkg"
with zipfile.ZipFile(ARGS.template) as template:
    template.extractall(work)
old_doc = (work / "word/document.xml").read_text(encoding="utf-8")
sect = re.sub(r' w:rsid\w*="[^"]*"', "", re.findall(r"<w:sectPr.*?</w:sectPr>", old_doc, flags=re.S)[-1])
document = old_doc[: old_doc.index("<w:body>")] + "<w:body>" + "".join(body) + sect + "</w:body></w:document>"
(work / "word/document.xml").write_text(document, encoding="utf-8")

# 樣式：樣式來源是實驗報告（無襯線字、藍色標題）；論文稿改用 Times New Roman＋新細明體與黑色標題
styles_path = work / "word/styles.xml"
styles = styles_path.read_text(encoding="utf-8")
for old, new in (
    ('w:ascii="Calibri" w:eastAsia="微軟正黑體" w:hAnsi="Calibri" w:cs="Calibri"',
     'w:ascii="Times New Roman" w:eastAsia="新細明體" w:hAnsi="Times New Roman" w:cs="Times New Roman"'),
    ('<w:color w:val="0B0B0B"/>', '<w:color w:val="000000"/>'),
    ('<w:color w:val="1F3864"/>', '<w:color w:val="000000"/>'),
    ('<w:color w:val="2E5496"/>', '<w:color w:val="000000"/>'),
):
    assert old in styles, f"樣式來源沒有預期的設定：{old}"
    styles = styles.replace(old, new)
styles_path.write_text(styles, encoding="utf-8")

# 圖片：拿掉樣式來源原有的圖與關聯，換成論文的五張
for old_image in (work / "word/media").glob("*"):
    old_image.unlink()
rels_path = work / "word/_rels/document.xml.rels"
rels = re.sub(r'<Relationship Id="[^"]+" Type="[^"]+/image" Target="[^"]+"/>', "", rels_path.read_text(encoding="utf-8"))
for number, source in enumerate(FIGURES, 1):
    shutil.copyfile(source, work / f"word/media/image{number}.png")
    rels = rels.replace(
        "</Relationships>",
        f'<Relationship Id="rIdFig{number}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="media/image{number}.png"/></Relationships>',
    )
rels_path.write_text(rels, encoding="utf-8")

header = (work / "word/header2.xml").read_text(encoding="utf-8")
assert header.count("實驗結果報告") == 1 and header.count(" 2026-10-18") == 1, "樣式來源的頁首不是預期的格式"
header = header.replace("實驗結果報告", "論文初稿　第 3、4 章").replace(" 2026-10-18", f" {ARGS.date}")
(work / "word/header2.xml").write_text(header, encoding="utf-8")

shutil.rmtree(work / "word/webextensions")
types = (work / "[Content_Types].xml").read_text(encoding="utf-8")
(work / "[Content_Types].xml").write_text(re.sub(r'<Override PartName="/word/webextensions/[^"]+" ContentType="[^"]+"/>', "", types), encoding="utf-8")
package_rels = (work / "_rels/.rels").read_text(encoding="utf-8")
(work / "_rels/.rels").write_text(re.sub(r'<Relationship Id="[^"]+" Type="[^"]*webextensiontaskpanes" Target="[^"]+"/>', "", package_rels), encoding="utf-8")

now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
(work / "docProps/core.xml").write_text(
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    "<dc:title>IDT 歸因分類：研究方法與實驗結果（論文初稿第 3、4 章）</dc:title><dc:creator>Chih-Chien Hsieh</dc:creator>"
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
print(f"{OUT}（{OUT.stat().st_size:,} bytes；表 {document.count('<w:tbl>')}、圖 {_counter['pic']}、公式 {_counter['eq']}）")
