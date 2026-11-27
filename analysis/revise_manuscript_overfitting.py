"""2026-11-27 修訂論文稿第 3、4 章：加入過度擬合的防範與檢驗（直接改 .docx，不重新產生）。

docs/manuscript_ch3_ch4_2026-11-23.docx 在 2026-11-24 之後由作者在 Word 裡手動編修過（拿掉【　】、部分引用與
參考文獻），所以不能再用 analysis.build_manuscript 重新產生——那會蓋掉手動的修改。這支腳本只動需要改的段落，
其餘內容逐位元組保留：

    3.11（新）  過度擬合的防範與檢驗；原 3.11、3.12 順延為 3.12、3.13，內文的「3.11 節」改為「3.12 節」
    4.2         第二段：數字不變，改為先說明選定的檢查點在測試集的 log-loss 不高於驗證集
    4.4         標題改為「調適範圍與訓練資料」；訓練動態的段落與圖 3 移到 4.5
    4.5（新）  訓練動態與過度擬合的檢驗：新的圖 3（三格）、表 5（驗證集對測試集）、與訓練資料的相似度、模型容量
    4.6–4.9     原 4.5–4.8 順延；表 5–7 順延為表 6–8；門檻段落數字不變，補上驗證集決定的門檻與測試集最佳值的比較
    其他        4.3 的兩段、4.6 的召回率段落與 4.8 的錯誤分析各改一句：數字不變，改成先說明結果本身的寫法

數字由 results/analysis/idt_overfitting.json（python -m analysis.idt_overfitting）與 idt_comparison.json 代入，
敘述以 assert 綁在數字上；圖 3 用 results/analysis/ms_fig3_overfitting.png（python -m analysis.manuscript_figures）。
只能套用在尚未修訂的稿件上，已修訂過（已有「過度擬合的防範與檢驗」）就中止。

在專案根目錄執行：
    python -m analysis.revise_manuscript_overfitting --source <修訂前的稿件> --out <輸出>
    python -m analysis.revise_manuscript_overfitting      # 就地修訂 docs/manuscript_ch3_ch4_2026-11-23.docx，
                                                          # 修訂前的版本另存為同目錄的「…（修訂前備份）.docx」
產生後請用 Word 開啟檢查分頁。
"""

import argparse
import json
import re
import shutil
import unicodedata
import zipfile
from functools import lru_cache
from html import unescape
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageFont

ROOT = Path.cwd()
DEFAULT = ROOT / "docs/manuscript_ch3_ch4_2026-11-23.docx"
parser = argparse.ArgumentParser(description="在論文稿第 3、4 章加入過度擬合的防範與檢驗")
parser.add_argument("--source", type=Path, default=DEFAULT, help="尚未修訂的稿件")
parser.add_argument("--out", type=Path, default=None, help="輸出；預設就地修訂 --source，並先備份")
parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results/analysis")
ARGS = parser.parse_args()
OUT = ARGS.out or ARGS.source
ANALYSIS = ARGS.analysis_dir
FIGURE = ANALYSIS / "ms_fig3_overfitting.png"
for needed in (ARGS.source, ANALYSIS / "idt_overfitting.json", ANALYSIS / "idt_comparison.json", FIGURE):
    if not needed.is_file():
        raise SystemExit(f"找不到 {needed}；產生方式見這個檔案開頭的說明。")
lock = OUT.with_name("~$" + OUT.name[2:])
if lock.exists():
    raise SystemExit(f"{OUT.name} 正在 Word 裡開啟（{lock.name}）；請先關閉文件再執行。")
O = json.loads((ANALYSIS / "idt_overfitting.json").read_text(encoding="utf-8"))
D = json.loads((ANALYSIS / "idt_comparison.json").read_text(encoding="utf-8"))
GEN, SIM, TR = O["generalization"], O["similarity"], O["training"]
S_KEY, L_KEY, TF_KEY, ZS_KEY = "v1-epoch-1", "v2-epoch-2", "tfidf", "v1-zero-shot"
assert O["selected"] == {"v1": S_KEY, "v2": L_KEY}


# ------------------------------ 數字格式（與 build_manuscript 相同）------------------------------
def minus(text):
    return text.replace("-", "−")


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x):
    return f"{x * 100:.1f}%"


def signed(x):
    text = f"{x:+.3f}"
    return minus(text[1:] if float(text) == 0 else text)


def dci(item):
    return f"{signed(item['diff'])} [{signed(item['ci'][0])}, {signed(item['ci'][1])}]"


def vci(value, interval, digits=3):
    return f"{value:.{digits}f} [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def includes(interval, value=0.0):
    return interval[0] <= value <= interval[1]


# ------------------------------ WordprocessingML ------------------------------
BODY, TABLE = 24, 18
CELL_PADDING_PT = 11
LETTERS = "A-Za-zΑ-Ωα-ω"


@lru_cache
def _times(bold):
    path = Path("/System/Library/Fonts/Supplemental") / ("Times New Roman Bold.ttf" if bold else "Times New Roman.ttf")
    return ImageFont.truetype(str(path), 200) if path.is_file() else None


def text_width(text, points, bold=False):
    """一行文字在 Word 裡的寬度（pt）估計：全形字元算一個字寬，其餘以 Times New Roman 量測。"""
    font = _times(bold)
    if font is None:
        return None
    wide = sum(unicodedata.east_asian_width(ch) in "WF" for ch in text)
    narrow = "".join(ch for ch in text if unicodedata.east_asian_width(ch) not in "WF")
    return (wide + font.getlength(narrow) / 200) * points


def inline_math(source, size, color=None, bold=False):
    """τ^* 這類行內符號 → 一般文字的 <w:r>（斜體與上標），四個字型欄位都指定 Times New Roman。"""

    def run(text, italic, align=None):
        props = '<w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
        props += "<w:b/><w:bCs/>" if bold else ""
        props += "<w:i/><w:iCs/>" if italic else ""
        props += f'<w:color w:val="{color}"/>' if color else ""
        props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        props += f'<w:vertAlign w:val="{align}"/>' if align else ""
        return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'

    out, position = [], 0
    for match in re.finditer(r"([_^])(?:\{([^}]*)\}|(.))", source):
        piece = source[position : match.start()]
        out.extend(run(p, bool(re.match(f"[{LETTERS}]", p))) for p in re.findall(f"[{LETTERS}]+|[^{LETTERS}]+", piece))
        script = match.group(2) if match.group(2) is not None else match.group(3)
        out.append(run(script, len(script) == 1 and bool(re.match(f"[{LETTERS}]", script)), "subscript" if match.group(1) == "_" else "superscript"))
        position = match.end()
    rest = source[position:]
    out.extend(run(p, bool(re.match(f"[{LETTERS}]", p))) for p in re.findall(f"[{LETTERS}]+|[^{LETTERS}]+", rest))
    return "".join(out)


def runs(text, size=BODY, color=None, bold=False):
    """文字 → 一串 <w:r>；`**粗體**`、換行，τ 與 τ* 排成數學符號。"""
    out = []
    for i, part in enumerate(text.split("**")):
        if not part:
            continue
        strong = bold or i % 2 == 1
        props = ("<w:b/><w:bCs/>" if strong else "") + (f'<w:color w:val="{color}"/>' if color else "")
        rpr = f'<w:rPr>{props}<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
        for token in re.split(r"(τ\*?)", part):
            if token in ("τ", "τ*"):
                out.append(inline_math("τ^*" if token == "τ*" else "τ", size, color, strong))
                continue
            for j, line in enumerate(token.split("\n")):
                if j:
                    out.append(f"<w:r>{rpr}<w:br/></w:r>")
                if line:
                    out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(line)}</w:t></w:r>')
    return "".join(out)


# 以下的段落屬性照 Word 存檔後的寫法（2026-11-24 的稿件）
BODY_PPR = '<w:pPr><w:spacing w:line="480" w:lineRule="auto"/><w:ind w:firstLineChars="200" w:firstLine="480"/><w:jc w:val="both"/></w:pPr>'


def heading2(text):
    return f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr>{runs(text, size=24)}</w:p>'


def para(text):
    return f"<w:p>{BODY_PPR}{runs(text)}</w:p>"


def caption(label, text, keep_next):
    ppr = '<w:pPr><w:keepNext/><w:spacing w:before="200" w:after="80"/></w:pPr>' if keep_next else '<w:pPr><w:spacing w:before="60" w:after="240"/><w:jc w:val="both"/></w:pPr>'
    return f"<w:p>{ppr}{runs(label, size=20, bold=True)}{runs('　' + text, size=20)}</w:p>"


def note(text):
    return (
        '<w:p><w:pPr><w:spacing w:before="60" w:after="200" w:line="260" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
        + runs(text, size=17, color="3A3A3A")
        + "</w:p>"
    )


RULE = '<w:{side} w:val="single" w:sz="{size}" w:space="0" w:color="000000"/>'
CELL_MARGIN = (
    '<w:tcMar><w:top w:w="45" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
    '<w:bottom w:w="45" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tcMar>'
)


EMPTY_MARK = f'<w:rPr><w:sz w:val="{TABLE}"/><w:szCs w:val="{TABLE}"/></w:rPr>'  # 空白儲存格的段落符號也用表格字級，各列才會等高


def table(number, headers, rows, widths, aligns, rule_before=()):
    """三線表（與稿件其他表格相同的寫法）；每一格的每一行都必須放得進欄寬。"""
    assert sum(widths) == 9000 and all(len(row) == len(headers) for row in rows)
    for r, row in enumerate([headers, *rows]):
        for c, (content, width) in enumerate(zip(row, widths, strict=True)):
            for line in str(content).split("\n"):
                needed = text_width(line, TABLE / 2)
                assert needed is None or needed + CELL_PADDING_PT <= width / 20, f"表 {number} 第 {r} 列第 {c + 1} 欄放不下「{line}」：需要 {needed + CELL_PADDING_PT:.0f} pt，欄寬 {width / 20:.0f} pt"
    jc = {"l": "", "c": '<w:jc w:val="center"/>', "r": '<w:jc w:val="right"/>'}

    def row_xml(cells, header=False, top_rule=False):
        borders = ""
        if header:
            borders = f"<w:tcBorders>{RULE.format(side='bottom', size=6)}</w:tcBorders>"
        elif top_rule:
            borders = f"<w:tcBorders>{RULE.format(side='top', size=4)}</w:tcBorders>"
        xml = "".join(
            f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{borders}{CELL_MARGIN}<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:keepNext/><w:spacing w:line="260" w:lineRule="auto"/>{jc["c"] if header else jc[a]}{"" if str(cell) else EMPTY_MARK}</w:pPr>'
            f"{runs(str(cell), size=TABLE)}</w:p></w:tc>"
            for cell, w, a in zip(cells, widths, aligns, strict=True)
        )
        props = '<w:trPr><w:tblHeader/><w:jc w:val="center"/></w:trPr>' if header else '<w:trPr><w:cantSplit/><w:jc w:val="center"/></w:trPr>'
        return f"<w:tr>{props}{xml}</w:tr>"

    return (
        '<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>'
        + RULE.format(side="top", size=12)
        + '<w:left w:val="nil"/>'
        + RULE.format(side="bottom", size=12)
        + '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders><w:tblLayout w:type="fixed"/>'
        '<w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
        "</w:tblPr><w:tblGrid>"
        + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        + "</w:tblGrid>"
        + row_xml(headers, header=True)
        + "".join(row_xml(row, top_rule=i in rule_before) for i, row in enumerate(rows))
        + "</w:tbl>"
    )


# ------------------------------ 既有內容的定位與修改 ------------------------------
TOKEN = re.compile(r"<(/?)(w:p|w:tbl|w:sectPr)(?=[ >/])([^>]*)>")
T_ELEMENT = re.compile(r"<w:t(?: [^>]*)?>([^<]*)</w:t>")


def split_body(doc):
    """document.xml → (開頭, [[種類, XML], …], 結尾)；每個元素是 <w:body> 底下的一個段落、表格或 sectPr。"""
    start, end = doc.index("<w:body>") + len("<w:body>"), doc.rindex("</w:body>")
    body, blocks, depth, opened, kind = doc[start:end], [], 0, None, None
    for m in TOKEN.finditer(body):
        closing, name, rest = m.group(1), m.group(2), m.group(3)
        if not closing:
            if depth == 0:
                opened, kind = m.start(), name
            if not rest.endswith("/"):
                depth += 1
            elif depth == 0:
                blocks.append([kind, body[opened : m.end()]])
        else:
            depth -= 1
            if depth == 0:
                blocks.append([kind, body[opened : m.end()]])
    assert "".join(xml for _, xml in blocks) == body
    return doc[:start], blocks, doc[end:]


def text_of(xml):
    return unescape("".join(T_ELEMENT.findall(xml)))


def substitute(xml, old, new, count=1):
    """把段落文字中的 old 換成 new。Word 會把一句話拆成許多 <w:r>，所以先接起所有 <w:t> 的文字找位置，
    再改回各自的 <w:t>：new 放進 old 起點所在的那個 <w:t>，old 其餘的字元從後面的 <w:t> 刪掉；格式不動。"""
    elements = list(T_ELEMENT.finditer(xml))
    pieces = [unescape(m.group(1)) for m in elements]
    joined = "".join(pieces)
    assert joined.count(old) == count, f"「{old}」在段落中出現 {joined.count(old)} 次，預期 {count} 次：{joined[:60]}"
    offsets, position = [], 0
    for piece in pieces:
        offsets.append(position)
        position += len(piece)
    for start in reversed([m.start() for m in re.finditer(re.escape(old), joined)]):
        end, placed = start + len(old), False
        for i in reversed(range(len(pieces))):
            lo, hi = offsets[i], offsets[i] + len(pieces[i])
            if hi <= start or lo >= end:
                continue
            cut_from, cut_to = max(start, lo) - lo, min(end, hi) - lo
            insert = new if lo <= start < hi else ""
            placed = placed or lo <= start < hi
            pieces[i] = pieces[i][:cut_from] + insert + pieces[i][cut_to:]
        assert placed, old
    out, position = [], 0
    for m, piece in zip(elements, pieces, strict=True):
        out.append(xml[position : m.start()])
        out.append(f'<w:t xml:space="preserve">{escape(piece)}</w:t>')
        position = m.end()
    out.append(xml[position:])
    return "".join(out)


def rebuild(xml, text):
    """整段重寫：保留原段落的起始標籤與段落屬性，文字換成 text。"""
    opening = re.match(r"<w:p(?: [^>]*)?>", xml).group(0)
    ppr = re.search(r"<w:pPr>.*?</w:pPr>", xml, flags=re.S).group(0)
    return f"{opening}{ppr}{runs(text)}</w:p>"


# ------------------------------ 讀稿件 ------------------------------
with zipfile.ZipFile(ARGS.source) as source:
    entries = [(info, source.read(info.filename)) for info in source.infolist()]
parts = {info.filename: data for info, data in entries}
head, blocks, tail = split_body(parts["word/document.xml"].decode("utf-8"))
original_blocks = [xml for _, xml in blocks]
texts = [text_of(xml) for _, xml in blocks]
if any(t.endswith("過度擬合的防範與檢驗") for t in texts):
    raise SystemExit(f"{ARGS.source} 已經修訂過（已有「過度擬合的防範與檢驗」）；請從修訂前的版本開始。")


def find(prefix, kind="w:p"):
    hits = [i for i, (k, _) in enumerate(blocks) if k == kind and texts[i].startswith(prefix)]
    assert len(hits) == 1, f"找不到唯一一個以「{prefix}」開頭的段落（{len(hits)} 個）；稿件與預期的修訂前版本不同。"
    return hits[0]


full_text = "\n".join(texts)
references = {k: len(re.findall(k, full_text)) for k in (r"表 5(?!\d)", r"表 6(?!\d)", r"表 7(?!\d)", r"表 8(?!\d)", r"3\.11 節", r"圖 3")}
assert references == {r"表 5(?!\d)": 3, r"表 6(?!\d)": 2, r"表 7(?!\d)": 2, r"表 8(?!\d)": 0, r"3\.11 節": 1, r"圖 3": 2}, references
assert not any(re.search(r"表 [567]", texts[i]) for i, (k, _) in enumerate(blocks) if k == "w:tbl")

H_311, H_312 = find("3.11　資訊來源分析"), find("3.12　實作環境")
H_44, H_45, H_46, H_47, H_48 = (find(t) for t in ("4.4　調適範圍、訓練資料與訓練動態", "4.5　決策門檻與分類表現", "4.6　作為篩檢工具的效益", "4.7　錯誤分析", "4.8　判定資訊的來源"))
P_SOURCE = find("資料取自")
P_42 = find("LoRA-S 的三個檢查點顯示，延長訓練並未再提升排序能力")
P_44_SCOPE = find("LoRA-L 的可訓練參數約為 LoRA-S 的")
P_44_DYNAMICS = find("圖 3 以累計更新步數對齊兩種設定的訓練過程")
P_FIG3 = P_44_DYNAMICS + 1
P_FIG3_CAPTION = find("圖 3　兩種微調設定的驗證損失（左）與訓練損失（右）")
P_THRESHOLD = find("門檻在相當大的範圍內對 macro-F1 的影響有限")
P_43_MAIN = find("在主要比較上，選定的微調模型並未優於 TF-IDF＋LR。")
P_43_RECALL = find("表 4 中，微調模型與 TF-IDF＋LR 之間唯一達統計顯著的差異")
P_46_RECALL = find("選定的 LoRA-L 在 τ* = ")
P_48_ERRORS = find("測試集的 62 筆「系統」事件中，LoRA-S 與 LoRA-L（τ*）分別漏判")
assert (P_44_SCOPE, P_44_DYNAMICS, P_FIG3, P_FIG3_CAPTION, H_45) == (H_44 + 1, H_44 + 2, H_44 + 3, H_44 + 4, H_44 + 5)
assert "<w:drawing>" in blocks[P_FIG3][1] and texts[P_FIG3] == ""

# ------------------------------ 取值與核對 ------------------------------
gS, gL, gT, gZ = (GEN[k] for k in (S_KEY, L_KEY, TF_KEY, ZS_KEY))
hS, hL = TR["v1"]["history"], TR["v2"]["history"]
steps_s, steps_l = [h["global_step"] for h in hS], [h["global_step"] for h in hL]
assert TR["v1"]["selected_is_lowest_val_loss"] and TR["v2"]["selected_is_lowest_val_loss"]
assert TR["v1"]["val_loss_rises_after_selected"] and (TR["v1"]["selected_step"], TR["v2"]["selected_step"]) == (steps_s[0], steps_l[1])
assert hS[0]["train_loss"] > hS[1]["train_loss"] > hS[2]["train_loss"] and hS[0]["val_loss"] < hS[1]["val_loss"] < hS[2]["val_loss"]
early = [hL[0]["val_loss"], hL[1]["val_loss"], hS[0]["val_loss"]]  # 288、576、864 步
assert hL[1]["val_loss"] <= hL[0]["val_loss"] and max(early) - min(early) < 0.001 and steps_l[-1] < steps_s[0]
assert hS[0]["test_log_loss"] < hS[1]["test_log_loss"] < hS[2]["test_log_loss"] and abs(hL[0]["test_log_loss"] - hL[1]["test_log_loss"]) < 0.01
assert all(g["test"]["log_loss"] <= g["val"]["log_loss"] and g["test"]["auroc"] >= g["val"]["auroc"] for g in (gS, gL))
METRICS = ("log_loss", "auroc", "macro_f1_thr")
assert all(includes(g["test_minus_val"][m]["ci"]) for g in (gS, gL, gT, gZ) for m in METRICS)  # 表 5 的十二個區間都包含 0
better = sum((g["test_minus_val"][m]["diff"] < 0) if m == "log_loss" else (g["test_minus_val"][m]["diff"] > 0) for g in (gS, gL) for m in METRICS)
assert better == 5 and gL["test_minus_val"]["macro_f1_thr"]["diff"] < 0
assert abs(gZ["test_minus_val"]["auroc"]["diff"]) >= max(abs(g["test_minus_val"]["auroc"]["diff"]) for g in (gS, gL))

near, near_aug = SIM["nearest_train"], SIM["nearest_train_with_paraphrases"]
excluded, low, high, whole = SIM["exclude_near_duplicates"]["0.8"], SIM["low_half"], SIM["high_half"], SIM["all"]
n_test, n_system = whole["records"], whole["system"]
assert O["records"]["test"] == {"all": n_test, "system": n_system}
assert O["near_duplicate_level"] == 0.8 and near["ge_0.8"] == near_aug["ge_0.8"] == n_test - excluded["records"]
assert all(abs(excluded["auroc"][k]["value"] - whole["auroc"][k]["value"]) < 0.005 for k in (S_KEY, L_KEY))
gain_low = [low["paired"][f"{k}_minus_{ZS_KEY}"] for k in (S_KEY, L_KEY)]
gain_high = [high["paired"][f"{k}_minus_{ZS_KEY}"] for k in (S_KEY, L_KEY)]
assert all(item["ci"][0] > 0 for item in (*gain_low, *gain_high))

c_ll_ls = D["comparisons"]["v2e2_thr_minus_v1e1_thr"]["auroc"]
ratio = TR["v2"]["trainable_params"] / TR["v1"]["trainable_params"]
assert includes(c_ll_ls["ci"]) and round(ratio) == 36
assert all(includes(D["comparisons"][k]["auroc"]["ci"]) for k in ("v2e2_thr_minus_tfidf_thr", "v1e1_thr_minus_tfidf_thr"))
share_s, share_l = TR["v1"]["trainable_share"], TR["v2"]["trainable_share"]
assert f"{share_s * 100:.3f}%" == "0.012%" and f"{share_l * 100:.2f}%" == "0.42%" and TR["v1"]["lora_dropout"] == TR["v2"]["lora_dropout"] > 0
assert O["resamples"]["test"] == O["resamples"]["val"] == 2000

# 4.2 與 4.6 改寫的段落沿用原稿的數字：由 idt_comparison.json 取值，並核對原稿確實是這些數字
CFG, OC, P = {row["key"]: row for row in D["configs"]}, D["overconfidence"], D["points"]
oc_s = [OC[f"v1-epoch-{i}/test"]["pos_lt_0.01"] for i in (1, 2, 3)]
oc_l = [OC[f"v2-epoch-{i}/test"]["pos_lt_0.01"] for i in (1, 2)]
va = [CFG[f"v1-epoch-{i}"]["val"]["auroc"] for i in (1, 2, 3)]
vl = [CFG[f"v1-epoch-{i}"]["val"]["log_loss"] for i in (1, 2, 3)]
vl_l = [CFG[f"v2-epoch-{i}"]["val"]["log_loss"] for i in (1, 2)]
OLD_42 = texts[P_42]
for expected in (
    f"驗證集 AUROC 介於 {min(va):.3f}–{max(va):.3f}", f"由 {f4(vl[0])} 上升至 {f4(vl[1])} 與 {f4(vl[2])}",
    f"{n_system} 筆「系統」事件中 P(系統) 低於 0.01 者由 {oc_s[0]} 筆增加為 {oc_s[1]} 筆與 {oc_s[2]} 筆",
    f"（{f4(vl_l[0])} 與 {f4(vl_l[1])}）", f"分別為 {oc_l[0]} 筆與 {oc_l[1]} 筆", "同時也是測試集 log-loss 最低的檢查點",
):
    assert expected in OLD_42, expected
assert oc_s[0] < oc_s[1] < oc_s[2] and vl[0] < vl[1] < vl[2] and (f4(vl[0]), f4(vl_l[1])) == (f4(gS["val"]["log_loss"]), f4(gL["val"]["log_loss"]))

LS, LL, TFT, TF5 = P["v1-epoch-1@thr"], P["v2-epoch-2@thr"], P["tfidf@thr"], P["tfidf@0.5"]
sens = {round(r["threshold"], 3): r["test"] for r in D["sensitivity_v2e2"]}
band = [v["macro_f1"] for t, v in sens.items() if 0.05 <= t <= 0.3]
ceil = {k: v["test_best_macro_f1"] for k, v in D["ceiling"].items()}
ceil_all = [ceil[k] for k in ("v1-epoch-1", "v1-epoch-2", "v1-epoch-3", "v2-epoch-1", "v2-epoch-2")]
majority = 1 - D["system_rate_test"]
OLD_THRESHOLD = texts[P_THRESHOLD]
for expected in (
    f"測試集 macro-F1 介於 {min(band):.3f}–{max(band):.3f}", f"由 {sens[0.05]['tp']} 筆降為 {sens[0.3]['tp']} 筆", f"由 {sens[0.05]['fp']} 筆降為 {sens[0.3]['fp']} 筆",
    f"上限也僅為 {min(ceil_all):.3f}–{max(ceil_all):.3f}", f"判為「個人」時的 {pct(majority)}",
    f"在驗證集為 {f3(gL['val']['macro_f1_thr'])}，在測試集為 {f3(gL['test']['macro_f1_thr'])}", f"macro-F1（{f3(TFT['macro_f1'])}）低於門檻 0.5 的結果（{f3(TF5['macro_f1'])}）",
):
    assert expected in OLD_THRESHOLD, expected
assert (f3(LS["macro_f1"]), f3(LL["macro_f1"])) == (f3(gS["test"]["macro_f1_thr"]), f3(gL["test"]["macro_f1_thr"]))
assert 0 <= ceil[S_KEY] - LS["macro_f1"] < 0.02 and 0 <= ceil[L_KEY] - LL["macro_f1"] < 0.02 and TF5["macro_f1"] > TFT["macro_f1"]

# ------------------------------ 新的內容 ------------------------------
METHODS = [
    heading2("3.11　過度擬合的防範與檢驗"),
    para(
        "為降低過度擬合的風險，本研究在設計上採取下列措施。第一，資料切分以事件描述分組，相同的文字不會同時出現在訓練與評估資料"
        "（3.3 節），驗證集與測試集也不做資料增生。第二，微調只更新低秩增量，可訓練參數僅佔模型全部參數的 "
        f"{share_s * 100:.3f}%（LoRA-S）與 {share_l * 100:.2f}%（LoRA-L），並使用 dropout（表 1）。第三，檢查點依驗證集的 log-loss "
        "選擇（3.8 節），作用相當於提早停止（early stopping）：訓練若持續到驗證損失回升之後，該檢查點不會被選用。第四，測試集不參與"
        "檢查點與門檻的選擇；兩種設定全部五個檢查點在測試集的結果均予以報告（表 3），而非只報告表現最佳者。"
    ),
    para(
        "在此之外，我們以三項分析檢驗選定的模型是否過度擬合。第一，訓練動態：比較兩種設定在各檢查點的訓練損失與驗證損失，並檢視"
        "測試集的 log-loss 是否隨驗證集變化。第二，由驗證集到測試集的表現變化：檢查點與門檻由驗證集選出，若這些選擇過度擬合於"
        "驗證集，測試集的表現將低於驗證集；我們計算 log-loss、AUROC 與 τ* 下的 macro-F1 在兩個子集的差值，並對驗證集與測試集各自"
        f"獨立重抽樣 {O['resamples']['test']:,} 次，以百分位法取得差值的 95% 信賴區間。zero-shot 模型未經任何訓練，其差值只反映"
        "兩個子集的抽樣變異，作為參照。第三，與訓練資料的相似度：以字元 2 至 3 元組的 TF-IDF 餘弦相似度，計算每筆測試通報與"
        "訓練集中最相近通報的相似度；我們排除相似度達 0.8 以上的測試通報後重新計算 AUROC，並依相似度的中位數將測試集分為兩半，"
        "比較微調模型相對於 zero-shot 的配對差異。模型的表現若來自記憶訓練資料，排除相近的通報後表現應下降，微調的增益也應只出現在"
        f"與訓練資料相近的通報。此外，LoRA-L 的可訓練參數為 LoRA-S 的 {ratio:.0f} 倍，兩者的比較可用以檢視模型容量增加時測試集的"
        "表現是否下降。"
    ),
]

# 4.2 第二段：數字與原稿相同，改為先說明選定的檢查點
NEW_42 = (
    f"選定的兩個檢查點在測試集的 log-loss（{f4(gS['test']['log_loss'])} 與 {f4(gL['test']['log_loss'])}）都不高於驗證集"
    f"（{f4(gS['val']['log_loss'])} 與 {f4(gL['val']['log_loss'])}），AUROC 也不低於驗證集（表 5），未見模型選擇過度擬合於驗證集的跡象；"
    "各項檢驗見 4.5 節。以驗證集 log-loss 選出的 LoRA-S 第 1 個 epoch，同時也是測試集 log-loss 最低的檢查點，以高信心誤判的「系統」事件也最少："
    f"測試集 {n_system} 筆「系統」事件中 P(系統) 低於 0.01 者為 {oc_s[0]} 筆，第 2、3 個 epoch 為 {oc_s[1]} 筆與 {oc_s[2]} 筆。"
    f"延長訓練並未再提升排序能力（驗證集 AUROC 介於 {min(va):.3f}–{max(va):.3f}），驗證集 log-loss 則由 {f4(vl[0])} 回升至 "
    f"{f4(vl[1])} 與 {f4(vl[2])}，這兩個檢查點因此未被選用。LoRA-L 兩個檢查點的驗證集 log-loss 相近（{f4(vl_l[0])} 與 "
    f"{f4(vl_l[1])}），P(系統) 低於 0.01 的「系統」事件分別為 {oc_l[0]} 筆與 {oc_l[1]} 筆。"
)

RESULTS_LEAD = para(
    "本節由訓練動態、驗證集與測試集的表現比較、與訓練資料的相似度以及模型容量四個面向檢驗過度擬合。四項分析的結果一致："
    "選定的兩個模型均未呈現過度擬合的跡象。"
)
RESULTS_1 = para(
    "圖 3 以累計更新步數對齊兩種設定的訓練過程。圖 3a 與圖 3b 的損失為式 (2) 的數值，即每個答案 token 的平均交叉熵；圖 3c 為"
    "表 3 所列、以標籤機率計算的 log-loss，兩者尺度不同。選定的兩個檢查點（LoRA-S 的第 "
    f"{steps_s[0]} 步與 LoRA-L 的第 {steps_l[1]} 步）都是各自設定中驗證損失最低者（{f3(hS[0]['val_loss'])} 與 {f3(hL[1]['val_loss'])}）；"
    f"兩種設定在相近的訓練量下（{steps_l[0]} 至 {steps_s[0]} 步）驗證損失一致（{min(early):.3f}–{max(early):.3f}），LoRA-L 的驗證損失"
    f"在兩個檢查點之間持平（{f4(hL[0]['val_loss'])} 與 {f4(hL[1]['val_loss'])}）。LoRA-S 的驗證損失在第一個 epoch 之後回升"
    f"（{steps_s[1]:,} 與 {steps_s[2]:,} 步時為 {f3(hS[1]['val_loss'])} 與 {f3(hS[2]['val_loss'])}），訓練損失則持續下降；依 3.8 節的準則，"
    "模型選擇停在驗證損失回升之前，避免了延長訓練所造成的過度擬合。測試集的結果與驗證集一致（圖 3c）：LoRA-S 依驗證集選出的"
    f"檢查點同時也是測試集 log-loss 最低者（{f3(hS[0]['test_log_loss'])}，其後為 {f3(hS[1]['test_log_loss'])} 與 "
    f"{f3(hS[2]['test_log_loss'])}），LoRA-L 兩個檢查點的測試集 log-loss 相近（{f3(hL[0]['test_log_loss'])} 與 {f3(hL[1]['test_log_loss'])}）。"
)
FIG3_CAPTION = caption(
    "圖 3",
    "兩種微調設定的訓練動態。(a) 訓練損失與 (b) 驗證損失為式 (2) 的數值；(c) 為驗證集（實線）與測試集（虛線）的 log-loss。"
    "橫軸為累計更新步數，每個點代表一個 epoch 結束；黑色外框的點為依驗證集 log-loss 選定的檢查點。訓練損失為各 epoch "
    "訓練過程中的平均值，驗證損失於 epoch 結束時計算，兩者不宜直接相減。",
    keep_next=False,
)
d = {name: g["test_minus_val"] for name, g in (("S", gS), ("L", gL), ("Z", gZ))}
RESULTS_2 = para(
    "檢查點與門檻由驗證集選出，若這些選擇過度擬合於驗證集，測試集的表現應低於驗證集。表 5 顯示並無此現象。LoRA-S 與 LoRA-L "
    f"在測試集的 log-loss（{f4(gS['test']['log_loss'])} 與 {f4(gL['test']['log_loss'])}）都不高於驗證集"
    f"（{f4(gS['val']['log_loss'])} 與 {f4(gL['val']['log_loss'])}），差值為 {dci(d['S']['log_loss'])} 與 {dci(d['L']['log_loss'])}；"
    f"AUROC 的差值為 {dci(d['S']['auroc'])} 與 {dci(d['L']['auroc'])}；τ* 下 macro-F1 的差值為 {dci(d['S']['macro_f1_thr'])} 與 "
    f"{dci(d['L']['macro_f1_thr'])}。六個差值的信賴區間均包含 0，其中五個的點估計為測試集較佳。作為參照，未經任何訓練的 "
    f"zero-shot 模型在兩個子集之間也有相同幅度的差值（AUROC {dci(d['Z']['auroc'])}，macro-F1 {dci(d['Z']['macro_f1_thr'])}），"
    "顯示這樣的差值在子集的抽樣變異範圍之內。"
)


def generalization_rows(metric, label, digits):
    rows = []
    for i, (name, g) in enumerate((("LoRA-S（epoch 1）", gS), ("LoRA-L（epoch 2）", gL), ("TF-IDF＋LR", gT), ("Zero-shot", gZ))):
        rows.append([label if i == 0 else "", name, f"{g['val'][metric]:.{digits}f}", f"{g['test'][metric]:.{digits}f}", dci(g["test_minus_val"][metric])])
    return rows


TABLE_5 = [
    caption("表 5", "各模型在驗證集與測試集的表現比較", keep_next=True),
    table(
        5,
        ["指標", "模型", "驗證集", "測試集", "差值（測試集 − 驗證集）\n[95% CI]"],
        generalization_rows("log_loss", "log-loss", 4) + generalization_rows("auroc", "AUROC", 3) + generalization_rows("macro_f1_thr", "macro-F1（τ*）", 3),
        [1600, 2100, 1150, 1150, 3000],
        ["l", "l", "r", "r", "r"],
        rule_before=(4, 8),
    ),
    note(
        "LoRA-S 與 LoRA-L 為依驗證集 log-loss 選定的檢查點；zero-shot 未經訓練，其差值只反映兩個子集的抽樣變異。信賴區間由驗證集"
        f"（{O['records']['val']['all']} 筆）與測試集（{n_test} 筆）各自獨立重抽樣 {O['resamples']['test']:,} 次取得，所有區間均包含 0。macro-F1 以各模型在"
        "驗證集決定的門檻 τ* 計算，驗證集的數值即選擇門檻時的最大值。log-loss 以低為佳，AUROC 與 macro-F1 以高為佳。CI：信賴區間。"
    ),
]
RESULTS_3 = para(
    f"測試集的表現也並非仰賴與訓練資料高度相近的通報。{n_test} 筆測試通報與訓練集中最相近通報的相似度，中位數為 {f3(near['median'])}"
    f"（四分位距 {f3(near['q1'])}–{f3(near['q3'])}）；相似度達 0.8 以上者僅 {near['ge_0.8']} 筆（{pct(near['ge_0.8'] / n_test)}），"
    f"將 LoRA-S 訓練時使用的改寫一併納入比對時仍為 {near_aug['ge_0.8']} 筆。排除這 {near['ge_0.8']} 筆後，LoRA-S 與 LoRA-L 的 AUROC 為 "
    f"{vci(excluded['auroc'][S_KEY]['value'], excluded['auroc'][S_KEY]['ci'])} 與 {vci(excluded['auroc'][L_KEY]['value'], excluded['auroc'][L_KEY]['ci'])}，"
    f"與全部測試集的結果（{f3(whole['auroc'][S_KEY]['value'])} 與 {f3(whole['auroc'][L_KEY]['value'])}）沒有差別。依相似度的中位數將測試集"
    "分為兩半時，微調相對於 zero-shot 的 AUROC 增益在兩半都達統計顯著：與訓練資料較不相似的一半為 "
    f"{dci(gain_low[0])}（LoRA-S）與 {dci(gain_low[1])}（LoRA-L），較相似的一半為 {dci(gain_high[0])} 與 {dci(gain_high[1])}。"
    "微調學到的判別能力可以延伸到與訓練資料差異較大的通報，與「表現來自記憶訓練資料」的解釋不符。"
)
RESULTS_4 = para(
    f"模型容量的增加也沒有使測試集的表現下降：LoRA-L 的可訓練參數為 LoRA-S 的 {ratio:.0f} 倍，兩者在測試集的 AUROC 差異為 "
    f"{dci(c_ll_ls)}（表 4）；兩者的 AUROC 也與參數少得多、帶有 L2 正則化的 TF-IDF＋LR 相當（4.3 節）。"
)

# 4.6 的門檻段落：數字與原稿相同，補上「驗證集決定的門檻接近測試集上的最佳值」與驗證集到測試集的差值
NEW_THRESHOLD = (
    f"門檻在相當大的範圍內對 macro-F1 的影響有限：LoRA-L 的門檻由 0.05 調整至 0.30 時，測試集 macro-F1 介於 {min(band):.3f}–"
    f"{max(band):.3f}，但找出的「系統」事件由 {sens[0.05]['tp']} 筆降為 {sens[0.3]['tp']} 筆，誤報由 {sens[0.05]['fp']} 筆降為 "
    f"{sens[0.3]['fp']} 筆；門檻的選擇主要決定召回率與誤報之間的取捨。由驗證集決定的門檻也接近測試集上的最佳值：直接以測試集選擇"
    f"門檻時，LoRA-S 與 LoRA-L 的 macro-F1 上限為 {f3(ceil[S_KEY])} 與 {f3(ceil[L_KEY])}（五個微調檢查點為 {min(ceil_all):.3f}–"
    f"{max(ceil_all):.3f}），τ* 的結果（{f3(LS['macro_f1'])} 與 {f3(LL['macro_f1'])}）與之相差不到 0.02。門檻由驗證集套用到測試集時，"
    f"macro-F1 的變化未達統計顯著（LoRA-S {dci(d['S']['macro_f1_thr'])}，LoRA-L {dci(d['L']['macro_f1_thr'])}；表 5）。準確率受類別"
    f"不平衡支配，將所有通報判為「個人」即達 {pct(majority)}，因此僅列為參考。TF-IDF＋LR 的 τ* 在測試集的 macro-F1"
    f"（{f3(TFT['macro_f1'])}）低於門檻 0.5 的結果（{f3(TF5['macro_f1'])}），4.3 節因此另以後者為對照。"
)

# 其他段落：數字與原稿相同，把「未優於」「並未高於」這類寫法改成先說明結果本身（期刊論文的寫法）
c_ls_tf, c_ll5_greedy = D["comparisons"]["v1e1_thr_minus_tfidf_thr"], D["comparisons"]["v2e2_05_minus_04"]
recall_gain = c_ls_tf["system_recall"]
flag_ls, flag_tf = LS["tp"] + LS["fp"], TFT["tp"] + TFT["fp"]
assert recall_gain["ci"][0] > 0 and LS["tp"] > TFT["tp"] and all(includes(c_ll5_greedy[k]["ci"]) for k in ("macro_f1", "system_recall", "system_precision"))
OLD_43_RECALL = texts[P_43_RECALL]
assert f"（{dci(recall_gain)}）" in OLD_43_RECALL and f"（{flag_ls} 筆，TF-IDF＋LR 為 {flag_tf} 筆）" in OLD_43_RECALL
MAIN_OLD, MAIN_NEW = "選定的微調模型並未優於 TF-IDF＋LR。", "選定的微調模型與 TF-IDF＋LR 的排序能力相當。"
NEW_43_RECALL = (
    "在驗證集決定的門檻下，LoRA-S 找出的「系統」事件多於 TF-IDF＋LR：召回率高出 "
    f"{recall_gain['diff']:.3f} [{recall_gain['ci'][0]:.3f}, {recall_gain['ci'][1]:.3f}]（表 4），判為「系統」的通報為 {flag_ls} 筆，"
    f"TF-IDF＋LR 為 {flag_tf} 筆。這項差異來自操作點的不同；如前所述，兩者的排序能力相當。"
)
LS5, LL5 = P["v1-epoch-1@0.5"], P["v2-epoch-2@0.5"]
OLD_46_TAIL = (
    f"召回率的提升來自決策門檻：在門檻 0.5 時，LoRA-L 與 LoRA-S 的召回率分別為 {pct(LL5['system_recall'])} 與 {pct(LS5['system_recall'])}，"
    f"並未高於生成式判定（LoRA-L 與生成式判定的 macro-F1 差異 {dci(c_ll5_greedy['macro_f1'])}）；門檻降至 τ* 後才提高至 "
    f"{pct(LL['system_recall'])} 與 {pct(LS['system_recall'])}。"
)
NEW_46_TAIL = (
    f"召回率的提升來自機率評分所提供的門檻調整：門檻由 0.5 降至 τ* 時，LoRA-L 與 LoRA-S 的召回率由 {pct(LL5['system_recall'])} 與 "
    f"{pct(LS5['system_recall'])} 提高至 {pct(LL['system_recall'])} 與 {pct(LS['system_recall'])}；門檻固定於 0.5 時，則與生成式判定"
    f"沒有顯著差異（LoRA-L 與生成式判定的 macro-F1 差異 {dci(c_ll5_greedy['macro_f1'])}）。"
)
assert texts[P_46_RECALL].count(OLD_46_TAIL) == 1 and texts[P_46_RECALL].endswith(OLD_46_TAIL)
ERRORS_OLD = "，亦即模型並非處於不確定的狀態，而是以高信心判為「個人」。"
ERRORS_NEW = "，顯示僅就事件描述而言，這些事件與「個人」事件難以區分。"
assert texts[P_48_ERRORS].count(ERRORS_OLD) == 1 and len([c for c in D["all3_cases"] if c["p_v2e2"] < 0.05]) == 9 and D["overlap"]["all3"] == 11

# ------------------------------ 套用 ------------------------------
for i, (kind, xml) in enumerate(blocks):  # 表號順延（由大到小，避免連續進位）；只改數字那一個字元
    if kind != "w:p":
        continue
    for old, new in (("表 7", "表 8"), ("表 6", "表 7"), ("表 5", "表 6")):
        hits = len(re.findall(old + r"(?!\d)", text_of(xml)))
        if hits:
            xml = substitute(xml, old, new, count=hits)
    blocks[i][1] = xml
blocks[P_SOURCE][1] = substitute(blocks[P_SOURCE][1], "3.11 節", "3.12 節")
for index, old, new in (
    (H_311, "3.11　", "3.12　"), (H_312, "3.12　", "3.13　"), (H_44, "　調適範圍、訓練資料與訓練動態", "　調適範圍與訓練資料"),
    (H_45, "4.5　", "4.6　"), (H_46, "4.6　", "4.7　"), (H_47, "4.7　", "4.8　"), (H_48, "4.8　", "4.9　"),
):
    blocks[index][1] = substitute(blocks[index][1], old, new)
blocks[P_42][1] = rebuild(blocks[P_42][1], NEW_42)
blocks[P_THRESHOLD][1] = rebuild(blocks[P_THRESHOLD][1], NEW_THRESHOLD)
blocks[P_43_MAIN][1] = substitute(blocks[P_43_MAIN][1], MAIN_OLD, MAIN_NEW)
blocks[P_43_RECALL][1] = rebuild(blocks[P_43_RECALL][1], NEW_43_RECALL)
blocks[P_46_RECALL][1] = rebuild(blocks[P_46_RECALL][1], texts[P_46_RECALL].replace(OLD_46_TAIL, NEW_46_TAIL))
blocks[P_48_ERRORS][1] = rebuild(blocks[P_48_ERRORS][1], texts[P_48_ERRORS].replace(ERRORS_OLD, ERRORS_NEW))

with Image.open(FIGURE) as image:
    width, height = image.size
drawing = blocks[P_FIG3][1]
extent = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', drawing)
cx, old_cy = int(extent.group(1)), int(extent.group(2))
cy = round(cx * height / width)
assert drawing.count(f'cx="{cx}" cy="{old_cy}"') == 2  # wp:extent 與 a:ext
drawing = drawing.replace(f'cx="{cx}" cy="{old_cy}"', f'cx="{cx}" cy="{cy}"')
embed = re.search(r'<a:blip r:embed="([^"]+)"/>', drawing).group(1)
relationships = parts["word/_rels/document.xml.rels"].decode("utf-8")
target = re.search(rf'<Relationship Id="{embed}" Type="[^"]+/image" Target="([^"]+)"/>', relationships).group(1)
parts[f"word/{target}"] = FIGURE.read_bytes()

section_45 = [
    [None, heading2("4.5　訓練動態與過度擬合的檢驗")],
    [None, RESULTS_LEAD],
    [None, RESULTS_1],
    [None, drawing],
    [None, FIG3_CAPTION],
    [None, RESULTS_2],
    *[[None, xml] for xml in TABLE_5],
    [None, RESULTS_3],
    [None, RESULTS_4],
]
changed = {P_SOURCE, H_311, H_312, H_44, H_45, H_46, H_47, H_48, P_42, P_THRESHOLD, P_44_DYNAMICS, P_FIG3, P_FIG3_CAPTION,
           P_43_MAIN, P_43_RECALL, P_46_RECALL, P_48_ERRORS}
renumbered = {i for i, (_, xml) in enumerate(blocks) if xml != original_blocks[i]} - changed
assert all(blocks[i][0] == "w:p" and re.search(r"表 [678]", text_of(blocks[i][1])) for i in renumbered)
# 由後往前插入，前面的索引才不會跑掉：先換掉 4.4 的訓練動態段落、圖 3 與圖題，再插入 3.11
blocks[P_44_DYNAMICS : P_FIG3_CAPTION + 1] = section_45
blocks[H_311:H_311] = [[None, xml] for xml in METHODS]
document = head + "".join(xml for _, xml in blocks) + tail
parts["word/document.xml"] = document.encode("utf-8")

# ------------------------------ 檢查修訂後的全文 ------------------------------
_, after, _ = split_body(document)
after_text = [text_of(xml) for _, xml in after]
headings = [t for (_, xml), t in zip(after, after_text, strict=True) if 'w:val="Heading2"' in xml]
assert [h.split("　")[0] for h in headings] == [f"3.{i}" for i in range(1, 14)] + [f"4.{i}" for i in range(1, 10)], headings
table_captions = [int(m.group(1)) for t in after_text if (m := re.match(r"表 (\d+)　", t))]
figure_captions = [int(m.group(1)) for t in after_text if (m := re.match(r"圖 (\d+)　", t))]
assert table_captions == list(range(1, 9)) and figure_captions == list(range(1, 6)), (table_captions, figure_captions)
assert document.count("<w:tbl>") == 8 and document.count("<w:drawing>") == 5
joined = "\n".join(after_text)
assert not re.search(r"表 (9|\d\d)", joined) and "3.11 節" not in joined and joined.count("3.12 節") == 1
untouched = [xml for i, xml in enumerate(original_blocks) if i not in changed and i not in renumbered]
assert all(xml in document for xml in untouched)  # 其餘段落、表格逐位元組不變

# ------------------------------ 寫檔 ------------------------------
if OUT.resolve() == ARGS.source.resolve():
    backup = ARGS.source.with_name(f"{ARGS.source.stem}（修訂前備份）.docx")
    if backup.exists():
        raise SystemExit(f"備份 {backup.name} 已經存在；為免蓋掉修訂前的版本，請先移走或改用 --out。")
    shutil.copy2(ARGS.source, backup)
    print(f"修訂前的版本已另存：{backup}")
OUT.parent.mkdir(parents=True, exist_ok=True)
temporary = OUT.with_name(OUT.name + ".tmp")
with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
    for info, _ in entries:  # 與原檔相同的順序（[Content_Types].xml 在最前面）
        z.writestr(info.filename, parts[info.filename])
temporary.replace(OUT)
print(f"{OUT}（{OUT.stat().st_size:,} bytes；表 {document.count('<w:tbl>')}、圖 {document.count('<w:drawing>')}；新增 3.11 與 4.5，圖 3 為 {width}×{height} px）")
