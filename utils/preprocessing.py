"""步驟 ①：讀取與清洗。

正規化人工 IDT 標籤、剔除無法使用的列、為每筆保留來源識別碼 source_id，
並統計每種剔除原因的列數，讓每次執行都能回報輸入、剔除與輸出筆數。
"""

import unicodedata
from collections import Counter
from dataclasses import dataclass, field

# 人工 IDT 標籤只有兩類；其他值一律視為非標籤值剔除，不猜測。
VALID_IDT_LABELS = ("個人", "系統")
LABEL_COLUMN = "idt_target"
TEXT_COLUMN = "description"
DIRECTIVE_COLUMN = "directive"
REQUIRED_COLUMNS = (LABEL_COLUMN, TEXT_COLUMN)

# 剔除原因依檢查順序排列；每列只記錄第一個命中的原因。
DROP_REASONS = {
    "idt_blank": "IDT 標籤空白",
    "idt_invalid": "IDT 標籤不是「個人」或「系統」",
    "description_blank": "描述為空",
    "duplicate": "標籤 + 描述 + 批示完全重複",
}


def normalize_label(value: str | None) -> str:
    """NFKC 把全形字元轉成一般字元，再移除所有空白（含標籤中間的空格）。"""
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    return "".join(text.split())


def normalize_text(value: str | None) -> str | None:
    """文字欄位只去頭尾空白，內文（含換行）原樣保留；空字串回傳 None。"""
    text = ("" if value is None else str(value)).strip()
    return text or None


def count_labels(records: list[dict]) -> dict[str, int]:
    """依固定順序回報各 IDT 標籤的筆數，沒有的標籤也列出 0。"""
    counts = Counter(record[LABEL_COLUMN] for record in records)
    return {label: counts.get(label, 0) for label in VALID_IDT_LABELS}


@dataclass
class CleanReport:
    """一次清洗的統計；dropped 的 key 對應 DROP_REASONS。"""

    source: str
    input_rows: int = 0
    dropped: Counter = field(default_factory=Counter)
    output_rows: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "input_rows": self.input_rows,
            "dropped": {key: self.dropped.get(key, 0) for key in DROP_REASONS},
            "output_rows": self.output_rows,
            "label_counts": self.label_counts,
        }


def clean_records(rows: list[dict], source_name: str) -> tuple[list[dict], CleanReport]:
    """清洗 CSV 讀入的列，回傳可用紀錄與統計。

    rows 的第 i 列（從 0 起算）對應試算表第 i + 2 列（標題列是第 1 列），
    source_id 以「檔名:列號」記錄，之後回查原始資料或稽核切分時可直接對照。
    除了 idt_target 正規化成標籤值，其他欄位（含批示與空的 emotion_target）
    原樣保留並轉成去頭尾空白的字串，空值為 None。
    """
    # csv.DictReader 會把多出來的欄位掛在 None 底下，這些不是正式欄位。
    columns = [column for column in rows[0] if column is not None] if rows else []
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if rows and missing:
        raise ValueError(f"原始資料缺少欄位 {missing}；現有欄位：{columns}")

    report = CleanReport(source=source_name, input_rows=len(rows))
    records: list[dict] = []
    seen: set[tuple] = set()
    for index, row in enumerate(rows):
        label = normalize_label(row.get(LABEL_COLUMN))
        description = normalize_text(row.get(TEXT_COLUMN))
        if not label:
            report.dropped["idt_blank"] += 1
            continue
        if label not in VALID_IDT_LABELS:
            report.dropped["idt_invalid"] += 1
            continue
        if description is None:
            report.dropped["description_blank"] += 1
            continue
        record = {"source_id": f"{source_name}:{index + 2}"}
        for column in columns:
            record[column] = normalize_text(row.get(column))
        record[LABEL_COLUMN] = label
        # 重複只看標籤、描述與批示三者完全相同，保留第一次出現的那筆。
        key = (label, description, record.get(DIRECTIVE_COLUMN))
        if key in seen:
            report.dropped["duplicate"] += 1
            continue
        seen.add(key)
        records.append(record)

    report.output_rows = len(records)
    report.label_counts = count_labels(records)
    return records, report
