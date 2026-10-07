"""讀取月份工作表、清洗既有 IDT 標籤，並保留來源供資料切分稽核。"""

import json
import os
import re
import unicodedata

import pandas as pd

RAW_FILE = "data/raw_data/raw data去辨識0612.xlsx"
COLUMNS = ["IDT分析(個人,系統)", "事件描述", "批示"]
NEW_COLUMN_NAMES = ["idt_target", "description", "directive"]
TARGET_COLUMN = NEW_COLUMN_NAMES[0]
VALID_IDT_LABELS = ["個人", "系統"]
TRAIN_SHEETS = [f"112.{month:02d}" for month in range(1, 13)]
TEST_SHEETS = ["113.01", "113.02"]
SOURCE_COLUMNS = ["_source_sheet", "_source_excel_row"]


def _clean_text(value):
    """僅整理空值、前後空白與換行，不改動醫療敘述內容。"""
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_idt_label(value):
    """正規化既有標籤的全形字元與空白；不推測或新增標籤。"""
    text = unicodedata.normalize("NFKC", _clean_text(value))
    return re.sub(r"\s+", "", text)


def load_and_merge_data(file_path, sheet_names, columns):
    """讀取指定工作表，同時記錄工作表名稱與原始 Excel 列號。"""
    if not sheet_names:
        raise ValueError("至少需要指定一個工作表。")
    data_frames = []
    with pd.ExcelFile(file_path) as workbook:
        for sheet_name in sheet_names:
            if sheet_name not in workbook.sheet_names:
                raise ValueError(f"找不到工作表：{sheet_name}")
            sheet_data = pd.read_excel(workbook, sheet_name=sheet_name)
            normalized_columns = [str(name).strip() for name in sheet_data.columns]
            if len(normalized_columns) != len(set(normalized_columns)):
                raise ValueError(f"工作表 {sheet_name} 的欄名清理後有重複值。")
            sheet_data.columns = normalized_columns
            missing = [name for name in columns if name not in sheet_data.columns]
            if missing:
                raise ValueError(f"工作表 {sheet_name} 缺少欄位：{', '.join(missing)}")
            selected = sheet_data.loc[:, columns].copy()
            selected["_source_sheet"] = str(sheet_name)
            # 第一列為欄名，因此資料的第一列是 Excel 第 2 列。
            selected["_source_excel_row"] = range(2, len(selected) + 2)
            data_frames.append(selected)
    return pd.concat(data_frames, ignore_index=True)


def rename_columns(data, new_column_names):
    """重新命名資料欄位，保留來源欄位與呼叫端的 DataFrame。"""
    columns = [column for column in data.columns if column not in SOURCE_COLUMNS]
    if len(columns) != len(new_column_names):
        raise ValueError("原始欄位數與新欄位名稱數量不一致。")
    return data.rename(columns=dict(zip(columns, new_column_names))).copy()


def keep_valid_target(data, target_column="idt_target", valid_labels=VALID_IDT_LABELS):
    """寫回正規化 IDT 標籤，再剔除沒有合法標籤的資料。"""
    cleaned = data.copy()
    cleaned[target_column] = cleaned[target_column].map(normalize_idt_label)
    allowed = {normalize_idt_label(label) for label in valid_labels}
    return cleaned.loc[cleaned[target_column].isin(allowed)].reset_index(drop=True)


def clean_data(data, target_column="idt_target"):
    """標籤清洗 → 空描述剔除 → 相同標籤及內容精確去重。"""
    required = {target_column, "description", "directive"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"待清洗資料缺少欄位：{', '.join(missing)}")

    cleaned = keep_valid_target(data, target_column=target_column)
    invalid_targets = len(data) - len(cleaned)
    for column in ("description", "directive"):
        cleaned[column] = cleaned[column].map(_clean_text)
    nonempty = cleaned["description"].ne("")
    empty_descriptions = int((~nonempty).sum())
    cleaned = cleaned.loc[nonempty].copy()
    duplicates = int(cleaned.duplicated([target_column, "description", "directive"]).sum())
    cleaned = cleaned.drop_duplicates([target_column, "description", "directive"]).reset_index(drop=True)
    cleaned.attrs["cleaning_stats"] = {
        "input_records": len(data),
        "invalid_targets": invalid_targets,
        "empty_descriptions": empty_descriptions,
        "duplicates": duplicates,
        "output_records": len(cleaned),
    }
    return cleaned


def save_data_to_json(data, output_file, target_column="idt_target"):
    """輸出既有 IDT 標籤、待標註的情緒欄位與不含原文的來源識別碼。"""
    content_columns = [c for c in data.columns if c != target_column and c not in SOURCE_COLUMNS]
    records = []
    for index, row in data.iterrows():
        if all(column in data.columns for column in SOURCE_COLUMNS):
            source_id = f"{row['_source_sheet']}:{int(row['_source_excel_row'])}"
        else:
            source_id = f"row:{index + 1}"
        records.append({
            target_column: row[target_column],
            "emotion_target": "",
            "content": {column: _clean_text(row[column]) for column in content_columns},
            "record_id": source_id,
            "source_id": source_id,
            "is_augmented": False,
            "augmentation": None,
        })
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2, allow_nan=False)


def run(sheet_names, out_path, raw_file=RAW_FILE):
    """資料讀取 → 清洗 → JSON；訓練與測試工作表由呼叫端分開指定。"""
    data = load_and_merge_data(raw_file, sheet_names, COLUMNS)
    data = rename_columns(data, NEW_COLUMN_NAMES)
    data = clean_data(data, target_column=TARGET_COLUMN)
    save_data_to_json(data, out_path, target_column=TARGET_COLUMN)
    stats = data.attrs["cleaning_stats"]
    print(
        f"已輸出 {out_path}（{stats['input_records']} → {stats['output_records']} 筆；"
        f"非法 IDT {stats['invalid_targets']}、空描述 {stats['empty_descriptions']}、"
        f"重複 {stats['duplicates']}）"
    )
    return data


def main():
    run(TRAIN_SHEETS, "data/processed_data/train_data.json")
    run(TEST_SHEETS, "data/processed_data/test_data.json")


if __name__ == "__main__":
    main()
