"""讀取月份工作表、清洗既有 IDT 標籤，並保留來源供資料切分稽核。"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.json_io import load_records, save_json
from src.prompts import VALID_IDT_LABELS

# 預設為先前由本模組從 Excel 清洗輸出的 JSON；仍可用 --raw-file 指定原始 Excel。
RAW_FILE = "data/raw_data/data.json"
# Excel 的人工 IDT、事件描述與批示依原欄位順序對應到下游 JSON 欄位。
COLUMNS = ["IDT分析(個人,系統)", "事件描述", "批示"]
NEW_COLUMN_NAMES = ["idt_target", "description", "directive"]
TARGET_COLUMN = NEW_COLUMN_NAMES[0]
# 預設納入 112.01–114.12 全部月份；訓練／驗證／測試改依比例切分，不再以月份區分。
ALL_SHEETS = [f"{year}.{month:02d}" for year in (112, 113, 114) for month in range(1, 13)]
# 來源欄位僅供追蹤資料與切分稽核，不放入提供給模型的 content。
SOURCE_COLUMNS = ["_source_sheet", "_source_excel_row"]


def _clean_text(value) -> str:
    """僅整理空值、前後空白與換行，不改動醫療敘述內容。"""
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_idt_label(value) -> str:
    """正規化既有標籤的全形字元與空白；不推測或新增標籤。"""
    text = unicodedata.normalize("NFKC", _clean_text(value))
    return re.sub(r"\s+", "", text)


def _validate_names(names, name: str) -> list[str]:
    """檢查欄名或工作表名稱，避免重複選取後產生含糊的來源紀錄。"""
    if isinstance(names, str):
        raise ValueError(f"{name} 必須是名稱序列，不可直接傳入字串。")
    try:
        names = list(names)
    except TypeError:
        raise ValueError(f"{name} 必須是名稱序列。") from None
    if not names or any(not isinstance(item, str) or not item.strip() for item in names):
        raise ValueError(f"{name} 必須包含至少一個非空白名稱。")
    if len(names) != len(set(names)):
        raise ValueError(f"{name} 不可包含重複名稱。")
    return names


def load_and_merge_data(file_path: str | Path, sheet_names, columns) -> pd.DataFrame:
    """讀取指定工作表，同時記錄工作表名稱與原始 Excel 列號。"""
    sheet_names = _validate_names(sheet_names, "工作表名稱")
    columns = _validate_names(columns, "資料欄位")
    if set(columns).intersection(SOURCE_COLUMNS):
        raise ValueError("資料欄位不可使用保留的來源欄名。")
    data_frames = []
    with pd.ExcelFile(file_path) as workbook:
        for sheet_name in sheet_names:
            if sheet_name not in workbook.sheet_names:
                raise ValueError(f"找不到工作表：{sheet_name}")
            sheet_data = pd.read_excel(workbook, sheet_name=sheet_name)
            # 只整理表頭的前後空白；先確認無重名，再依指定欄位選取資料。
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


def load_json_records(file_path: str | Path, sheet_names) -> pd.DataFrame:
    """依 source_id 的工作表名稱，從清洗後的 JSON 選出指定月份的紀錄。

    JSON 已是下游欄位，不需重新命名；來源工作表與列號由 source_id 還原，
    讓後續清洗、去重與輸出沿用 Excel 路徑的同一條流程。
    """
    sheet_names = _validate_names(sheet_names, "工作表名稱")
    # 依呼叫端指定的月份順序輸出，與逐張讀取 Excel 時的順序一致。
    by_sheet = {name: [] for name in sheet_names}
    for index, record in enumerate(load_records(file_path), start=1):
        if record.get("is_augmented"):
            raise ValueError(f"第 {index} 筆是增生資料，原始資料檔不可含改寫版本。")
        content = record.get("content")
        source_id = record.get("source_id")
        if not isinstance(content, dict) or not isinstance(source_id, str):
            raise ValueError(f"第 {index} 筆缺少 content 物件或 source_id。")
        sheet, _, row = source_id.rpartition(":")
        if not sheet or not row.isdecimal():
            raise ValueError(f"第 {index} 筆的 source_id 必須是「工作表:Excel 列號」。")
        if sheet in by_sheet:
            by_sheet[sheet].append(
                {
                    TARGET_COLUMN: record.get(TARGET_COLUMN),
                    "description": content.get("description"),
                    "directive": content.get("directive"),
                    "_source_sheet": sheet,
                    "_source_excel_row": int(row),
                }
            )
    # 清洗後整張工作表可能沒有合法紀錄；此時與找不到工作表一樣明確中止，不靜默讀成空資料。
    missing = [name for name, rows in by_sheet.items() if not rows]
    if missing:
        raise ValueError(f"資料檔中找不到工作表的紀錄：{', '.join(missing)}")
    return pd.DataFrame([row for rows in by_sheet.values() for row in rows])


def rename_columns(data: pd.DataFrame, new_column_names) -> pd.DataFrame:
    """重新命名資料欄位，保留來源欄位與呼叫端的 DataFrame。"""
    new_column_names = _validate_names(new_column_names, "新欄位名稱")
    if not data.columns.is_unique:
        raise ValueError("原始資料不可包含重複欄名。")
    if set(new_column_names).intersection(SOURCE_COLUMNS):
        raise ValueError("新欄位名稱不可使用保留的來源欄名。")
    columns = [column for column in data.columns if column not in SOURCE_COLUMNS]
    if len(columns) != len(new_column_names):
        raise ValueError("原始欄位數與新欄位名稱數量不一致。")
    return data.rename(columns=dict(zip(columns, new_column_names))).copy()


def keep_valid_target(
    data: pd.DataFrame,
    target_column: str = "idt_target",
    valid_labels=VALID_IDT_LABELS,
) -> pd.DataFrame:
    """寫回正規化 IDT 標籤，再剔除沒有合法標籤的資料。"""
    if target_column not in data.columns:
        raise ValueError(f"待清洗資料缺少欄位：{target_column}")
    cleaned = data.copy()
    cleaned[target_column] = cleaned[target_column].map(normalize_idt_label)
    allowed = {normalize_idt_label(label) for label in valid_labels}
    return cleaned.loc[cleaned[target_column].isin(allowed)].reset_index(drop=True)


def clean_data(data: pd.DataFrame, target_column: str = "idt_target") -> pd.DataFrame:
    """標籤清洗 → 空描述剔除 → 相同標籤及內容精確去重。"""
    required = {target_column, "description", "directive"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"待清洗資料缺少欄位：{', '.join(missing)}")

    cleaned = keep_valid_target(data, target_column=target_column)
    # 每個剔除數量都依前一階段的剩餘資料計算，同一筆不會重複計入統計。
    invalid_targets = len(data) - len(cleaned)
    for column in ("description", "directive"):
        cleaned[column] = cleaned[column].map(_clean_text)
    nonempty = cleaned["description"].ne("")
    empty_descriptions = int((~nonempty).sum())
    cleaned = cleaned.loc[nonempty].copy()
    # IDT、描述與批示三者都相同才去重，並保留第一次出現的來源列。
    duplicates = int(cleaned.duplicated([target_column, "description", "directive"]).sum())
    cleaned = cleaned.drop_duplicates([target_column, "description", "directive"]).reset_index(
        drop=True
    )
    # attrs 提供 run() 顯示清洗摘要，不會變成紀錄內容或輸出 JSON 欄位。
    cleaned.attrs["cleaning_stats"] = {
        "input_records": len(data),
        "invalid_targets": invalid_targets,
        "empty_descriptions": empty_descriptions,
        "duplicates": duplicates,
        "output_records": len(cleaned),
    }
    return cleaned


def save_data_to_json(
    data: pd.DataFrame,
    output_file: str | Path,
    target_column: str = "idt_target",
) -> None:
    """輸出既有 IDT 標籤、待標註的情緒欄位與不含原文的來源識別碼。"""
    if target_column not in data.columns:
        raise ValueError(f"輸出資料缺少欄位：{target_column}")
    if not data.columns.is_unique:
        raise ValueError("輸出資料不可包含重複欄名。")
    source_columns = set(SOURCE_COLUMNS).intersection(data.columns)
    if source_columns and source_columns != set(SOURCE_COLUMNS):
        raise ValueError("來源工作表與 Excel 列號必須同時存在。")
    content_columns = [
        column
        for column in data.columns
        if column != target_column and column not in SOURCE_COLUMNS
    ]
    records = []
    # 以輸出順序編號，避免呼叫端保留字串或不連續索引時無法建立 fallback ID。
    for position, (_, row) in enumerate(data.iterrows(), start=1):
        if source_columns:
            source_id = f"{row['_source_sheet']}:{int(row['_source_excel_row'])}"
        else:
            source_id = f"row:{position}"
        # 原始紀錄的兩個 ID 相同；增生後 record_id 獨立，source_id 仍指向此來源。
        records.append(
            {
                target_column: row[target_column],
                "emotion_target": "",
                "content": {column: _clean_text(row[column]) for column in content_columns},
                "record_id": source_id,
                "source_id": source_id,
                "is_augmented": False,
                "augmentation": None,
            }
        )
    save_json(records, output_file)


def run(sheet_names, out_path: str | Path, raw_file: str | Path = RAW_FILE) -> pd.DataFrame:
    """資料讀取 → 清洗 → JSON；訓練／驗證／測試的切分交給下一階段的 data_split。

    raw_file 副檔名為 .json 時讀取清洗後的紀錄，其餘視為原始 Excel。
    """
    if Path(raw_file).suffix.lower() == ".json":
        data = load_json_records(raw_file, sheet_names)
    else:
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
    """讀取預設全部月份並清洗，保留來源資訊；訓練／驗證／測試的切分由 data_split 負責。"""
    run(ALL_SHEETS, "data/processed_data/cleaned_data.json")


if __name__ == "__main__":
    main()
