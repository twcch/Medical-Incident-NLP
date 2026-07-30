import json
import os

import pandas as pd

RAW_FILE = "data/raw_data/raw data去辨識0612.xlsx"
COLUMNS = ["IDT分析(個人,系統)", "事件描述", "批示"]
NEW_COLUMN_NAMES = ["idt_target", "description", "directive"]
TARGET_COLUMN = NEW_COLUMN_NAMES[0]
VALID_IDT_LABELS = ["個人", "系統"]  # idt_target 只允許這兩類，其餘整筆刪除
TRAIN_SHEETS = ["112.01", "112.02", "112.03", "112.04", "112.05", "112.06", "112.07", "112.08", "112.09", "112.10", "112.11", "112.12"]
TEST_SHEETS = ["113.01", "113.02"]


def load_and_merge_data(file_path, sheet_names, columns):
    """
    Load the given sheets from an Excel file, keep only the required columns,
    and merge them into one DataFrame.

    Parameters:
    - file_path: str, path to the Excel file.
    - sheet_names: list of str, names of the sheets to load and merge.
    - columns: list of str, names of the columns to keep.

    Returns:
    - DataFrame containing the merged data from all sheets.
    """
    data_frames = []
    for sheet_name in sheet_names:
        sheet_data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns)
        data_frames.append(sheet_data[columns])

    merged_data = pd.concat(data_frames, ignore_index=True)

    return merged_data


def rename_columns(data, new_column_names):
    """
    Rename the columns of the given DataFrame.

    Parameters:
    - data: DataFrame, the data whose columns need to be renamed.
    - new_column_names: list of str, new names for the columns.

    Returns:
    - DataFrame with renamed columns.
    """
    data.columns = new_column_names
    return data


def keep_valid_target(data, target_column="idt_target", valid_labels=VALID_IDT_LABELS):
    """
    Keep only rows whose target is one of the valid labels; drop everything else.
    This covers null/blank as well as out-of-scope junk values (ward codes such as
    "1905-02", "OR", "HDR", stray numbers, etc.).

    Parameters:
    - data: DataFrame, the data to clean.
    - target_column: str, name of the target column.
    - valid_labels: iterable of allowed target values.

    Returns:
    - DataFrame with only valid-target rows and the index reset.
    """
    normalized = data[target_column].astype(str).str.strip()
    keep = normalized.isin(valid_labels)
    return data[keep].reset_index(drop=True)


def save_data_to_json(data, output_file, target_column="idt_target"):
    """
    Save the DataFrame to a JSON file. Each record has three fields:
    - <target_column>: the value of the target column.
    - emotion_target: an empty placeholder for later model annotation.
    - content: a nested object holding the remaining columns.

    Parameters:
    - data: DataFrame, the data to save.
    - output_file: str, path to the output JSON file.
    - target_column: str, name of the column used as the target (also the JSON key).
    """
    content_columns = [c for c in data.columns if c != target_column]
    # description / directive 沒有值時填空字串 ""
    contents = data[content_columns].fillna("").to_dict(orient="records")
    # target 為空時保留 null（正常流程已先用 keep_valid_target 篩掉非法 target）
    targets = data[target_column].where(data[target_column].notna(), None)

    records = [
        # emotion_target 預留空字串，供後續 model 標注填入
        {target_column: target, "emotion_target": "", "content": content}
        for target, content in zip(targets, contents)
    ]

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def run(sheet_names, out_path):
    """
    Run the preprocessing pipeline for a single dataset (one set of sheets):
    merge -> rename -> drop null target -> save as JSON.
    """
    data = load_and_merge_data(RAW_FILE, sheet_names, COLUMNS)
    data = rename_columns(data, NEW_COLUMN_NAMES)
    data = keep_valid_target(data, target_column=TARGET_COLUMN)
    save_data_to_json(data, out_path, target_column=TARGET_COLUMN)
    print(f"已輸出 {out_path}（{len(data)} 筆）")
    return data


def main():
    run(TRAIN_SHEETS, "data/processed_data/train_data.json")
    run(TEST_SHEETS, "data/processed_data/test_data.json")


if __name__ == "__main__":
    main()
