"""資料檔讀寫：CSV 讀成紀錄清單、JSON 原子寫入與讀取。"""

import csv
import json
import os
import tempfile
from pathlib import Path


def read_csv_records(path: str | Path, encoding: str = "utf-8-sig") -> list[dict]:
    """讀取 CSV 為 dict 清單；每個欄位都是字串，空格為空字串。

    utf-8-sig 會自動去掉 Excel 另存「CSV UTF-8」時加在檔頭的 BOM；
    newline="" 讓 csv 模組正確處理欄位內的換行（批示常有多行）。
    """
    with open(path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} 是空檔案，沒有標題列。")
        return list(reader)


def save_json(data, path: str | Path) -> Path:
    """先寫到同目錄的暫存檔再改名，中途失敗不會留下寫到一半的檔案。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
