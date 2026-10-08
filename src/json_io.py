"""統一資料 JSON 的格式驗證與寫入，避免失敗時損壞既有實驗結果。"""

import json
import math
import os
import tempfile
from pathlib import Path


def _reject_nonfinite(value: str) -> None:
    """Python 預設接受 NaN／Infinity，但它們不是合法的 JSON 數值。"""
    raise ValueError(f"JSON 不允許 {value}。")


def _parse_finite_float(value: str) -> float:
    """也檢查指數溢位；例如合法數字 1e400 會被 Python 轉成 inf。"""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON 不允許非有限數值。")
    return number


def parse_json(raw: str) -> object:
    """資料檔與模型輸出共用標準 JSON 解析，拒絕非有限數值。"""
    return json.loads(raw, parse_constant=_reject_nonfinite, parse_float=_parse_finite_float)


def load_records(path: str | Path) -> list[dict]:
    """讀取紀錄陣列；在進入模型流程前回報資料格式與錯誤筆數。"""
    records = parse_json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("資料 JSON 最外層必須是陣列。")
    # 此處只保證通用紀錄結構；description、標籤等任務欄位由各流程自行驗證。
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index} 筆資料必須是物件。")
    return records


def save_json(data: object, path: str | Path) -> None:
    """完整序列化後才原子替換檔案；適用於紀錄、評估摘要及訓練資訊。"""
    # 先驗證可序列化性；遇到非法值時，不建立目錄也不碰舊輸出。
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        # 暫存檔必須與目標在同一個檔案系統，replace 才能原子完成。
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
        # 檔案已完整寫入並關閉後才替換，讀取者會看到完整的舊版或新版內容。
        os.replace(temporary_path, destination)
    finally:
        # 寫入或替換失敗時也清除暫存檔，不留下不完整的實驗資料。
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
