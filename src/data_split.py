"""依比例切分清洗後的資料為訓練／驗證／測試；切分必須發生在增生之前。"""

import math
import os
import random
from collections import Counter
from numbers import Real
from pathlib import Path

from src.json_io import load_records, save_json
from src.prompts import VALID_IDT_LABELS

DEFAULT_RATIO = (7, 1, 2)
DEFAULT_SEED = 42
SPLIT_NAMES = ("train", "val", "test")
TEXT_FIELD = "description"
IDT_TARGET = "idt_target"


def validate_ratio(ratio) -> tuple[float, float, float]:
    """比例依序為訓練、驗證、測試；只看相對大小，不要求加總為 10 或 1。"""
    if isinstance(ratio, str):
        raise ValueError("切分比例必須是三個數值，不可直接傳入字串。")
    try:
        values = tuple(ratio)
    except TypeError:
        raise ValueError("切分比例必須是三個數值。") from None
    if len(values) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError("切分比例必須是訓練、驗證、測試三個大於 0 的有限數值。")
    return tuple(float(value) for value in values)


def validate_seed(seed) -> int:
    """bool 在 Python 也是 int，需明確排除。"""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("切分 seed 必須是非負整數。")
    return seed


def format_ratio(ratio) -> str:
    """以 7:1:2 的形式呈現，供日誌與設定預覽使用。"""
    return ":".join(f"{value:g}" for value in validate_ratio(ratio))


def split_records(records: list, ratio=DEFAULT_RATIO, *, seed: int = DEFAULT_SEED) -> dict:
    """分層並依描述分組切分，回傳 train／val／test，各集合保留原始順序。

    以人工 IDT 標籤分層，讓稀少的類別在三個集合都有代表；描述相同的紀錄視為
    同一組、不拆開，避免同一段事件文字同時出現在訓練與測試。
    """
    weights = validate_ratio(ratio)
    validate_seed(seed)
    if not isinstance(records, list):
        raise ValueError("資料 JSON 最外層必須是陣列。")

    groups: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        number = position + 1
        if not isinstance(record, dict) or not isinstance(record.get("content"), dict):
            raise ValueError(f"第 {number} 筆的 content 必須是物件。")
        if record.get("is_augmented"):
            # 增生版本若被切到不同集合，改寫文字會與其原文跨集合，造成資料洩漏。
            raise ValueError(f"第 {number} 筆是增生資料；請在增生之前切分。")
        text = record["content"].get(TEXT_FIELD)
        label = record.get(IDT_TARGET)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"第 {number} 筆的 description 必須是非空字串。")
        if not isinstance(label, str) or label.strip() not in VALID_IDT_LABELS:
            raise ValueError(f"第 {number} 筆沒有合法的人工 IDT 標籤。")
        groups.setdefault(text.strip(), []).append(position)

    # 同組若出現互相衝突的標籤，整組跟隨第一筆的標籤分層，仍不拆開。
    strata: dict[str, list[list[int]]] = {}
    for positions in groups.values():
        label = records[positions[0]][IDT_TARGET].strip()
        strata.setdefault(label, []).append(positions)

    rng = random.Random(seed)
    chosen: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    total_weight = sum(weights)
    for label in sorted(strata):
        stratum = strata[label]
        rng.shuffle(stratum)
        size = sum(len(positions) for positions in stratum)
        targets = [size * weight / total_weight for weight in weights]
        counts = [0, 0, 0]
        for positions in stratum:
            # 指派給目前離目標筆數缺口最大的集合，使每個標籤內的比例都貼近設定值。
            deficits = [target - count for target, count in zip(targets, counts)]
            index = deficits.index(max(deficits))
            chosen[SPLIT_NAMES[index]].extend(positions)
            counts[index] += len(positions)

    empty = [name for name in SPLIT_NAMES if not chosen[name]]
    if empty:
        raise ValueError(f"資料量太少，依比例切分後 {'、'.join(empty)} 集合沒有任何紀錄。")
    return {name: [records[position] for position in sorted(chosen[name])] for name in SPLIT_NAMES}


def run(
    in_path: str | Path,
    train_out: str | Path,
    val_out: str | Path,
    test_out: str | Path,
    *,
    ratio=DEFAULT_RATIO,
    seed: int = DEFAULT_SEED,
) -> dict:
    """讀取清洗後的資料，切分成功後才寫入三個 JSON。"""
    outputs = {"train": train_out, "val": val_out, "test": test_out}
    destinations = {os.path.realpath(path) for path in outputs.values()}
    if len(destinations) != len(outputs) or os.path.realpath(in_path) in destinations:
        raise ValueError("切分的三個輸出路徑必須互不相同，且不可覆寫輸入資料。")
    records = load_records(in_path)
    splits = split_records(records, ratio, seed=seed)
    for name, path in outputs.items():
        save_json(splits[name], path)
    print(f"已切分 {in_path}（{len(records)} 筆；比例 {format_ratio(ratio)}；seed {seed}）")
    for name, path in outputs.items():
        labels = Counter(record[IDT_TARGET].strip() for record in splits[name])
        detail = "／".join(f"{label} {labels[label]}" for label in VALID_IDT_LABELS)
        print(f"  {name}: {len(splits[name])} 筆（{detail}）→ {path}")
    return splits
