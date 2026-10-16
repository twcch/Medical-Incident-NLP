"""步驟 ②：依比例切分訓練／驗證／測試。

依人工 IDT 標籤分層，讓稀少的「系統」類別在三個集合的比例相近；
描述文字相同的紀錄視為同一組、不拆開，避免同一段文字同時出現在訓練與測試；
隨機順序只由 seed 決定，與訓練用的 seed 分開。先切分、之後才增生，
改寫版本一定與其原文同在訓練集。
"""

import math
import random
from collections import Counter, defaultdict

from utils.preprocessing import LABEL_COLUMN, TEXT_COLUMN

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIO = (0.7, 0.1, 0.2)
DEFAULT_SEED = 42


def validate_ratio(ratio) -> tuple[float, float, float]:
    """接受 0.7 0.1 0.2 或 7 1 2 這類寫法，三個值都要大於 0，回傳總和為 1 的比例。"""
    values = tuple(float(value) for value in ratio)
    if len(values) != 3:
        raise ValueError("切分比例必須是訓練、驗證、測試三個數值。")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("切分比例的三個數值都必須大於 0。")
    total = sum(values)
    return tuple(value / total for value in values)


def validate_seed(seed: int) -> int:
    """bool 在 Python 也是 int，需明確排除。"""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("切分種子必須是不小於 0 的整數。")
    return seed


def target_counts(total: int, ratio) -> tuple[int, int, int]:
    """四捨五入訓練與驗證的目標筆數，餘數全部給測試集。"""
    train = round(total * ratio[0])
    val = round(total * ratio[1])
    return train, val, max(total - train - val, 0)


def split_records(
    records: list[dict],
    ratio=DEFAULT_RATIO,
    seed: int = DEFAULT_SEED,
    *,
    group_key: str = TEXT_COLUMN,
    stratify_key: str = LABEL_COLUMN,
) -> dict[str, list[dict]]:
    """回傳 {"train": [...], "val": [...], "test": [...]}；各集合內維持原始順序。"""
    ratio = validate_ratio(ratio)
    validate_seed(seed)

    # 1. 描述相同的紀錄歸為一組，整組同進同出。
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record[group_key], []).append(record)

    # 2. 以組內多數標籤決定分層；同描述不同標籤極少見，取多數即可。
    strata: dict[str, list[list[dict]]] = defaultdict(list)
    for members in groups.values():
        label = Counter(member[stratify_key] for member in members).most_common(1)[0][0]
        strata[label].append(members)

    # 3. 各層獨立洗牌，依累計筆數先填滿訓練、再填驗證，剩餘歸測試。
    #    整組一起放，所以訓練與驗證最多只會比目標多出該組的筆數。
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for label in sorted(strata):
        members_list = strata[label]
        rng.shuffle(members_list)
        targets = target_counts(sum(len(members) for members in members_list), ratio)
        counts = [0, 0, 0]
        current = 0
        for members in members_list:
            while current < 2 and counts[current] >= targets[current]:
                current += 1
            splits[SPLIT_NAMES[current]].extend(members)
            counts[current] += len(members)

    # 4. 回復原始順序，輸出檔穩定、方便用 source_id 對照。
    order = {id(record): index for index, record in enumerate(records)}
    for name in SPLIT_NAMES:
        splits[name].sort(key=lambda record: order[id(record)])
    return splits


def shared_groups(splits: dict[str, list[dict]], group_key: str = TEXT_COLUMN) -> set[str]:
    """回傳同時出現在兩個以上集合的描述；切分正確時應為空集合。"""
    first_seen: dict[str, str] = {}
    shared: set[str] = set()
    for name, records in splits.items():
        for record in records:
            if first_seen.setdefault(record[group_key], name) != name:
                shared.add(record[group_key])
    return shared
