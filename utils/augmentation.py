"""步驟 ③：訓練集增生。

只對訓練集進行，每筆產生 n 個改寫、每個改寫單獨呼叫模型一次，改寫沿用原本的人工
IDT 標籤 y。改寫必須是合法 JSON、非空、且與原文及所有已有描述都不同；不合格就
重試並在提示中說明上次不可用的原因，每個改寫最多嘗試 max_attempts 次。
湊不滿時 keep-partial 保留已取得的版本並記錄缺額，abort 則立即中止。
"""

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from utils.preprocessing import LABEL_COLUMN, TEXT_COLUMN
from utils.progress import Progress, fingerprint
from utils.prompts import AUGMENT_PROMPT_ID, build_augment_messages, parse_augment_response

# generate(messages, *, temperature, seed) -> 模型輸出的文字
Generate = Callable[..., str]

ON_FAILURE_CHOICES = ("keep-partial", "abort")


def derive_seed(base: int, *parts) -> int:
    """由基礎 seed 與紀錄、改寫序號、嘗試次數推出每次呼叫的 seed，重跑可重現、重試會不同。"""
    joined = "\x1f".join(str(part) for part in (base, *parts))
    return int.from_bytes(hashlib.sha1(joined.encode("utf-8")).digest()[:4], "big")


def comparable(text: str) -> str:
    """比較是否相同時忽略所有空白，避免只差換行或空格的「改寫」過關。"""
    return "".join(text.split())


@dataclass(frozen=True)
class AugmentConfig:
    n: int = 2
    max_attempts: int = 4
    temperature: float = 0.8
    on_failure: str = "keep-partial"

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError("augment n 必須是不小於 0 的整數，0 表示不增生。")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError("augment max_attempts 必須是大於 0 的整數。")
        if self.max_attempts < 1:
            raise ValueError("augment max_attempts 必須是大於 0 的整數。")
        if not isinstance(self.temperature, int | float) or not 0 <= self.temperature <= 2:
            raise ValueError("augment temperature 必須介於 0 與 2 之間。")
        if self.on_failure not in ON_FAILURE_CHOICES:
            raise ValueError(f"augment on_failure 必須是 {'、'.join(ON_FAILURE_CHOICES)} 之一。")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_meta(self) -> dict:
        """寫進進度檔的設定；on_failure 不影響輸出內容，不列入。提示改了舊進度就不能沿用。"""
        return {
            "stage": "augment",
            "prompt": AUGMENT_PROMPT_ID,
            "n": self.n,
            "max_attempts": self.max_attempts,
            "temperature": self.temperature,
        }


@dataclass
class AugmentReport:
    records: int = 0
    reused: int = 0
    rewrites_requested: int = 0
    rewrites_obtained: int = 0
    calls: int = 0
    shortfalls: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "records": self.records,
            "reused_from_progress": self.reused,
            "rewrites_requested": self.rewrites_requested,
            "rewrites_obtained": self.rewrites_obtained,
            "model_calls": self.calls,
            "records_short": len(self.shortfalls),
            "shortfalls": self.shortfalls,
        }


def generate_rewrites(
    description: str,
    *,
    generate: Generate,
    config: AugmentConfig,
    taken: set[str],
    seed: int,
    seed_parts: tuple = (),
    existing: list[str] | None = None,
    label: str | None = None,
) -> tuple[list[str], int]:
    """為一段描述補足到 n 個改寫；回傳改寫清單與本次呼叫模型的次數。

    人工 IDT 標籤與該筆已取得的改寫會一起送給模型（要求不改變分類依據、避免重複）。
    """
    rewrites = list(existing or [])
    calls = 0
    for index in range(len(rewrites), config.n):
        last_failure = None
        for attempt in range(config.max_attempts):
            messages = build_augment_messages(
                description, last_failure, label=label, existing=rewrites
            )
            raw = generate(
                messages,
                temperature=config.temperature,
                seed=derive_seed(seed, *seed_parts, index, attempt),
            )
            calls += 1
            try:
                candidate = parse_augment_response(raw)
                key = comparable(candidate)
                if key == comparable(description):
                    raise ValueError("與原文相同")
                if key in taken or key in {comparable(r) for r in rewrites}:
                    raise ValueError("與既有的描述重複")
            except ValueError as exc:
                last_failure = str(exc)
                continue
            rewrites.append(candidate)
            break
    return rewrites, calls


def _original_record(record: dict) -> dict:
    return {"record_id": record["source_id"], "augmented": False, **record}


def _rewrite_record(record: dict, index: int, text: str) -> dict:
    """改寫沿用原紀錄的人工標籤與批示，只換描述；emotion_target 留給 ④ 填。"""
    return {
        "record_id": f"{record['source_id']}#aug{index}",
        "augmented": True,
        **record,
        TEXT_COLUMN: text,
        "emotion_target": None,
    }


def augment_records(
    records: list[dict],
    *,
    generate: Generate,
    config: AugmentConfig,
    progress: Progress,
    seed: int,
    on_record: Callable[[int, dict, list[str], int], None] | None = None,
) -> tuple[list[dict], AugmentReport]:
    """逐筆增生，回傳「原文 + 改寫」的紀錄清單與統計。

    已在進度檔且內容雜湊相符的紀錄直接沿用；abort 模式下湊不滿的紀錄會再補生成。
    """
    taken = {comparable(record[TEXT_COLUMN]) for record in records}
    report = AugmentReport(records=len(records), rewrites_requested=config.n * len(records))
    output: list[dict] = []
    for position, record in enumerate(records, start=1):
        source_id = record["source_id"]
        fp = fingerprint(source_id, record[TEXT_COLUMN], record[LABEL_COLUMN])
        entry = progress.get(source_id, fp)
        existing = list(entry["rewrites"]) if entry else []
        complete = entry is not None and (
            len(existing) >= config.n or config.on_failure == "keep-partial"
        )
        if complete:
            rewrites, calls = existing, 0
            report.reused += 1
        else:
            rewrites, calls = generate_rewrites(
                record[TEXT_COLUMN],
                generate=generate,
                config=config,
                taken=taken,
                seed=seed,
                seed_parts=(source_id,),
                existing=existing,
                label=record[LABEL_COLUMN],
            )
            progress.add(source_id, fp, rewrites=rewrites, calls=calls)
            report.calls += calls
        taken.update(comparable(text) for text in rewrites)
        report.rewrites_obtained += len(rewrites)
        if len(rewrites) < config.n:
            report.shortfalls[source_id] = config.n - len(rewrites)
            if config.on_failure == "abort":
                raise RuntimeError(
                    f"{source_id} 只取得 {len(rewrites)}/{config.n} 個改寫，"
                    "依 on_failure=abort 中止；已完成的進度已保存。"
                )
        output.append(_original_record(record))
        output.extend(_rewrite_record(record, i, text) for i, text in enumerate(rewrites, 1))
        if on_record:
            on_record(position, record, rewrites, calls)
    return output, report
