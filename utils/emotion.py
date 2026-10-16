"""步驟 ④：撰寫者情緒標註。

用原始模型（不套 adapter）、溫度 0，從十種情緒擇一標註「撰寫者書寫當下的情緒」；
emotion_target 是模型標註，不是人工真值。輸出無效就在提示中說明原因後重試，
達到次數上限仍無效就中止，不用「中性」填補。逐筆存進度，中斷後可續跑。
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from utils.preprocessing import LABEL_COLUMN, TEXT_COLUMN
from utils.progress import Progress, fingerprint
from utils.prompts import EMOTION_PROMPT_ID, build_emotion_messages, parse_emotion_response

Generate = Callable[..., str]


class EmotionLabelError(RuntimeError):
    """某筆連續輸出無效；訊息會說明是哪一筆與最後一次的原因。"""


@dataclass(frozen=True)
class EmotionConfig:
    max_attempts: int = 3  # 首次 + 重試 2 次
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError("emotion max_attempts 必須是大於 0 的整數。")
        if self.max_attempts < 1:
            raise ValueError("emotion max_attempts 必須是大於 0 的整數。")
        if not isinstance(self.temperature, int | float) or not 0 <= self.temperature <= 2:
            raise ValueError("emotion temperature 必須介於 0 與 2 之間。")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_meta(self) -> dict:
        """寫進進度檔的設定；提示或情緒清單改了，舊進度就不能沿用。"""
        return {
            "stage": "emotion",
            "prompt": EMOTION_PROMPT_ID,
            "max_attempts": self.max_attempts,
            "temperature": self.temperature,
        }


@dataclass
class EmotionReport:
    records: int = 0
    reused: int = 0
    calls: int = 0
    label_counts: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "records": self.records,
            "reused_from_progress": self.reused,
            "model_calls": self.calls,
            "label_counts": dict(self.label_counts),
        }


def label_emotion(
    description: str, *, generate: Generate, config: EmotionConfig
) -> tuple[str, int]:
    """標註一段描述；回傳標籤與呼叫次數，全部失敗則拋出 EmotionLabelError。"""
    last_failure = None
    for attempt in range(config.max_attempts):
        messages = build_emotion_messages(description, last_failure)
        raw = generate(messages, temperature=config.temperature, seed=attempt)
        try:
            return parse_emotion_response(raw), attempt + 1
        except ValueError as exc:
            last_failure = str(exc)
    raise EmotionLabelError(f"連續 {config.max_attempts} 次輸出無效（最後一次：{last_failure}）")


def record_key(record: dict) -> str:
    """增生後的紀錄用 record_id；直接拿 01 的輸出標註時退回 source_id。"""
    return record.get("record_id") or record["source_id"]


def label_records(
    records: list[dict],
    *,
    generate: Generate,
    config: EmotionConfig,
    progress: Progress,
    on_record: Callable[[int, dict, str, int], None] | None = None,
) -> tuple[list[dict], EmotionReport]:
    """逐筆標註，回傳填好 emotion_target 的新紀錄清單與統計。"""
    report = EmotionReport(records=len(records))
    output: list[dict] = []
    for position, record in enumerate(records, start=1):
        key = record_key(record)
        fp = fingerprint(key, record[TEXT_COLUMN], record[LABEL_COLUMN])
        entry = progress.get(key, fp)
        if entry is None:
            try:
                label, calls = label_emotion(record[TEXT_COLUMN], generate=generate, config=config)
            except EmotionLabelError as exc:
                raise EmotionLabelError(f"{key}：{exc}；已完成的進度已保存，可續跑。") from exc
            progress.add(key, fp, emotion=label, calls=calls)
            report.calls += calls
        else:
            label, calls = entry["emotion"], 0
            report.reused += 1
        output.append({**record, "emotion_target": label})
        report.label_counts[label] += 1
        if on_record:
            on_record(position, record, label, calls)
    return output, report
