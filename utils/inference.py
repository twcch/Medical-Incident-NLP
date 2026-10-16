"""步驟 ⑥⑦⑨：對測試集逐筆做 IDT 預測、情緒預測與情緒評審。

三個階段都是溫度 0；輸出無效就把原因寫進提示重試，用完次數仍無效就記 None，
不猜標籤也不填預設值，評估時 None 算錯誤。每筆結果都寫進進度檔，中斷後續跑；
記成 None 的那幾筆重跑時會再試一次，成功的直接沿用。
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from utils.emotion import record_key
from utils.preprocessing import TEXT_COLUMN
from utils.progress import Progress, fingerprint
from utils.prompts import (
    build_emotion_messages,
    build_idt_messages,
    build_judge_messages,
    parse_emotion_response,
    parse_idt_response,
    parse_judge_response,
)

Generate = Callable[..., str]
IDT_PREDICTION = "idt_pred"
EMOTION_PREDICTION = "emotion_pred"
EMOTION_JUDGE = "emotion_judge"


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3  # 首次 + 重試 2 次
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError("max_attempts 必須是大於 0 的整數。")
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必須是大於 0 的整數。")
        if not isinstance(self.temperature, int | float) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature 必須介於 0 與 2 之間。")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StageReport:
    stage: str
    records: int = 0
    reused: int = 0
    calls: int = 0
    failed: list[dict] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "records": self.records,
            "reused_from_progress": self.reused,
            "model_calls": self.calls,
            "failed": self.failed,
            "counts": dict(self.counts),
        }


def predict_with_retries(
    build_messages: Callable[[str | None], list[dict]],
    parse: Callable[[str], object],
    *,
    generate: Generate,
    config: RetryConfig,
) -> tuple[object | None, int, str | None]:
    """回傳 (結果或 None, 呼叫次數, 最後一次的失敗原因)；溫度 0 下重試會改變提示。"""
    last_failure = None
    for attempt in range(config.max_attempts):
        raw = generate(build_messages(last_failure), temperature=config.temperature, seed=attempt)
        try:
            return parse(raw), attempt + 1, None
        except ValueError as exc:
            last_failure = str(exc)
    return None, config.max_attempts, last_failure


def apply_stage(
    records: list[dict],
    *,
    stage: str,
    field_name: str,
    progress: Progress,
    predict: Callable[[dict], tuple[object | None, int, str | None]],
    fingerprint_of: Callable[[dict], str],
    count_of: Callable[[object], str] | None = None,
    on_record: Callable[[int, dict, object | None, int], None] | None = None,
) -> tuple[list[dict], StageReport]:
    """逐筆套用 predict，把結果放進 field_name；進度檔裡成功的沿用、None 的重試。"""
    report = StageReport(stage=stage, records=len(records))
    output: list[dict] = []
    for position, record in enumerate(records, start=1):
        key = record_key(record)
        fp = fingerprint_of(record)
        entry = progress.get(key, fp)
        if entry is not None and entry.get("value") is not None:
            value, calls = entry["value"], 0
            report.reused += 1
        else:
            value, calls, failure = predict(record)
            progress.add(key, fp, value=value, calls=calls, failure=failure)
            report.calls += calls
            if value is None:
                report.failed.append({"key": key, "reason": failure})
        if value is not None:
            report.counts[count_of(value) if count_of else str(value)] += 1
        output.append({**record, field_name: value})
        if on_record:
            on_record(position, record, value, calls)
    return output, report


def predict_idt(records, *, generate, config, progress, on_record=None):
    """⑥ 用套了 adapter 的模型預測 IDT；只接受「個人」或「系統」。"""

    def predict(record):
        return predict_with_retries(
            lambda failure: build_idt_messages(record[TEXT_COLUMN], failure),
            parse_idt_response,
            generate=generate,
            config=config,
        )

    return apply_stage(
        records,
        stage="idt",
        field_name=IDT_PREDICTION,
        progress=progress,
        predict=predict,
        fingerprint_of=lambda record: fingerprint(record_key(record), record[TEXT_COLUMN]),
        on_record=on_record,
    )


def predict_emotion(records, *, generate, config, progress, on_record=None):
    """⑦ 用原始模型（不套 adapter）預測撰寫者情緒，提示與訓練集標註相同。"""

    def predict(record):
        return predict_with_retries(
            lambda failure: build_emotion_messages(record[TEXT_COLUMN], failure),
            parse_emotion_response,
            generate=generate,
            config=config,
        )

    return apply_stage(
        records,
        stage="emotion",
        field_name=EMOTION_PREDICTION,
        progress=progress,
        predict=predict,
        fingerprint_of=lambda record: fingerprint(record_key(record), record[TEXT_COLUMN]),
        on_record=on_record,
    )


def judge_emotion(
    records,
    *,
    generate,
    config,
    progress,
    emotion_field=EMOTION_PREDICTION,
    model_name: str | None = None,
    on_record=None,
):
    """⑨ 評審模型只看描述與被評的情緒標籤；標籤為空的紀錄不評、記 None。"""

    def predict(record):
        emotion = record.get(emotion_field)
        if emotion is None:
            return None, 0, f"{emotion_field} 為空，無法評審"
        value, calls, failure = predict_with_retries(
            lambda failure: build_judge_messages(record[TEXT_COLUMN], emotion, failure),
            parse_judge_response,
            generate=generate,
            config=config,
        )
        if value is not None and model_name:
            value = {**value, "model": model_name}  # 分數跟著評審模型走，彙整時不靠命令列參數
        return value, calls, failure

    return apply_stage(
        records,
        stage="judge",
        field_name=EMOTION_JUDGE,
        progress=progress,
        predict=predict,
        fingerprint_of=lambda record: fingerprint(
            record_key(record), record[TEXT_COLUMN], record.get(emotion_field)
        ),
        count_of=lambda value: f"總分≈{round(value['overall'])}",
        on_record=on_record,
    )
