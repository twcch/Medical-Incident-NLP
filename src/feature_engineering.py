"""使用 Gemma 標註事件描述的撰寫者情緒；此標註不是人工真實標籤。"""

from collections.abc import Callable
from pathlib import Path

if __package__:
    from .config import RuntimeConfig
    from .json_io import load_records
    from .json_io import save_json as write_json
    from .llm import ModelSession, parse_json_object
    from .prompts import EMOTION_LABELS, EMOTION_SYSTEM_PROMPT
else:
    from config import RuntimeConfig
    from json_io import load_records
    from json_io import save_json as write_json
    from llm import ModelSession, parse_json_object
    from prompts import EMOTION_LABELS, EMOTION_SYSTEM_PROMPT

TEXT_FIELD = "description"
EMOTION_FIELD = "emotion_target"  # 保留既有資料欄位，內容為 Gemma 產生的標註


def description_text(record: dict, index: int | None = None) -> str:
    """拒絕空白或非文字輸入，避免以中性掩蓋資料或模型錯誤。"""
    location = f"第 {index} 筆" if index is not None else "資料"
    if not isinstance(record, dict) or not isinstance(record.get("content"), dict):
        raise ValueError(f"{location} 必須包含 content 物件。")
    text = record["content"].get(TEXT_FIELD)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{location} 的 content.description 必須是非空白文字。")
    return text.strip()


def parse_label_response(raw: str, field: str, valid_labels) -> str:
    """只接受合法單一標籤，或僅含指定欄位的 JSON 物件。

    IDT 與情緒共用此解析器；呼叫端以 field 及 valid_labels 指定各自的格式。
    """
    if not isinstance(raw, str):
        raise ValueError("模型標籤輸出必須是文字。")
    cleaned = raw.strip()
    if cleaned in valid_labels:
        return cleaned
    result = parse_json_object(cleaned)
    if set(result) != {field}:
        raise ValueError(f"模型必須只回傳 {field} 欄位。")
    label = result[field]
    if not isinstance(label, str) or label not in valid_labels:
        raise ValueError(f"模型輸出無效的 {field} 標籤。")
    return label


def annotate_emotion(
    text: str,
    max_retries: int = 2,
    *,
    generator: Callable | None = None,
    model_name: str | None = None,
    config: RuntimeConfig | None = None,
) -> str:
    """重試格式錯誤的輸出；仍失敗時中止，不替換為中性。

    max_retries 不包含第一次生成；generator 可注入假模型或已載入的 session。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("情緒標註的輸入必須是非空白文字。")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries 必須是非負整數。")
    config = config or RuntimeConfig()
    model_name = model_name or config.gemma_model
    if generator is None:
        with ModelSession(model_name, config) as session:
            return annotate_emotion(
                text,
                max_retries,
                generator=session.generate,
                model_name=model_name,
                config=config,
            )
    last_error = None
    for _ in range(max_retries + 1):
        try:
            raw = generator(
                EMOTION_SYSTEM_PROMPT,
                text.strip(),
                temperature=0.0,
                max_new_tokens=config.max_new_tokens,
            )
            return parse_label_response(raw, "emotion", EMOTION_LABELS)
        except (ValueError, TypeError) as exc:
            last_error = exc
    raise RuntimeError(
        f"{model_name} 情緒標註在 {max_retries + 1} 次嘗試後仍未產生合法標籤。"
    ) from last_error


def add_emotion_feature(
    records: list,
    *,
    generator: Callable | None = None,
    model_name: str | None = None,
    config: RuntimeConfig | None = None,
) -> list:
    """整批成功後回填情緒與模型來源；失敗不留下部分更新的輸入。"""
    if not isinstance(records, list):
        raise ValueError("資料 JSON 最外層必須是陣列。")
    # 在載入模型前檢查所有描述，避免處理到後段才發現輸入缺漏。
    texts = [description_text(record, index) for index, record in enumerate(records, 1)]
    config = config or RuntimeConfig()
    model_name = model_name or config.gemma_model
    if generator is None and records:
        # 整批標註只載入一次；空資料直接返回，不開啟模型 session。
        with ModelSession(model_name, config) as session:
            return add_emotion_feature(
                records, generator=session.generate, model_name=model_name, config=config
            )
    # 模型呼叫可能中途失敗，先收集所有結果，再一起修改呼叫端的資料。
    emotions = []
    for index, text in enumerate(texts, 1):
        emotion = annotate_emotion(text, generator=generator, model_name=model_name, config=config)
        emotions.append(emotion)
        print(f"[情緒標註 {index}/{len(records)}] {emotion}")
    for record, emotion in zip(records, emotions):
        record[EMOTION_FIELD] = emotion
        # 即使欄名保留 target，來源仍是模型推測，不能宣稱為人工情緒真值。
        record["emotion_annotation_model"] = model_name
        record["emotion_annotation_source"] = "llm_generated"
        record["emotion_annotation_is_gold"] = False
    return records


def load_json(path: str | Path) -> list:
    """沿用資料流程的讀取介面，共用 JSON 格式與紀錄驗證。"""
    return load_records(path)


def save_json(records, path: str | Path) -> None:
    """輸出標準 JSON，寫入失敗時保留前一次的完整結果。"""
    write_json(records, path)


def run(in_path: str, out_path: str, *, config: RuntimeConfig | None = None) -> list:
    """標註訓練資料；情緒欄位不會作為情緒 fine-tuning 的目標。"""
    records = add_emotion_feature(load_json(in_path), config=config)
    save_json(records, out_path)
    print(f"已輸出 {out_path}（{len(records)} 筆）")
    return records


def main():
    """為增生完成的訓練資料補上模型情緒標註及可追蹤的來源資訊。"""
    run("data/interim/train_augmented.json", "data/interim/train_features.json")


if __name__ == "__main__":
    main()
