"""使用 Gemma 標註事件描述的撰寫者情緒；此標註不是人工真實標籤。"""

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

if __package__:
    from .config import RuntimeConfig
    from .json_io import load_records
    from .json_io import save_json as write_json
    from .llm import ModelSession, parse_json_object
    from .progress import ProgressFile, completion_marker, format_eta, is_complete, mark_complete
    from .prompts import EMOTION_LABELS, EMOTION_SYSTEM_PROMPT
else:
    from config import RuntimeConfig
    from json_io import load_records
    from json_io import save_json as write_json
    from llm import ModelSession, parse_json_object
    from progress import ProgressFile, completion_marker, format_eta, is_complete, mark_complete
    from prompts import EMOTION_LABELS, EMOTION_SYSTEM_PROMPT

TEXT_FIELD = "description"
EMOTION_FIELD = "emotion_target"  # 保留既有資料欄位，內容為 Gemma 產生的標註
# 標註規則或解析方式改變時更新，避免舊規則留下的進度檔被套用到新規則。
METHOD_VERSION = "emotion-v1"


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
    resume: dict | None = None,
    on_record: Callable | None = None,
) -> list:
    """整批成功後回填情緒與模型來源；失敗不留下部分更新的輸入。

    resume 是 {紀錄序號（1 起算）: 情緒}，用來略過已完成的紀錄；
    on_record(序號, 情緒) 在每筆完成後呼叫，供呼叫端保存進度。
    """
    if not isinstance(records, list):
        raise ValueError("資料 JSON 最外層必須是陣列。")
    # 在載入模型前檢查所有描述，避免處理到後段才發現輸入缺漏。
    texts = [description_text(record, index) for index, record in enumerate(records, 1)]
    config = config or RuntimeConfig()
    model_name = model_name or config.gemma_model
    resume = resume or {}
    pending = [index for index in range(1, len(records) + 1) if index not in resume]
    if generator is None and pending:
        # 整批標註只載入一次；空資料或全部已還原時不開啟模型 session。
        with ModelSession(model_name, config) as session:
            return add_emotion_feature(
                records,
                generator=session.generate,
                model_name=model_name,
                config=config,
                resume=resume,
                on_record=on_record,
            )
    if resume:
        restored = len(records) - len(pending)
        print(f"續跑：從進度檔還原 {restored} 筆，其餘 {len(pending)} 筆重新標註")
    # 模型呼叫可能中途失敗，先收集所有結果，再一起修改呼叫端的資料。
    emotions = []
    started = time.monotonic()
    annotated = 0
    for index, text in enumerate(texts, 1):
        if index in resume:
            emotions.append(resume[index])
            continue
        emotion = annotate_emotion(text, generator=generator, model_name=model_name, config=config)
        emotions.append(emotion)
        annotated += 1
        if on_record is not None:
            on_record(index, emotion)
        average = (time.monotonic() - started) / annotated
        remaining = format_eta(average * (len(records) - index))
        print(f"[情緒標註 {index}/{len(records)}] {emotion}（預估剩餘 {remaining}）")
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


def _restore_progress(entries: list, records: list, progress: ProgressFile) -> dict:
    """把進度檔還原成 {序號: 情緒}；與輸入資料對不上時中止，不默默略過。"""
    resume = {}
    for entry in entries:
        index = entry.get("index") if isinstance(entry, dict) else None
        valid = (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 1 <= index <= len(records)
            and entry.get("emotion") in EMOTION_LABELS
            and entry.get("record_id") == records[index - 1].get("record_id")
        )
        if not valid:
            raise ValueError(
                f"進度檔 {progress.path} 與輸入資料對不上；確認不需要後請刪除它再重新執行。"
            )
        resume[index] = entry["emotion"]
    return resume


def run(in_path: str, out_path: str, *, config: RuntimeConfig | None = None) -> list:
    """標註訓練資料；情緒欄位不會作為情緒 fine-tuning 的目標。

    每標註一筆就追加到「輸出檔.partial.jsonl」，中斷後重新執行同一指令會從該處繼續；
    全部完成並寫出輸出檔後，進度檔才會刪除，並留下「輸出檔.done.json」完成標記：輸入與設定
    都沒變時再次執行會直接略過。
    """
    config = config or RuntimeConfig()
    records = load_json(in_path)
    # 檔頭記錄輸入與規則的指紋；提示詞改了，舊進度就不能再沿用。
    header = {
        "input_sha256": hashlib.sha256(Path(in_path).read_bytes()).hexdigest(),
        "model": config.gemma_model,
        "method": METHOD_VERSION,
        "prompt_sha256": hashlib.sha256(EMOTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }
    if is_complete(out_path, header):
        marker = completion_marker(out_path)
        print(f"已有完成的情緒標註，略過：{out_path}（要重做請刪除該檔與 {marker}）")
        return load_json(out_path)
    progress = ProgressFile(f"{out_path}.partial.jsonl", header)
    resume = _restore_progress(progress.load(), records, progress)
    try:
        with progress:
            annotated = add_emotion_feature(
                records,
                config=config,
                resume=resume,
                on_record=lambda index, emotion: progress.append(
                    {
                        "index": index,
                        "record_id": records[index - 1].get("record_id"),
                        "emotion": emotion,
                    }
                ),
            )
    except BaseException:
        # 含 Ctrl+C：已完成的紀錄都在進度檔裡，告訴使用者重跑即可接續。
        if progress.count:
            print(f"已保留進度：{progress.path}（重新執行同一指令會從中斷處繼續）")
        raise
    save_json(annotated, out_path)
    progress.remove()
    mark_complete(out_path, header)
    print(f"已輸出 {out_path}（{len(annotated)} 筆）")
    return annotated


def main():
    """為增生完成的訓練資料補上模型情緒標註及可追蹤的來源資訊。"""
    run("data/interim/train_augmented.json", "data/interim/train_features.json")


if __name__ == "__main__":
    main()
