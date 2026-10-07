"""Gemma IDT LoRA 推論、Gemma 情緒標註，以及獨立的 Qwen 情緒評分。"""

from pathlib import Path
from typing import Callable

if __package__:
    from .config import RuntimeConfig
    from .evaluation import evaluate_emotions, evaluate_idt
    from .feature_engineering import (
        annotate_emotion, description_text, load_json, parse_label_response, save_json,
    )
    from .llm import ModelSession
    from .prompts import IDT_SYSTEM_PROMPT, VALID_IDT_LABELS
else:
    from config import RuntimeConfig
    from evaluation import evaluate_emotions, evaluate_idt
    from feature_engineering import (
        annotate_emotion, description_text, load_json, parse_label_response, save_json,
    )
    from llm import ModelSession
    from prompts import IDT_SYSTEM_PROMPT, VALID_IDT_LABELS

TEXT_FIELD = "description"
IDT_TARGET = "idt_target"
EMOTION_TARGET = "emotion_target"  # 僅供既有模型標註資料相容，不能作為情緒真值


def predict_idt(
    text: str,
    *,
    generator: Callable,
    config: RuntimeConfig,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> str:
    """驗證 IDT 輸出是個人／系統；不將其他回答當成有效預測。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("IDT 推論的輸入必須是非空白文字。")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries 必須是非負整數。")
    last_error = None
    for _ in range(max_retries + 1):
        try:
            raw = generator(
                IDT_SYSTEM_PROMPT,
                text.strip(),
                temperature=temperature,
                max_new_tokens=config.max_new_tokens,
            )
            return parse_label_response(raw, "idt", VALID_IDT_LABELS)
        except (ValueError, TypeError) as exc:
            last_error = exc
    raise RuntimeError(
        f"{config.gemma_model} IDT 推論在 {max_retries + 1} 次嘗試後仍未產生合法標籤。"
    ) from last_error


def run_inference(
    records: list,
    *,
    config: RuntimeConfig | None = None,
    models_dir: str = "models",
    temperature: float = 0.0,
) -> tuple[list, list]:
    """先完成 IDT adapter 推論，再以原始 Gemma 重新預測情緒。"""
    if not isinstance(records, list):
        raise ValueError("資料 JSON 最外層必須是陣列。")
    texts = [description_text(record, index) for index, record in enumerate(records, 1)]
    config = config or RuntimeConfig()
    adapter_path = Path(models_dir) / "idt_adapter"
    if records:
        # 兩個階段各自釋放模型，情緒標註不套用只為 IDT 訓練的 adapter。
        with ModelSession(config.gemma_model, config, adapter_path=adapter_path) as session:
            for index, (record, text) in enumerate(zip(records, texts), 1):
                prediction = predict_idt(
                    text, generator=session.generate, config=config, temperature=temperature
                )
                record["idt_pred"] = prediction
                record["idt_prediction_model"] = config.gemma_model
                record["idt_adapter_path"] = str(adapter_path)
                print(f"[IDT 推論 {index}/{len(records)}] {prediction}")
        with ModelSession(config.gemma_model, config) as session:
            for index, (record, text) in enumerate(zip(records, texts), 1):
                prediction = annotate_emotion(text, generator=session.generate, config=config)
                record["emotion_pred"] = prediction
                record["emotion_prediction_model"] = config.gemma_model
                record["emotion_prediction_source"] = "llm_generated"
                # 更新預測後，先清除前一次的評分，避免舊評分套用到新標籤。
                record.pop("emotion_judge", None)
                print(f"[情緒推論 {index}/{len(records)}] {prediction}")
    idt_summary = evaluate_idt(records, model_name=config.gemma_model)
    idt_summary["adapter_path"] = str(adapter_path)
    emotion_summary = {
        "task": "emotion",
        "model": config.gemma_model,
        "prediction_field": "emotion_pred",
        "n": len(records),
        "evaluation_status": "not_scored",
        "limitation": "情緒沒有人工真實標籤，需另外使用 Qwen 評分。",
    }
    return records, [idt_summary, emotion_summary]


def run(
    in_path: str = "data/processed_data/test_data.json",
    out_dir: str = "results",
    models_dir: str = "models",
    *,
    config: RuntimeConfig | None = None,
    evaluate_emotion: bool = True,
) -> list:
    """輸出 IDT／情緒預測，並以 Qwen 對情緒給出 1 至 5 分與理由。"""
    config = config or RuntimeConfig()
    records, summaries = run_inference(
        load_json(in_path), config=config, models_dir=models_dir
    )
    output_dir = Path(out_dir)
    prediction_path = output_dir / "inference_predictions.json"
    evaluation_path = output_dir / "inference_evaluation.json"
    # 評分失敗時仍保留已完成的 Gemma 預測，可使用 evaluate 流程重試。
    save_json(records, prediction_path)
    # 同步更新評估狀態，避免 Qwen 失敗後留下上一次推論的評估結果。
    save_json(summaries, evaluation_path)
    if evaluate_emotion:
        # run_inference 已關閉 Gemma，此處才載入 Qwen，降低 GPU 記憶體需求。
        emotion_summary = evaluate_emotions(records, config=config)
        emotion_summary["annotation_model"] = config.gemma_model
        summaries[1] = emotion_summary
        save_json(records, prediction_path)
    else:
        summaries[1]["evaluation_status"] = "skipped"
    save_json(summaries, evaluation_path)
    print(f"已輸出預測：{prediction_path}")
    print(f"已輸出評估：{evaluation_path}")
    return summaries


def main():
    run()


if __name__ == "__main__":
    main()
