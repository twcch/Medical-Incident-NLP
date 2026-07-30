import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:  # 透過 run.py (`from src import ...`) 匯入時
    from src.trainer import IDT_SYSTEM_PROMPT, EMOTION_SYSTEM_PROMPT
except ImportError:  # 直接 `python3 src/inference.py` 執行時
    from trainer import IDT_SYSTEM_PROMPT, EMOTION_SYSTEM_PROMPT

load_dotenv()

TEXT_FIELD = "description"        # content 內作為輸入的事件描述
IDT_TARGET = "idt_target"         # 真實標籤（個人/系統）
EMOTION_TARGET = "emotion_target"  # 情緒標籤（模型標注）

_client = None


def get_client() -> OpenAI:
    """Lazily build the OpenAI client (key only needed when actually calling the API)."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未設定，請在 .env 設定後再執行。")
        _client = OpenAI(api_key=api_key)
    return _client


def read_model_id(path: str) -> str:
    """讀取 models/*.txt 內存的 fine-tuned model id。"""
    return Path(path).read_text(encoding="utf-8").strip()


def predict_single(text, model_name, system_prompt, temperature=0.0):
    client = get_client()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def run_inference(records, tasks, temperature=0.0):
    """
    對每個 task 用對應的 fine-tuned 模型逐筆預測，回填預測欄位，並在有標籤時計算 accuracy。

    Parameters:
    - records: list of dicts，每筆含 content.description 與（可選）標籤欄位。
    - tasks: list of dicts，每個含 name / model_name / system_prompt / output_field / label_field。

    Returns:
    - (records, summaries)：records 已回填預測欄位；summaries 為各 task 的評估結果。
    """
    total = len(records)
    summaries = []

    for task in tasks:
        name = task["name"]
        model_name = task["model_name"]
        system_prompt = task["system_prompt"]
        output_field = task["output_field"]
        label_field = task.get("label_field")

        print(f"[{name}] 用 {model_name} 預測 {total} 筆 -> {output_field}")
        correct = 0
        labeled = 0
        for i, record in enumerate(records, start=1):
            text = str(record["content"].get(TEXT_FIELD) or "").strip()
            pred = predict_single(text, model_name, system_prompt, temperature) if text else ""
            record[output_field] = pred

            truth = str(record.get(label_field) or "").strip() if label_field else ""
            if truth:
                labeled += 1
                is_correct = pred == truth
                correct += int(is_correct)
                print(f"  [{i}/{total}] truth={truth} pred={pred} {'OK' if is_correct else 'X'}")
            else:
                print(f"  [{i}/{total}] {pred}")

        if labeled:
            accuracy = correct / labeled
            print(f"[{name}] accuracy: {accuracy:.4f} ({correct}/{labeled})")
            summaries.append(
                {
                    "task": name,
                    "model": model_name,
                    "label_field": label_field,
                    "n": labeled,
                    "n_correct": correct,
                    "accuracy": accuracy,
                }
            )

    return records, summaries


def run(
    in_path: str = "data/interim/test_features.json",
    out_dir: str = "results",
    models_dir: str = "models",
):
    """用 models/ 下的兩個 fine-tuned 模型，對 test_features.json 做 idt + emotion 預測與評估。"""
    with open(in_path, encoding="utf-8") as f:
        records = json.load(f)

    tasks = [
        {
            "name": "idt",
            "model_name": read_model_id(f"{models_dir}/idt_model.txt"),
            "system_prompt": IDT_SYSTEM_PROMPT,
            "output_field": "idt_pred",
            "label_field": IDT_TARGET,
        },
        {
            "name": "emotion",
            "model_name": read_model_id(f"{models_dir}/emotion_model.txt"),
            "system_prompt": EMOTION_SYSTEM_PROMPT,
            "output_field": "emotion_pred",
            "label_field": EMOTION_TARGET,
        },
    ]

    records, summaries = run_inference(records, tasks)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir_path / "inference_predictions.json"
    pred_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已輸出預測: {pred_path}")

    eval_path = out_dir_path / "inference_evaluation.json"
    eval_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已輸出評估: {eval_path}")

    return summaries


def main():
    run()


if __name__ == "__main__":
    main()
