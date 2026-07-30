import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

TEXT_FIELD = "description"  # content 內要改寫的事件描述欄位

_client = None


def get_client() -> OpenAI:
    """
    Lazily build the OpenAI client so that importing this module does not
    require an API key (the key is only needed when actually calling the API).
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未設定，請在 .env 設定後再執行。")
        _client = OpenAI(api_key=api_key)
    return _client


def paraphrase_text(text: str, n: int = 3, max_retries: int = 3) -> list:
    """
    Generate up to n paraphrased versions of a medical incident description
    that keep the meaning but vary the wording.

    Returns:
    - list of unique paraphrases (may be fewer than n if the model repeats itself).
    """
    seen = {text}
    accepted = []

    for _ in range(max_retries + 1):
        need = n - len(accepted)
        if need <= 0:
            break

        avoid_block = ""
        if accepted:
            avoid_list = "\n".join(f"- {t}" for t in accepted)
            avoid_block = f"\n            # Avoid\n                以下版本已經產生過，請勿再次產生重複內容：\n{avoid_list}\n"

        prompt = f"""
            # Task
                你將會對醫療事件描述文本進行改寫，user 將會提供給你一段醫療事件描述，請將其改寫為 {need} 個語意相同但用詞不同的版本，保持醫療事實不變，並且按照 #Output 的格式輸出。每個版本之間必須彼此不同，也不可與原文相同。

            # Role
                你是一位醫療事件報告改寫專家，專門負責將醫療事件描述改寫成多個版本，以增加訓練資料的多樣性。

            # Interaction
                N/A

            # Parameter
                n: 你需要生成的改寫版本數量，必須是一個正整數。
                text: 原始的醫療事件描述文本，這是你需要改寫的內容。

            # Constraint
                1. 生成的版本必須與原文語意相同，但用詞不同。
                2. 每個版本之間必須彼此不同。
                3. {avoid_block}

            # Output
                你需要以 JSON 格式回傳，格式如下：
                {{"augmented": ["版本1", "版本2", ...]}}

            # Input
                text: {text}
        """

        response = get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,
        )

        result = json.loads(response.choices[0].message.content)
        for aug_text in result.get("augmented", []):
            aug_clean = str(aug_text).strip()
            if not aug_clean or aug_clean in seen:
                continue
            seen.add(aug_clean)
            accepted.append(aug_clean)
            if len(accepted) >= n:
                break

    return accepted


def augment_records(records: list, n: int = 2) -> list:
    """
    Augment the dataset by paraphrasing each record's content.description.
    The original records are kept and the paraphrased versions are appended
    (same target and directive, only the description varies).

    Parameters:
    - records: list of dicts shaped like
      {"idt_target": ..., "emotion_target": ..., "content": {"description": ..., "directive": ...}}.
    - n: number of paraphrased versions to generate per record.

    Returns:
    - A new list = original records + augmented records.
    """
    augmented = []
    total = len(records)

    for i, record in enumerate(records, start=1):
        text = str(record["content"].get(TEXT_FIELD) or "").strip()
        if not text:
            print(f"[{i}/{total}] 跳過（description 為空）")
            continue

        versions = paraphrase_text(text, n=n)
        if len(versions) < n:
            print(f"[warn] 第 {i} 筆：只取得 {len(versions)}/{n} 個版本")

        for aug in versions:
            # 沿用原始整筆欄位（idt_target、emotion_target…），只替換改寫後的 description
            new_content = {**record["content"], TEXT_FIELD: aug}
            augmented.append({**record, "content": new_content})

        print(f"[{i}/{total}] +{len(versions)}")

    return records + augmented


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def run(in_path: str, out_path: str, n: int = 2) -> list:
    """Augment one dataset: paraphrase each record's description and append the versions."""
    records = load_json(in_path)
    augmented = augment_records(records, n=n)
    save_json(augmented, out_path)
    print(f"已輸出 {out_path}（{len(records)} → {len(augmented)} 筆）")
    return augmented


def main():
    # 只對訓練集做資料增強，測試集維持原樣
    run("data/processed_data/train_data.json", "data/interim/train_augmented.json", n=2)


if __name__ == "__main__":
    main()
