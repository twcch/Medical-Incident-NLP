import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

TEXT_FIELD = "description"        # content 內作為分析來源的事件描述欄位
EMOTION_FIELD = "emotion_target"  # 標注結果寫入的「撰寫者情緒」欄位（預處理預留的空槽）

EMOTION_LABELS = [
    "中性", "焦慮", "自責", "無奈", "擔憂", "沮喪", "憤怒", "驚慌", "困惑", "警覺"
]

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


def annotate_emotion(text: str, max_retries: int = 2) -> str:
    prompt = f"""
        # Task
            你將會閱讀一段由醫療人員撰寫的醫療事件描述，請判斷該描述「撰寫者」在書寫當下最可能的情緒，並從 # Constraint 的情緒清單中挑選一個最貼切的標籤。

        # Role
            你是一位具有臨床心理背景的醫療文本情緒分析專家，擅長從醫療事件報告的用詞、語氣與描述方式推斷撰寫者當下的情緒狀態。

        # Interaction
            N/A

        # Parameter
            text: 待分析的醫療事件描述文本。

        # Constraint
            1. 必須只回傳一個情緒標籤，且必須是以下其中之一：{EMOTION_LABELS}
            2. 判斷依據是「撰寫者」當下的情緒，不是病人或當事人的情緒。
            3. 若文字平鋪直敘、無明顯情緒色彩，請使用「中性」。
            4. 嚴格依照 # Output 的 JSON 格式輸出，不要附加任何其他文字。

        # Output
            {{"emotion": "情緒標籤"}}

        # Input
            text: {text}
    """

    client = get_client()
    for _ in range(max_retries + 1):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(response.choices[0].message.content)
        emotion = str(result.get("emotion", "")).strip()
        if emotion in EMOTION_LABELS:
            return emotion

    return "中性"


def add_emotion_feature(records: list) -> list:
    """
    Fill each record's emotion_target with the model-annotated emotion,
    inferred from content.description.

    Parameters:
    - records: list of dicts shaped like
      {"idt_target": ..., "emotion_target": "", "content": {"description": ..., "directive": ...}}.

    Returns:
    - The same list with EMOTION_FIELD (emotion_target) filled in.
    """
    total = len(records)
    for i, record in enumerate(records, start=1):
        text = str(record["content"].get(TEXT_FIELD) or "").strip()
        emotion = annotate_emotion(text) if text else "中性"
        record[EMOTION_FIELD] = emotion
        print(f"[{i}/{total}] {emotion}")
    return records


def load_json(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def run(in_path: str, out_path: str) -> list:
    """Add emotion features to one dataset."""
    print(f"=== {in_path} ===")
    records = load_json(in_path)
    records = add_emotion_feature(records)
    save_json(records, out_path)
    print(f"已輸出 {out_path}（{len(records)} 筆）")
    return records


def main():
    # train 讀「增強後」的檔；test 維持原始（不增強）
    run("data/interim/train_augmented.json", "data/interim/train_features.json")
    run("data/processed_data/test_data.json", "data/interim/test_features.json")


if __name__ == "__main__":
    main()
