"""送進模型的提示詞與回應解析；③ 增生與 ④ 情緒標註共用。

提示詞只放事件描述，批示不送模型。所有指示都放在單一 user 訊息，
不用 system 角色，避免各家 chat template 對 system 的支援不一致。
"""

import json
import unicodedata

from utils.preprocessing import VALID_IDT_LABELS

# ④ 的十種撰寫者情緒與定義。這份清單是研究設計的一部分，調整後必須重新標註。
EMOTION_LABELS: dict[str, str] = {
    "中性": "客觀陳述事件經過，看不出明顯情緒",
    "焦慮": "對事件後果或後續處理感到不安、心急",
    "擔憂": "掛念病人安全或病情，語氣帶有關切",
    "驚嚇": "事發突然、出乎意料，語氣帶有驚訝或受驚",
    "自責": "認為自己有疏失，語氣帶有檢討或歉意",
    "憤怒": "對他人或制度感到不滿、指責",
    "委屈": "覺得被誤解，或責任不在己卻要承擔後果",
    "無奈": "感到無力改變現況、只能接受",
    "困惑": "對事件原因或處置方式感到不解",
    "沮喪": "對結果失望、情緒低落",
}

AUGMENT_INSTRUCTIONS = """\
你是醫療事件通報資料的改寫助手。請把下面的事件描述改寫成「語意完全相同、但用字與句構明顯不同」的另一段描述。

規則：
1. 保留所有事實：人物、時間、數值、處置與結果都不可增減或更動。
2. 不可改變事件的歸因性質（屬於個人疏失或系統問題）。
3. 改寫後的文字必須與原文明顯不同，不可只替換少數字詞。
4. 使用繁體中文，長度與原文相近。
5. 只輸出一個 JSON 物件：{"description": "改寫後的描述"}，不要加任何說明或 Markdown 標記。"""

IDT_INSTRUCTIONS = """\
你是醫療事件通報的分析助手。請依 IDT（Incident Decision Tree）的精神，判斷下面這件事的主要歸因：
- 個人：主要肇因於個別人員的疏失、判斷或行為，例如未依規定執行、疏忽、明知故犯。
- 系統：主要肇因於流程、制度、人力、設備或環境等系統因素，例如規範不清、設備故障、人力不足。

只輸出「個人」或「系統」兩個字之一，不要加任何說明。"""

EMOTION_INSTRUCTIONS = """\
你是醫療事件通報資料的文本分析助手。請判斷「撰寫這段事件描述的人」在書寫當下最主要的情緒。

只能從下列十種情緒中選一種：
{labels}

注意：
- 只根據文字本身的語氣、用詞與寫法判斷，不要推測病人或其他當事人的情緒。
- 只輸出一個 JSON 物件：{"emotion": "<情緒標籤>"}，標籤必須是上列十種之一，不要加任何說明。"""


def build_augment_messages(description: str, last_failure: str | None = None) -> list[dict]:
    """組成一次改寫請求；重試時把上一次不可用的原因寫進提示。"""
    parts = [AUGMENT_INSTRUCTIONS, "", "事件描述：", description]
    if last_failure:
        parts += ["", f"注意：上一次的輸出不可用（{last_failure}），請重新改寫並遵守上述規則。"]
    return [{"role": "user", "content": "\n".join(parts)}]


def build_emotion_messages(description: str, last_failure: str | None = None) -> list[dict]:
    """組成一次情緒標註請求；溫度 0 下重試必須改變提示，否則會得到同樣的輸出。"""
    label_lines = "\n".join(
        f"- {label}：{definition}" for label, definition in EMOTION_LABELS.items()
    )
    parts = [EMOTION_INSTRUCTIONS.replace("{labels}", label_lines), "", "事件描述：", description]
    if last_failure:
        parts += [
            "",
            f"注意：上一次的輸出無效（{last_failure}），"
            '請只輸出 {"emotion": "<情緒標籤>"}，標籤必須是上列十種之一。',
        ]
    return [{"role": "user", "content": "\n".join(parts)}]


def parse_json_object(text: str) -> dict:
    """取出模型輸出中第一個完整的 JSON 物件；前後的說明文字或 Markdown 圍欄都忽略。"""
    text = text or ""
    start = text.find("{")
    if start == -1:
        raise ValueError("輸出中沒有 JSON 物件")
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 無法解析：{exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON 不是物件")
    return value


def parse_augment_response(text: str) -> str:
    """回傳改寫後的描述；格式不對就拋出 ValueError 並說明原因。"""
    value = parse_json_object(text).get("description")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("JSON 缺少非空的 description 欄位")
    return value.strip()


def parse_emotion_response(text: str) -> str:
    """回傳十種情緒之一；允許標籤前後有空白或全形字元，其他一律視為無效。"""
    value = parse_json_object(text).get("emotion")
    if not isinstance(value, str):
        raise ValueError("JSON 缺少 emotion 欄位")
    label = "".join(unicodedata.normalize("NFKC", value).split())
    if label not in EMOTION_LABELS:
        raise ValueError(f"「{value}」不在十種情緒標籤內")
    return label


def build_idt_messages(description: str, last_failure: str | None = None) -> list[dict]:
    """⑤ 訓練與 ⑥ 推論共用的 IDT 分類提示；只送事件描述。推論重試時附上上次無效的原因。"""
    parts = [IDT_INSTRUCTIONS, "", "事件描述：", description]
    if last_failure:
        parts += [
            "",
            f"注意：上一次的輸出無效（{last_failure}），請只輸出「個人」或「系統」兩個字之一。",
        ]
    return [{"role": "user", "content": "\n".join(parts)}]


def parse_idt_response(text: str) -> str:
    """只接受「個人」或「系統」；允許前後空白與句末標點，其他一律無效，不猜標籤。"""
    value = unicodedata.normalize("NFKC", text or "").strip().rstrip("。.!！")
    value = "".join(value.split())
    if value not in VALID_IDT_LABELS:
        raise ValueError(f"「{(text or '').strip()[:30]}」不是「個人」或「系統」")
    return value


# ⑨ 情緒評審的三個面向；分數由評審模型給，總分由程式取平均。
JUDGE_DIMENSIONS: dict[str, str] = {
    "plausibility": "情緒合理性：這個情緒標籤是否合理反映撰寫者的語氣與用詞",
    "evidence": "證據支持：描述文字中是否有具體的用詞或寫法支持這個標籤",
    "writer_perspective": "撰寫者視角：標籤描述的是撰寫者本人的情緒，而不是病人或其他當事人的情緒",
}
JUDGE_SCORE_RANGE = (1, 5)

JUDGE_INSTRUCTIONS = """\
你是醫療事件通報文本分析的評審。下面有一段事件描述，以及另一個模型判斷的「撰寫者書寫當下的情緒」。
請只根據描述文字本身，從三個面向各給 1 到 5 的整數分數，並用一兩句話說明理由：
{dimensions}

評分標準：1 完全不符、2 大多不符、3 部分符合、4 大致符合、5 完全符合。
只輸出一個 JSON 物件，不要加任何說明：
{"plausibility": 1-5, "evidence": 1-5, "writer_perspective": 1-5, "reason": "理由"}"""


def build_judge_messages(
    description: str, emotion: str, last_failure: str | None = None
) -> list[dict]:
    """組成一次評審請求；只送描述與被評的情緒標籤，不送批示與人工標籤。"""
    dimensions = "\n".join(f"- {key}（{text}）" for key, text in JUDGE_DIMENSIONS.items())
    parts = [
        JUDGE_INSTRUCTIONS.replace("{dimensions}", dimensions),
        "",
        "事件描述：",
        description,
        "",
        f"被評的情緒標籤：{emotion}",
    ]
    if last_failure:
        parts += ["", f"注意：上一次的輸出無效（{last_failure}），請依指定格式只輸出 JSON 物件。"]
    return [{"role": "user", "content": "\n".join(parts)}]


def parse_judge_response(text: str) -> dict:
    """回傳 {"scores": {面向: 1-5}, "overall": 三項平均, "reason": 理由}；任一分數無效就拋出。"""
    data = parse_json_object(text)
    low, high = JUDGE_SCORE_RANGE
    scores = {}
    for key in JUDGE_DIMENSIONS:
        value = data.get(key)
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} 必須是 {low} 到 {high} 的整數，收到 {value!r}")
        scores[key] = value
    reason = data.get("reason")
    reason = reason.strip() if isinstance(reason, str) else ""
    return {
        "scores": scores,
        "overall": round(sum(scores.values()) / len(scores), 3),
        "reason": reason,
    }
