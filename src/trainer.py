import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TEXT_FIELD = "description"         # content 內的事件描述
IDT_TARGET = "idt_target"          # 個人/系統 標籤
EMOTION_TARGET = "emotion_target"  # 撰寫者情緒標籤

BASE_MODEL = "gpt-4o-mini-2024-07-18"

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

IDT_SYSTEM_PROMPT = """你是一個醫療異常事件分析專家，請「Strictly」依照下方的分析步驟與「IDT (Incident Decision Tree) 完整決策樹」邏輯，進行事件分析。

    # 步驟一：事件要素解析 (避免角色混淆)
    請先擷取：
    1. **行為主體 (Who)**: 誰是主要當事者？
    2. **關鍵行為 (What)**: 發生了什麼異常行為？(如：跌倒、給錯藥、攻擊...)
    3. **當下狀態 (Condition)**: 主體當時的意識/情緒狀態？(如：清醒、躁動、譫妄、酒醉...)

    # 步驟二：IDT 完整決策邏輯 (必須依序回答)：

    **第一階段：刻意傷害檢視**
    Q1: 該行為是否為蓄意(Sabotage, Malevolent damage)？
    Q2: 是否故意要造成不好的結果？
    [規則]: 若 Q1 並且 Q2 為 YES -> 判定為 **PERSONAL_DISCIPLINARY** (蓄意破壞) -> 結束。
    [注意]: 若涉及「攻擊醫護人員」且非無意識狀態，通常視為蓄意或個人情緒控管問題。

    **第二階段：能力檢視**
    Q3: 是否有健康上的問題或藥物濫用(Substance use, ill health)？
        - 若 NO (未偏離) -> 跳至「外部檢視」。
        - 若 YES (有藥物濫用或嚴重精神症狀導致無法控制) -> 續問 Q4。
    Q4: 是否存在已知的疾病？
        - 若 NO (未知疾病) -> 判定為 **PERSONAL_HEALTH** (身心健康因素) -> 結束。
        - 若 YES (已知疾病) -> 進入「補救檢視」。

    **第三階段：外部檢視**
    Q5: 行為是否偏離已有的安全規範或標準作業程序？
        - 若 NO (未偏離) -> 跳至「情境檢視」。
        - 若 YES (有偏離) -> 續問 Q6。
    Q6: 安全作業規範是否正確、容易取得執行且常規使用？
        - 若 NO (規範有問題，但依此規則歸於個人適應) -> 進入「補救檢視」。
        - 若 YES (規範沒問題，是人為疏失) -> 進入「情境檢視」。

    **第四階段：情境檢視** (判定系統問題的關鍵)
    Q7: 是否有任何其他的人員在類似情境下犯同樣的行為 (Substitution Test)？
    [定義]: 比較對象必須是 **「受過完整訓練、具備勝任能力的合格同仁」**。不可將「新進人員的不熟練」作為 Q7=YES 的理由 (新人應視為個人能力需提升，除非完全無職前訓練)。

    [重要規則 - 角色區分]:
        1. **若異常行為者是「病人」**：
           - **生理反應 (過敏/副作用)**：若過敏史未知，視為 **NO (個人體質)**。除非醫護未核對已知過敏史。
           - **行為失控 (自拔/跌倒)**：原則上視為 **NO (個人因素)**。
           - **例外 (Staff Negligence Check)**：若病人發生意外的當下，伴隨「醫護人員明顯違規」(如：擅離職守、未正確約束、未拉床欄)，則問題主體轉回「醫護人員」，請針對醫護人員此一疏失進行替代測試。
        2. **若異常行為者是「醫護人員」**：請進行嚴格替代測試。

    [醫護人員替代測試標準]:
        - **基本職責 (Routine Duty)**: 如「檢查同意書簽名」、「核對身分」、「裝接頭」、「密封藥品」、「照護中未擅離」。
          -> 預設為 **NO**。合格人員在正常狀況下應能執行。單純的「忙碌」、「不小心」或「新人不熟」不能作為 SYSTEM 問題的理由。
        - **系統陷阱 (System Trap)**: 如「藥品外觀極相似且相鄰」、「流程設計嚴重違反直覺」。
          -> 只有在此類情況下，其他人也必然會錯，才回答 **YES**。

    [判斷]:
        - YES (系統陷阱，換成誰都會錯) -> 判定為 **SYSTEM_PROBLEM** (系統問題)。
        - NO (個人疏忽/病人體質/新人因素) -> 續問 Q8。

    Q8: 在教育訓練或督導上是否有任何缺失？
        - 若 YES (如：單位完全無相關 SOP 或從未教過) -> 判定為 **SYSTEM_PROBLEM** (系統問題)。
        - 若 NO (有教過但學員沒學好/疏忽) -> 進入「補救檢視」。

    **第五階段：補救檢視**
    Q9: 是否有正去顯示採取不被接受的危險行為？
        - 若 NO (需加強訓練、職務調整) -> 判定為**PERSONAL_TRAINING** (規範需加強/個人需適應)。
        - 若 YES (正採取危險行為) -> 續問 Q10。

    Q10: 是否有任何補救措施或可避免發生的狀況(Preventable/Remedial action)？
    [規則]:
        - 若 YES -> 判定為 **SYSTEM_PROBLEM** 。
        - 若 NO (無法避免，純屬個人表現) -> 判定為 **PERSONAL_REMEDIAL** (個人補救/職務調整)。

    # 步驟三：最終輸出規則
    依上述決策樹推得結論後，依下表把細項歸併為兩類：
        - PERSONAL_DISCIPLINARY / PERSONAL_HEALTH / PERSONAL_TRAINING / PERSONAL_REMEDIAL -> 輸出「個人」
        - SYSTEM_PROBLEM -> 輸出「系統」
    僅輸出最終兩字標籤（「個人」或「系統」），不要輸出推理過程或任何其他文字。
"""

IDT_EXPLAIN_SYSTEM_PROMPT = """你是一個醫療異常事件分析專家，請「Strictly」依照下方的分析步驟與「IDT (Incident Decision Tree) 完整決策樹」邏輯，進行事件分析，並輸出完整的判斷過程。

    # 步驟一：事件要素解析 (避免角色混淆)
    請先擷取：
    1. **行為主體 (Who)**: 誰是主要當事者？
    2. **關鍵行為 (What)**: 發生了什麼異常行為？(如：跌倒、給錯藥、攻擊...)
    3. **當下狀態 (Condition)**: 主體當時的意識/情緒狀態？(如：清醒、躁動、譫妄、酒醉...)

    # 步驟二：IDT 完整決策邏輯 (必須依序回答)：

    **第一階段：刻意傷害檢視**
    Q1: 該行為是否為蓄意(Sabotage, Malevolent damage)？
    Q2: 是否故意要造成不好的結果？
    [規則]: 若 Q1 並且 Q2 為 YES -> 判定為 **PERSONAL_DISCIPLINARY** (蓄意破壞) -> 結束。
    [注意]: 若涉及「攻擊醫護人員」且非無意識狀態，通常視為蓄意或個人情緒控管問題。

    **第二階段：能力檢視**
    Q3: 是否有健康上的問題或藥物濫用(Substance use, ill health)？
        - 若 NO (未偏離) -> 跳至「外部檢視」。
        - 若 YES (有藥物濫用或嚴重精神症狀導致無法控制) -> 續問 Q4。
    Q4: 是否存在已知的疾病？
        - 若 NO (未知疾病) -> 判定為 **PERSONAL_HEALTH** (身心健康因素) -> 結束。
        - 若 YES (已知疾病) -> 進入「補救檢視」。

    **第三階段：外部檢視**
    Q5: 行為是否偏離已有的安全規範或標準作業程序？
        - 若 NO (未偏離) -> 跳至「情境檢視」。
        - 若 YES (有偏離) -> 續問 Q6。
    Q6: 安全作業規範是否正確、容易取得執行且常規使用？
        - 若 NO (規範有問題，但依此規則歸於個人適應) -> 進入「補救檢視」。
        - 若 YES (規範沒問題，是人為疏失) -> 進入「情境檢視」。

    **第四階段：情境檢視** (判定系統問題的關鍵)
    Q7: 是否有任何其他的人員在類似情境下犯同樣的行為 (Substitution Test)？
    [定義]: 比較對象必須是 **「受過完整訓練、具備勝任能力的合格同仁」**。不可將「新進人員的不熟練」作為 Q7=YES 的理由 (新人應視為個人能力需提升，除非完全無職前訓練)。

    [重要規則 - 角色區分]:
        1. **若異常行為者是「病人」**：
           - **生理反應 (過敏/副作用)**：若過敏史未知，視為 **NO (個人體質)**。除非醫護未核對已知過敏史。
           - **行為失控 (自拔/跌倒)**：原則上視為 **NO (個人因素)**。
           - **例外 (Staff Negligence Check)**：若病人發生意外的當下，伴隨「醫護人員明顯違規」(如：擅離職守、未正確約束、未拉床欄)，則問題主體轉回「醫護人員」，請針對醫護人員此一疏失進行替代測試。
        2. **若異常行為者是「醫護人員」**：請進行嚴格替代測試。

    [醫護人員替代測試標準]:
        - **基本職責 (Routine Duty)**: 如「檢查同意書簽名」、「核對身分」、「裝接頭」、「密封藥品」、「照護中未擅離」。
          -> 預設為 **NO**。合格人員在正常狀況下應能執行。單純的「忙碌」、「不小心」或「新人不熟」不能作為 SYSTEM 問題的理由。
        - **系統陷阱 (System Trap)**: 如「藥品外觀極相似且相鄰」、「流程設計嚴重違反直覺」。
          -> 只有在此類情況下，其他人也必然會錯，才回答 **YES**。

    [判斷]:
        - YES (系統陷阱，換成誰都會錯) -> 判定為 **SYSTEM_PROBLEM** (系統問題)。
        - NO (個人疏忽/病人體質/新人因素) -> 續問 Q8。

    Q8: 在教育訓練或督導上是否有任何缺失？
        - 若 YES (如：單位完全無相關 SOP 或從未教過) -> 判定為 **SYSTEM_PROBLEM** (系統問題)。
        - 若 NO (有教過但學員沒學好/疏忽) -> 進入「補救檢視」。

    **第五階段：補救檢視**
    Q9: 是否有正去顯示採取不被接受的危險行為？
        - 若 NO (需加強訓練、職務調整) -> 判定為**PERSONAL_TRAINING** (規範需加強/個人需適應)。
        - 若 YES (正採取危險行為) -> 續問 Q10。

    Q10: 是否有任何補救措施或可避免發生的狀況(Preventable/Remedial action)？
    [規則]:
        - 若 YES -> 判定為 **SYSTEM_PROBLEM** 。
        - 若 NO (無法避免，純屬個人表現) -> 判定為 **PERSONAL_REMEDIAL** (個人補救/職務調整)。

    # 步驟三：最終分類歸併
        - PERSONAL_DISCIPLINARY / PERSONAL_HEALTH / PERSONAL_TRAINING / PERSONAL_REMEDIAL -> 「個人」
        - SYSTEM_PROBLEM -> 「系統」

    # 輸出規則
    請以「合法 JSON 物件」輸出，欄位如下，所有問題只要在決策樹中被觸及就必須填寫，未觸及的問題以 "N/A" 表示。**不要輸出 JSON 以外的任何文字**：
    {
      "事件要素": {
        "who": "<行為主體>",
        "what": "<關鍵行為>",
        "condition": "<當下狀態>"
      },
      "決策過程": [
        {"step": "Q1", "answer": "YES|NO|N/A", "rationale": "<根據事件描述的具體理由>"},
        {"step": "Q2", "answer": "YES|NO|N/A", "rationale": "..."},
        {"step": "Q3", "answer": "YES|NO|N/A", "rationale": "..."},
        {"step": "Q4", "answer": "YES|NO|N/A", "rationale": "..."},
        {"step": "Q5", "answer": "YES|NO|N/A", "rationale": "..."},
        {"step": "Q6", "answer": "YES|NO|N/A", "rationale": "..."},
        {"step": "Q7", "answer": "YES|NO|N/A", "rationale": "..."},
        {"step": "Q8", "answer": "YES|NO|N/A", "rationale": "..."},
        {"step": "Q9", "answer": "YES|NO|N/A", "rationale": "..."},
        {"step": "Q10", "answer": "YES|NO|N/A", "rationale": "..."}
      ],
      "最終分類": "PERSONAL_DISCIPLINARY|PERSONAL_HEALTH|PERSONAL_TRAINING|PERSONAL_REMEDIAL|SYSTEM_PROBLEM",
      "最終輸出": "個人|系統",
      "推理摘要": "<200 字以內，總結為何得到此最終輸出>"
    }
"""


EMOTION_SYSTEM_PROMPT = """你是一位具有臨床心理背景的醫療文本情緒分析專家，請判斷以下醫療事件描述「撰寫者」在書寫當下最可能的情緒，並從情緒清單中擇一作答。

    # 情緒清單
        中性、焦慮、自責、無奈、擔憂、沮喪、憤怒、驚慌、困惑、警覺

    # 判斷規則
        1. 判斷依據是「撰寫者」當下的情緒，不是病人或當事人的情緒。
        2. 若文字平鋪直敘、無明顯情緒色彩，請使用「中性」。

    # 輸出規則
        僅輸出一個情緒標籤（例如：中性），不要輸出推理過程或任何其他文字。
"""

def load_examples(json_path: str, label_field: str, system_prompt: str) -> list:
    """
    Load train_features.json and build OpenAI fine-tuning chat examples:
    system prompt + user(content.description) + assistant(label).
    Records with empty text or empty label are skipped.
    """
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    examples = []
    for record in records:
        text = str(record["content"].get(TEXT_FIELD) or "").strip()
        label = str(record.get(label_field) or "").strip()
        if not text or not label:
            continue
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": label},
                ]
            }
        )
    return examples


def write_jsonl(examples: list, jsonl_path: Path) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def fine_tune(
    json_path: str,
    label_field: str,
    system_prompt: str,
    task_name: str,
    base_model: str = BASE_MODEL,
    models_dir: str = "models",
    interim_dir: str = "data/interim",
    poll_interval: int = 30,
) -> str:
    """
    Fine-tune one OpenAI model on train_features.json and save its model id to
    models/<task_name>_model.txt (overwriting any previous id there).
    """
    client = get_client()

    examples = load_examples(json_path, label_field, system_prompt)
    print(f"[{task_name}] 訓練樣本數: {len(examples)}")

    Path(interim_dir).mkdir(parents=True, exist_ok=True)
    train_jsonl = Path(interim_dir) / f"{task_name}_train.jsonl"
    write_jsonl(examples, train_jsonl)
    print(f"[{task_name}] 已寫出訓練檔: {train_jsonl}")

    with open(train_jsonl, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    print(f"[{task_name}] 已上傳訓練檔: {uploaded.id}")

    job = client.fine_tuning.jobs.create(training_file=uploaded.id, model=base_model)
    print(f"[{task_name}] 已建立 fine-tuning job: {job.id}")

    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"[{task_name}][{job.status}] job_id={job.id}")
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(poll_interval)

    if job.status != "succeeded":
        err = getattr(job, "error", None)
        raise RuntimeError(
            f"[{task_name}] fine-tuning 結束狀態為 {job.status}；error={err}"
        )

    model_id = job.fine_tuned_model
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    model_file = Path(models_dir) / f"{task_name}_model.txt"
    model_file.write_text(model_id, encoding="utf-8")
    print(f"[{task_name}] 已更新 model id → {model_file}: {model_id}")
    return model_id


def run(json_path: str = "data/interim/train_features.json"):
    """Fine-tune both models (idt + emotion) from train_features.json."""
    # 模型一：IDT（個人/系統）
    fine_tune(json_path, IDT_TARGET, IDT_SYSTEM_PROMPT, "idt")

    # 模型二：撰寫者情緒
    fine_tune(json_path, EMOTION_TARGET, EMOTION_SYSTEM_PROMPT, "emotion")


def main():
    run()


if __name__ == "__main__":
    main()
