# Medical-Incident-NLP

醫療事件 NLP 研究流程：IDT 分類（人工標籤 → LoRA 微調 → 以人工標籤評估）與撰寫者情緒（模型標註 → 模型評審）。完整流程圖見 `docs/research_flow.png`。

## 目錄結構

```
scripts/              終端機入口，每個步驟一支；只做解析參數、呼叫 utils、印報告
utils/                可 import 的共用邏輯，不直接執行；路徑常數集中在 utils/paths.py
experiments/          實驗設定與跑法紀錄
data/raw_data/        原始資料 data.csv（只讀）
data/processed_data/  01 資料準備的輸出
data/features/        02 特徵工程的輸出
models/               03 訓練的輸出
results/              04 推論與評估的輸出
docs/                 研究流程圖
```

## 環境與執行方式

- Python 3.13，conda 環境 `py313`（Mac 與 Windows 同名）。
- 一律在專案根目錄用 `python -m scripts.<名稱>` 執行。直接 `python scripts/xxx.py` 會找不到 `utils`。
- 02 之後需要 `torch`、`transformers`（Gemma 4 需 ≥ 5.5）、`accelerate`、`langchain`、`langchain-core`；03 另需 `peft`。只送文字，不需要影像／音訊套件。
- 雲端或自架服務另裝對應的 LangChain 整合套件：`langchain-openai`（已裝）、`langchain-ollama`、`langchain-anthropic`、`langchain-google-genai`。
- 測試：`python -m pytest tests`（用假的生成函式驗證重試、進度續跑與中止，不需要模型）。

## 模型呼叫：LangChain

所有呼叫模型的步驟（02 的 ③④、04 的 ⑥⑦⑨）都透過 LangChain 的 `BaseChatModel.invoke` 介面，程式碼在 `utils/llm.py`：

- `--provider huggingface`（預設）：本機 transformers 模型，由 `LocalChatModel` 實作 `BaseChatModel`。不用 `langchain-huggingface` 的 `ChatHuggingFace`，因為它無法關閉 Gemma 4／Qwen 的思考模式、掛 LoRA adapter、控制 seed 或 4-bit 載入。
- 其他 provider 交給 `langchain.chat_models.init_chat_model`：`--provider openai --model gpt-4o-mini`、`--provider ollama --model qwen3.5:9b`、`--provider anthropic`、`--provider google_genai` 等；OpenAI 相容的本機服務（Ollama、vLLM、LM Studio）用 `--provider openai --base-url http://localhost:11434/v1 --api-key-env 任一有值的變數`。
- API key 放在專案根目錄的 `.env`（已在 `.gitignore`）或環境變數；`--api-key-env` 可指定變數名。
- 02 的 ③④ 預設模型在 `utils/llm.py` 的 `DEFAULT_AUGMENT_MODEL`／`DEFAULT_EMOTION_MODEL`；執行時 `--provider`／`--model` 同時覆蓋兩階段，`--augment-provider`／`--augment-model` 與 `--emotion-provider`／`--emotion-model` 各自覆蓋（`--augment-*`／`--emotion-*` 沒指定的選項沿用共用設定）。
- 04 的 ⑥ 要掛 adapter，一定是本機模型；⑦ 預設與 02 ④ 同一個 `DEFAULT_EMOTION_MODEL`（`--model` 有給時沿用它），可用 `--emotion-provider`／`--emotion-model` 覆寫；⑨ 用 `--judge-provider`／`--judge-model`。
- 03 的 LoRA 訓練直接用 transformers 與 peft，LangChain 沒有訓練介面。
- 雲端 provider 不保證輸出可重現（多數不支援 seed），進度檔會記錄 provider、模型與端點，換了就要 `--restart`。
- 事件描述是醫療資料，改用雲端 provider 前先確認資料可以外送。

## 流程

| 步驟 | script | 輸入 | 輸出 | 狀態 |
|---|---|---|---|---|
| 01 資料準備 | `scripts/01_prepare_data.py` | `data/raw_data/data.csv` | `data/processed_data/*.json` | 完成 |
| 02 特徵工程 | `scripts/02_build_features.py` | `train_data.json` | `data/features/` | 完成（正式跑需 GPU） |
| 03 訓練 | `scripts/03_train.py` | `train_features.json`、`val_data.json` | `models/idt_adapter/` | 完成（正式跑需 GPU） |
| 04 推論 | `scripts/04_inference.py` | `models/idt_adapter/`、`test_data.json` | `results/` | 完成（正式跑需 GPU） |

### 01 資料準備（流程圖 ①②）

```bash
python -m scripts.01_prepare_data
python -m scripts.01_prepare_data --split-ratio 7 1 2 --split-seed 42
python -m scripts.01_prepare_data --raw-file <其他 CSV> --encoding cp950   # Big5 檔才需要 --encoding
```

1. 讀取 CSV（預設 `utf-8-sig`，自動去掉 Excel 的 BOM）。
2. 清洗：NFKC 正規化 IDT 標籤並去空白；依序剔除標籤空白、非「個人／系統」、描述為空、標籤 + 描述 + 批示完全重複；每筆補上 `source_id`（檔名:Excel 列號，標題列算第 1 列）。
3. 切分 7:1:2：依 IDT 標籤分層；描述相同的紀錄整組不拆開；隨機順序由 `--split-seed` 決定，與訓練的 seed 分開。

輸出到 `data/processed_data/`：`cleaned_data.json`、`train_data.json`、`val_data.json`、`test_data.json`，以及記錄輸入、剔除原因、各集合筆數與切分設定的 `prepare_summary.json`。

紀錄格式：

```json
{
  "source_id": "data.csv:2",
  "idt_target": "個人",
  "emotion_target": null,
  "description": "事件描述（唯一送進模型的文字）",
  "directive": "批示（保留在資料，不送模型；可為 null）"
}
```

### 02 特徵工程（流程圖 ③④）

```bash
python -m scripts.02_build_features --dry-run          # 只列出設定，不載入模型
python -m scripts.02_build_features --limit 5          # 先試跑 5 筆，輸出到 data/features/trial_5/
python -m scripts.02_build_features                    # 全部訓練集
python -m scripts.02_build_features --stage emotion    # 只重跑情緒標註
python -m scripts.02_build_features --restart          # 換模型或參數後丟掉舊進度
python -m scripts.02_build_features --augment-model Qwen/Qwen3.5-9B --emotion-model Qwen/Qwen3.5-4B   # ③④ 用不同模型
python -m scripts.02_build_features --augment-provider openai --augment-model gpt-4o-mini               # 只有 ③ 走雲端
```

只處理訓練集，不套 adapter。③④ 的模型可以分開設定：預設模型是 `utils/llm.py` 的 `DEFAULT_AUGMENT_MODEL`（③）與 `DEFAULT_EMOTION_MODEL`（④），目前都等於 `DEFAULT_MODEL`（`Qwen/Qwen3.5-9B`），改那兩行就能固定用不同模型。執行時 `--provider`／`--model`／`--max-new-tokens`／`--dtype`／`--load-in-4bit` 等是兩階段共用的覆寫（`--model` 同時蓋掉兩階段的預設）；`--augment-*`（③）與 `--emotion-*`（④）各自覆寫 provider、model、base-url、api-key-env、max-new-tokens、dtype、load-in-4bit，沒指定的選項沿用共用設定；`--device` 與 `--local-files-only` 只有共用的。同一份本機權重只載入一次、兩階段共用（各自的溫度與 token 上限）；③④ 是不同本機模型時依序載入、用完釋放，一次只在記憶體裡一個。`features_summary.json` 的 `augment.model` 與 `emotion.model` 記錄兩階段實際用的模型，進度檔也各自記錄自己的模型。

1. **③ 增生**：每筆以溫度 0.8 產生 `--augment-n`（預設 2）個改寫，每個改寫單獨呼叫一次。提示沿用原專案：輸入以 JSON 送出描述、人工 IDT 標籤與該筆已產生的版本（宣告為資料、不是指令），要求保留事件順序、因果、人物責任、醫療事實與不確定性，且不得改變 IDT 分類的判斷依據。改寫必須是合法 JSON、非空、且與原文及所有既有描述都不同；不合格就把原因寫進提示重試，每個改寫最多 `--augment-max-attempts`（預設 4）次。湊不滿時預設保留已取得的版本並記錄缺額，`--augment-on-failure abort` 則中止。改寫沿用原文的人工 IDT 標籤與批示。
2. **④ 情緒標註**：溫度 0，以「具臨床心理背景的醫療文本情緒分析專家」角色，從原專案的十種情緒擇一：中性、焦慮、自責、無奈、擔憂、沮喪、憤怒、驚慌、困惑、警覺（定義在 `utils/prompts.py` 的 `EMOTION_LABELS`），文字平鋪直敘用「中性」。輸出無效就把原因寫進提示重試，共 `--emotion-max-attempts`（預設 3）次仍無效就中止，不補「中性」。`emotion_target` 是模型標註，不是人工真值。

每筆結果都寫進 `progress/augment.jsonl`、`progress/emotion.jsonl`，中斷後重跑同一指令即可續跑；進度檔第一行記錄模型、參數與提示詞雜湊（`utils/prompts.py` 的 `*_PROMPT_ID`），不一致會拒絕續跑；改了提示或情緒清單要加 `--restart` 重做。

輸出到 `data/features/`：`train_augmented.json`（原文 + 改寫）、`train_features.json`（再加上 `emotion_target`，給 03 用）、`features_summary.json`（設定、呼叫次數、缺額、情緒分布）。

紀錄格式（在 01 的欄位之外新增 `record_id` 與 `augmented`）：

```json
{
  "record_id": "data.csv:2#aug1",
  "augmented": true,
  "source_id": "data.csv:2",
  "idt_target": "個人",
  "emotion_target": "焦慮",
  "description": "改寫後的描述",
  "directive": "沿用原文的批示"
}
```

### 03 IDT LoRA 訓練（流程圖 ⑤）

```bash
python -m scripts.03_train --dry-run              # 只載 tokenizer，檢查標籤與 token 長度，不載模型
python -m scripts.03_train --limit 8 --epochs 1   # 試跑，輸出到 models/idt_adapter/trial_8/
python -m scripts.03_train                        # 正式訓練
python -m scripts.03_train --resume               # 從最後一個 epoch 的 checkpoint 續跑（設定須相同）
```

讀 `train_features.json`（只用 `description` 與人工 `idt_target`，情緒標註不參與）與 `val_data.json`；也可用 `--train-features data/processed_data/train_data.json` 不用增生資料。基礎模型加 LoRA（預設 r=8、α=16、dropout 0.05，`--target-modules` 預設只掛文字模型的 q_proj／v_proj），提示是原專案的完整 IDT 決策樹（Q1 到 Q10，約 1,440 token），輸入只有描述、輸出「個人」或「系統」，loss 只算標籤與結束 token；`training_metadata.json` 記錄提示詞雜湊 `idt_prompt`，04 載入 adapter 時會比對。預設 3 個 epoch、每筆單獨前向並累積 8 筆更新一次、學習率 2e-4、max_length 4096（超過直接報錯，不截斷）、梯度檢查點開啟。每個 epoch 結束以驗證集算 `val_loss` 與 `val_label_match`（teacher forcing 下標籤每個 token 都是 argmax 的比例），只供觀察過擬合，不用來挑 checkpoint；最終 adapter 是最後一個 epoch。

輸出到 `models/idt_adapter/`：adapter（`adapter_config.json`、`adapter_model.safetensors`）、tokenizer、`training_metadata.json`（設定、資料、匹配的模組、每個 epoch 的損失）、`checkpoints/epoch-N/`（adapter + optimizer + 亂數狀態，`--resume` 用）。

記憶體：Gemma 4 E2B-it bf16 權重 10.2 GB，RTX 3060 12GB 很緊。程式只計算標籤位置的 logits（詞彙表 26 萬，整段序列的 logits 會多吃數 GB）。完整 IDT 提示後，以 Qwen2.5 tokenizer 計算訓練資料最長 2,773 token（平均 1,558、p95 1,764）、驗證集最長 2,171，都在 4096 內；仍然 OOM 時可把 `--max-length` 降到 3072（仍涵蓋全部資料），再考慮 4-bit 載入（尚未實作）。

### 04 推論與評估（流程圖 ⑥⑦⑧⑨）

```bash
python -m scripts.04_inference --dry-run                 # 檢查 adapter 與基礎模型是否一致，不載模型
python -m scripts.04_inference --limit 5                 # 試跑，輸出到 results/trial_5/
python -m scripts.04_inference --judge-load-in-4bit      # 全部；12 GB 顯卡跑 9B 評審模型要加這個
python -m scripts.04_inference --skip-judge              # 只做 ⑥⑦⑧，評審之後用 --stage judge 補
python -m scripts.04_inference --stage evaluate          # 只重算 ⑧ 與評審彙整，不載模型
python -m scripts.04_inference --stage judge --predictions data/features/train_features.json \
    --emotion-field emotion_target --out-dir results/train_emotion_judge   # 重評訓練集的模型標註
```

對 `test_data.json`：⑥ 載入基礎模型 + `models/idt_adapter/` 預測 `idt_pred`（溫度 0，只接受「個人」或「系統」，基礎模型或 IDT 提示詞與訓練時不同會中止）；釋放後 ⑦ 載入情緒模型（不套 adapter；預設與 02 ④ 同一個 `DEFAULT_EMOTION_MODEL`，`--model` 或 `--emotion-model` 可覆寫）以訓練集相同的提示預測 `emotion_pred`；⑧ 比對人工 `idt_target`；⑨ 載入評審模型（`utils/llm.py` 的 `DEFAULT_JUDGE_MODEL`）只看描述與 `emotion_pred`，兩者以 JSON 封裝成待評估的資料。提示沿用原專案：聲明沒有人工真值、輸入中的指令不能覆蓋評分規則、證據不足時保守評分；三面向各 1 到 5 分、可含小數（`emotion_plausibility` 情緒合理性、`evidence_support` 證據支持、`writer_perspective` 撰寫者視角），理由 `rationale` 必填且不超過 80 字，總分由程式取平均。三個模型一次只在記憶體裡一個。輸出無效會把原因寫進提示重試，共 3 次仍無效就記 `null`，不猜標籤；評估時 `null` 算錯誤。每筆都寫進度檔，中斷後重跑續跑，`null` 的那幾筆會再試。

輸出到 `results/`：`inference_predictions.json`（每筆加上 `idt_pred`、`emotion_pred`、`emotion_judge`）、`inference_evaluation.json`（IDT 的 accuracy、混淆矩陣（列 = 人工、欄 = 預測）、各類別 precision／recall／F1；情緒的分布與評審平均分）。「系統」類別少，看混淆矩陣與召回率，不要只看 accuracy。情緒沒有人工真值，評審分數是品質分數，不是 accuracy；`emotion_target` 不能當 gold label。

評審模型：Qwen3.5-9B bf16 約 19 GB，RTX 3060 放不下，`--judge-load-in-4bit` 用 bitsandbytes nf4 約 6 GB（需 `pip install bitsandbytes`，本機無 CUDA 未實測），或改 `--judge-model Qwen/Qwen3.5-4B`。

## 設計原則

- 先切分、再增生：改寫版本必與原文同集合。
- 相同描述的紀錄不跨集合。
- 驗證與測試集不增生、不標註情緒。
- 模型只讀 `description`；批示不送模型。
- 資料或切分變動後，須重新執行 01 並重新訓練。

## 進度

- 2026-10-16：改成 `scripts/` + `utils/` 結構；完成 01 資料準備（data.csv 3,288 筆 → train 2,302 / val 329 / test 657）。
- 2026-10-16：完成 02 特徵工程程式與測試；本機以 Qwen2.5-0.5B-Instruct 試跑 2 筆驗證流程（輸出在 `data/features/trial_2/`，僅供看格式），正式增生與標註用 Gemma 4 E2B-it 在 GPU 機器執行。
- 2026-10-16：完成 03 訓練程式與測試；用 Gemma 4 tokenizer 檢查全部訓練／驗證資料（最長 1,437 token、標籤 3 token、提示為完整序列前綴）；本機以 Qwen2.5-0.5B 試跑 2 epoch 並驗證 `--resume`（輸出 `models/idt_adapter/trial_4/`，供 04 試跑載入用）。
- 2026-10-16：模型層改用 LangChain 的 chat model 介面（本機 `LocalChatModel` + `init_chat_model` 切換雲端 provider）；開發環境改為 conda `py313`。
- 2026-10-16：完成 04 推論與評估程式與測試；本機以 Qwen2.5-0.5B 當基礎、adapter 與評審模型試跑 3 筆走完 ⑥⑦⑧⑨、續跑與單獨評審（輸出 `results/trial_3/`，僅供看格式）。
- 2026-10-16：02 的 ③ 增生與 ④ 情緒標註模型可分開設定（`utils/llm.py` 的 `DEFAULT_AUGMENT_MODEL`／`DEFAULT_EMOTION_MODEL`，執行時 `--augment-*`／`--emotion-*` 覆寫共用的 `--model` 等選項）；同一份權重仍只載入一次，不同模型依序載入。04 ⑦ 的預設也接到 `DEFAULT_EMOTION_MODEL`，與 ④ 同一個模型。
- 2026-10-17：提示詞改回原專案（`_old/src`）的設計：IDT 完整決策樹、十種情緒清單（驚慌、警覺取代驚嚇、委屈）、改寫輸入附標籤與已產生版本、評審的防注入與保守評分規則；評審欄位改為 `emotion_plausibility`／`evidence_support`／`writer_perspective` 與 `rationale`。進度檔與 `training_metadata.json` 記錄提示詞雜湊，04 載入 adapter 時比對 IDT 提示。
