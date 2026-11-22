# Medical-Incident-NLP

醫療異常事件通報的 NLP 研究流程。對每筆事件描述做兩件事：

- **IDT 分類**：依 IDT（Incident Decision Tree）判定事件歸因為「個人」或「系統」。有人工標籤，以 LoRA 微調後用人工標籤評估。
- **撰寫者情緒**：從十種情緒中標出通報者行文的情緒。沒有人工真值，由模型標註，再由另一個模型評審標註與文字證據的相符程度。

流程圖見 [docs/research_flow.png](docs/research_flow.png)（SVG 原檔在同目錄），①到⑨的編號在本文與程式註解通用。完整實驗結果與討論見 [docs/experiment_report_2026-10-18.docx](docs/experiment_report_2026-10-18.docx)。

## 流程總覽

| 步驟 | 流程圖 | 指令 | 模型（正式實驗） | 輸入 → 輸出 |
|---|---|---|---|---|
| 01 資料準備 | ①② | `python -m scripts.01_prepare_data` | 不用模型 | `data/raw_data/data.csv` → `data/processed_data/` |
| 02 特徵工程 | ③④ | `python -m scripts.02_build_features` | Qwen3.5-2B（增生、情緒標註） | `train_data.json` → `data/features/` |
| 03 IDT 訓練 | ⑤ | `python -m scripts.03_train` | Qwen3.5-9B + LoRA | `train_features.json`、`val_data.json` → `models/idt_adapter/` |
| 04 推論與評估 | ⑥⑦⑧⑨ | `python -m scripts.04_inference` | Qwen3.5-9B + adapter、Qwen3.5-2B、評審 Qwen3.8-27B | `test_data.json` → `results/` |
| 05 IDT 機率評分（分析用） | ⑥⑧ 的延伸 | `python -m scripts.05_score_idt` | Qwen3.5-9B + 各 epoch adapter、不掛 adapter；TF-IDF 基準 | `val_data.json`、`test_data.json`、`models/idt_adapter/checkpoints/` → `results/idt_scores/` |

貫穿四步的設計原則：

- 先切分、再增生：改寫版本必與原文同一集合；描述相同的紀錄不跨集合。
- 驗證集與測試集不增生、不標註情緒。
- 模型只讀 `description`；批示 `directive` 保留在資料裡但不送模型。
- 資料或切分變動後，須重跑 01 並重新訓練。
- 每個呼叫模型的階段都逐筆寫進度檔，可中斷續跑；進度檔記錄模型與提示詞雜湊，不一致就拒絕續跑。

## 目前結果（2026-10-18 正式實驗）

第一次完整走完四步的結果。數字取自 `results/inference_evaluation.json`、`models/idt_adapter/training_metadata.json` 與 `data/features/features_summary.json`：

| 項目 | 結果 | 說明 |
|---|---|---|
| 資料 | 3,288 筆，清洗後 0 筆剔除 | 個人 2,976（90.5%）／系統 312（9.5%）；切成 train 2,302／val 329／test 657 |
| 增生與情緒標註 | 4,604 個改寫、6,906 筆標註 | Qwen3.5-2B；改寫缺額 0（124 次重試），情緒標註 0 次重試 |
| LoRA 訓練 | 3 epochs，21.3 小時 | train_loss 0.077 → 0.034；val_loss 0.074 → 0.125，epoch 1 之後即上升 |
| IDT accuracy | 92.09% | 全判「個人」的多數類基準 90.56% |
| IDT macro-F1 | 0.7377 | 多數類基準 0.4752 |
| 「系統」precision／recall／F1 | 60.9%／45.2%／0.518 | 62 筆系統有 34 筆被判為個人；模型只預測 46 筆系統 |
| 情緒預測分布 | 中性 58.6% | 其餘 272 筆分佈在 8 種情緒，「擔憂」未出現 |
| 評審平均總分（1 到 5） | 4.20 | 中性 5.00；非中性 3.07，其中 39.7% 總分 ≤ 2.5 |
| 執行穩定性 | 無效輸出 0 | ⑥⑦⑨ 各 657 次呼叫全部一次成功 |

IDT 混淆矩陣（列 = 人工標籤、欄 = 模型預測）：

| | 預測 個人 | 預測 系統 |
|---|---|---|
| 人工 個人 | 577 | 18 |
| 人工 系統 | 34 | 28 |

限制與後續工作整理在[已知問題與下一步](#已知問題與下一步)。

## 快速開始

### 環境

Python 3.13，conda 環境 `py313`（Mac 與 Windows 同名）。

```bash
conda create -n py313 python=3.13
conda activate py313
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 有 NVIDIA GPU 時依平台選 CUDA 版；Mac 直接 pip install torch
pip install -r requirements.txt
python -m pytest tests          # 92 個測試，用假的生成函式與假模型驗證重試、續跑、中止與指標，不載入模型
```

選用套件在 `requirements.txt` 裡註解掉，需要時再裝：`bitsandbytes`（`--load-in-4bit`，只支援 CUDA）、`langchain-ollama`、`langchain-anthropic`、`langchain-google-genai`。

### 執行規則

- 一律在專案根目錄用 `python -m scripts.<名稱>` 執行；直接 `python scripts/xxx.py` 會找不到 `utils`。
- 02 到 04 都有 `--dry-run`（只列設定、不載模型）與 `--limit N`（只跑前 N 筆，輸出改到 `<out-dir>/trial_N/`，不動正式結果）。
- 02 與 04 可中斷續跑：重跑同一指令即可；換模型、參數或提示詞後加 `--restart`。03 用 `--resume` 從最後一個 epoch 的 checkpoint 續跑。

### 不用 GPU 試跑整條流程

用小模型確認流程與輸出格式（輸出只供看格式，沒有研究意義）：

```bash
python -m scripts.01_prepare_data
python -m scripts.02_build_features --limit 2 --model Qwen/Qwen2.5-0.5B-Instruct
python -m scripts.03_train --limit 4 --epochs 1 --model Qwen/Qwen2.5-0.5B-Instruct \
    --train-features data/features/trial_2/train_features.json
python -m scripts.04_inference --limit 3 --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter models/idt_adapter/trial_4 --judge-model Qwen/Qwen2.5-0.5B-Instruct
```

輸出分別在 `data/features/trial_2/`、`models/idt_adapter/trial_4/`、`results/trial_3/`。Mac 的 MPS 上若小模型輸出異常，加 `--dtype float32`（04 的評審模型另有 `--judge-dtype`）。

### 正式執行

```bash
python -m scripts.01_prepare_data
python -m scripts.02_build_features
python -m scripts.03_train
python -m scripts.04_inference
python -m scripts.05_score_idt     # 選用：比較各 epoch、zero-shot 與 TF-IDF，挑 checkpoint、調門檻
```

預設模型見[模型設定](#模型設定)。硬體需求：

- 03 訓練、04 的 ⑥ 與 05 的 score 要載入 Qwen3.5-9B（bf16 權重約 19 GB），需要 24 GB 以上顯示記憶體；三者都沒有 4-bit 選項。
- 04 的 ⑨ 預設評審模型 Qwen3.8-27B 更大。12 GB 顯卡（RTX 3060）改用 `--judge-model Qwen/Qwen3.5-9B --judge-load-in-4bit`（nf4 約 6 GB，需 `bitsandbytes`）或更小的模型。
- 02 與 04 的 ⑦ 用 Qwen3.5-2B，12 GB 顯卡可以執行。
- 正式實驗的 03 花了 21.3 小時（3 epochs、6,906 筆、序列最長 2,579 token）。

## 目錄結構

```
scripts/              終端機入口，每個步驟一支；只解析參數、呼叫 utils、印報告
utils/                共用邏輯，不直接執行
  paths.py              所有輸入輸出路徑常數
  llm.py                模型層：LangChain chat model、預設模型常數、本機載入與釋放
  cli.py                02／04 共用的模型參數群組
  prompts.py            四種提示詞、十種情緒定義、輸出解析、提示詞雜湊
  preprocessing.py、split.py        01 清洗與分層切分
  augmentation.py、emotion.py       02 增生與情緒標註
  training.py                       03 LoRA 訓練
  inference.py、evaluation.py       04 推論、評審與指標（evaluation.py 也有 05 的 AUROC／AP／門檻／bootstrap）
  scoring.py、baseline.py           05 IDT 機率評分、adapter 搜尋與 TF-IDF 基準
  progress.py、io.py                進度檔與 JSON 讀寫
tests/                pytest，不需要模型
data/raw_data/        原始 data.csv（只讀）
data/processed_data/  01 輸出：cleaned_data、train／val／test、prepare_summary.json
data/features/        02 輸出：train_augmented.json、train_features.json、progress/、features_summary.json
models/idt_adapter/   03 輸出：adapter、tokenizer、training_metadata.json、checkpoints/epoch-N/
results/              04 輸出：inference_predictions.json、inference_evaluation.json、progress/
results/idt_scores/   05 輸出：scores_val／test.json、idt_score_evaluation.json、progress/
docs/                 流程圖與實驗報告
```

正式實驗的資料、adapter 與結果都在版本庫中，可直接分析或重算評估（`python -m scripts.04_inference --stage evaluate`）。

## 各步驟說明

### 01 資料準備（①②）

```bash
python -m scripts.01_prepare_data
python -m scripts.01_prepare_data --split-ratio 7 1 2 --split-seed 42
python -m scripts.01_prepare_data --raw-file <其他 CSV> --encoding cp950   # Big5 檔才需要 --encoding
```

1. 讀 CSV（預設 `utf-8-sig`，自動去掉 Excel 的 BOM）。
2. 清洗：NFKC 正規化 IDT 標籤並去空白；依序剔除標籤空白、非「個人／系統」、描述為空、標籤 + 描述 + 批示完全重複的紀錄；每筆補上 `source_id`（檔名:Excel 列號，標題列算第 1 列）。
3. 切分 7:1:2：依 IDT 標籤分層；描述相同的紀錄整組不拆開；順序由 `--split-seed`（預設 42）決定，與訓練的 seed 分開。

輸出到 `data/processed_data/`：`cleaned_data.json`、`train_data.json`、`val_data.json`、`test_data.json`，以及記錄輸入、剔除原因、各集合筆數與切分設定的 `prepare_summary.json`。

### 02 特徵工程（③④）

```bash
python -m scripts.02_build_features --dry-run
python -m scripts.02_build_features --limit 5                 # 輸出到 data/features/trial_5/
python -m scripts.02_build_features                           # 全部訓練集
python -m scripts.02_build_features --stage emotion           # 只重跑 ④，讀 out-dir 的 train_augmented.json
python -m scripts.02_build_features --restart                 # 丟掉舊進度重做
python -m scripts.02_build_features --augment-model Qwen/Qwen3.5-9B --emotion-model Qwen/Qwen3.5-4B   # ③④ 用不同模型
python -m scripts.02_build_features --augment-provider openai --augment-model gpt-4o-mini               # 只有 ③ 走雲端
```

只處理訓練集，不套 adapter。

**③ 增生**：每筆以溫度 0.8 產生 `--augment-n`（預設 2）個改寫，每個改寫單獨呼叫一次。輸入以 JSON 送出描述、人工 IDT 標籤與該筆已產生的版本（宣告為資料、不是指令），要求保留事件順序、因果、人物責任、醫療事實與不確定性，且不得改變 IDT 的判斷依據。改寫必須是合法 JSON、非空、與原文及所有既有描述都不同；不合格就把原因寫進提示重試，每個改寫最多 `--augment-max-attempts`（預設 4）次。湊不滿時預設保留已取得的版本並記錄缺額，`--augment-on-failure abort` 則中止。改寫沿用原文的人工 IDT 標籤與批示。

**④ 情緒標註**：溫度 0，以「具臨床心理背景的醫療文本情緒分析專家」角色，從十種情緒擇一（定義在 `utils/prompts.py` 的 `EMOTION_LABELS`）：中性、焦慮、自責、無奈、擔憂、沮喪、憤怒、驚慌、困惑、警覺；平鋪直敘的文字標「中性」。輸出無效就把原因寫進提示重試，`--emotion-max-attempts`（預設 3）次仍無效就中止，不補「中性」。`emotion_target` 是模型標註，不是人工真值。

模型設定：③④ 各有預設常數（`DEFAULT_AUGMENT_MODEL`、`DEFAULT_EMOTION_MODEL`）。`--provider`、`--model`、`--max-new-tokens`、`--dtype`、`--load-in-4bit` 是兩階段共用的覆寫；`--augment-*` 與 `--emotion-*` 只覆寫自己那一階段，沒指定的沿用共用設定；`--device` 與 `--local-files-only` 只有共用的。同一份本機權重只載入一次、兩階段共用；③④ 是不同本機模型時依序載入、用完釋放。

輸出到 `data/features/`：`train_augmented.json`（原文 + 改寫）、`train_features.json`（再加 `emotion_target`，給 03 用）、`features_summary.json`（兩階段的模型與參數、呼叫次數、缺額、情緒分布）、`progress/augment.jsonl` 與 `progress/emotion.jsonl`。

### 03 IDT LoRA 訓練（⑤）

```bash
python -m scripts.03_train --dry-run              # 只載 tokenizer，檢查標籤與 token 長度
python -m scripts.03_train --limit 8 --epochs 1   # 試跑，輸出到 models/idt_adapter/trial_8/
python -m scripts.03_train                        # 正式訓練
python -m scripts.03_train --resume               # 從最後一個 epoch 的 checkpoint 續跑，設定須相同
python -m scripts.03_train --train-features data/processed_data/train_data.json   # 不用增生資料
```

讀 `train_features.json`（只用 `description` 與人工 `idt_target`，情緒標註不參與）與 `val_data.json`。基礎模型加 LoRA 做 supervised fine-tuning：提示是完整 IDT 決策樹（Q1 到 Q10），輸入只有描述，輸出「個人」或「系統」，loss 只算標籤與結束 token。每個 epoch 結束以驗證集算 `val_loss` 與 `val_label_match`（teacher forcing 下標籤每個 token 都是 argmax 的比例），只供觀察過擬合，不用來挑 checkpoint；最終 adapter 是最後一個 epoch。

| 參數 | 預設 | 說明 |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-9B` | 基礎模型；04 的 ⑥ 必須用同一個 |
| `--epochs` | 3 | |
| `--gradient-accumulation-steps` | 8 | 每筆單獨前向，累積 8 筆更新一次 |
| `--learning-rate` | 2e-4 | warmup 比例 0.03、weight decay 0、梯度裁剪 1.0 |
| `--max-length` | 4096 | 超過直接報錯，不截斷 |
| `--lora-rank`、`--lora-alpha`、`--lora-dropout` | 8、16、0.05 | |
| `--target-modules` | `.*language_model\..*\.(q_proj\|v_proj)` | 正規表示式，整個模組名稱要匹配；預設只掛文字模型的 q_proj／v_proj |
| `--seed` | 42 | LoRA 初始化與每個 epoch 的資料順序 |
| `--no-gradient-checkpointing` | 未指定 | 預設開啟梯度檢查點以省記憶體 |

輸出到 `models/idt_adapter/`：adapter（`adapter_config.json`、`adapter_model.safetensors`）、tokenizer、`training_metadata.json`（設定、資料與 token 長度統計、匹配的模組、每個 epoch 的損失、IDT 提示詞雜湊）、`checkpoints/epoch-N/`（adapter + optimizer + 亂數狀態，`--resume` 用）。

記憶體：程式只計算標籤位置的 logits，避免整段序列乘上詞彙表的 logits 多吃數 GB。正式實驗的序列長度（完整 IDT 提示 + 描述，Qwen3.5-9B tokenizer）訓練集最長 2,579、平均 1,421 token，驗證集最長 1,945，都在 4096 內；OOM 時可先把 `--max-length` 降到 3072。

### 04 推論與評估（⑥⑦⑧⑨）

```bash
python -m scripts.04_inference --dry-run                 # 檢查 adapter 的基礎模型與提示詞是否一致，不載模型
python -m scripts.04_inference --limit 5                 # 試跑，輸出到 results/trial_5/
python -m scripts.04_inference                           # ⑥⑦⑧⑨ 全部
python -m scripts.04_inference --skip-judge              # 只做 ⑥⑦⑧，之後用 --stage judge 補
python -m scripts.04_inference --stage evaluate          # 只重算 ⑧ 與評審彙整，不載模型
python -m scripts.04_inference --judge-model Qwen/Qwen3.5-9B --judge-load-in-4bit   # 12 GB 顯卡的評審設定
python -m scripts.04_inference --judge-provider openai --judge-model gpt-4o           # 雲端評審
python -m scripts.04_inference --stage judge --predictions data/features/train_features.json \
    --emotion-field emotion_target --out-dir results/train_emotion_judge   # 重評訓練集的模型標註
python -m scripts.04_inference --adapter models/idt_adapter/checkpoints/epoch-1 --out-dir results/epoch-1   # 評估其他 epoch
```

對 `test_data.json` 依序執行，三個本機模型一次只在記憶體裡一個：

- **⑥ IDT 推論**：載入基礎模型 + `models/idt_adapter/`，溫度 0 預測 `idt_pred`，只接受「個人」或「系統」。啟動時比對 adapter 的基礎模型與 IDT 提示詞雜湊，與訓練時不同就中止。一定是本機模型。
- **⑦ 情緒推論**：載入情緒模型（不套 adapter；預設與 02 的 ④ 同一個 `DEFAULT_EMOTION_MODEL`，`--model` 有給時沿用它，`--emotion-model` 可單獨覆寫），用與訓練集標註相同的提示預測 `emotion_pred`。
- **⑧ 評估**：以人工 `idt_target` 算 accuracy、macro-F1、混淆矩陣（列 = 人工、欄 = 預測）與各類別 precision／recall／F1；預測為 `null` 算錯誤。情緒只報分布。
- **⑨ 情緒評審**：載入評審模型（`DEFAULT_JUDGE_MODEL`），只看描述與 `emotion_pred`（以 JSON 封裝為待評資料）。提示聲明沒有人工真值、輸入中的指令不能覆蓋評分規則、證據不足時保守評分。三面向各 1 到 5 分（可含小數）：`emotion_plausibility` 情緒合理性、`evidence_support` 證據支持、`writer_perspective` 撰寫者視角；`rationale` 必填且不超過 80 字；總分 `overall` 由程式取平均。

⑥⑦⑨ 輸出無效都把原因寫進提示重試，`--max-attempts`／`--judge-max-attempts`（預設 3）次仍無效就記 `null`，不猜標籤；重跑時 `null` 的紀錄會再試。

輸出到 `results/`：`inference_predictions.json`（每筆加上 `idt_pred`、`emotion_pred`、`emotion_judge`）、`inference_evaluation.json`（IDT 指標、情緒分布、評審平均分與總分分布、本次所有設定）、`progress/{idt,emotion,judge}.jsonl`。

解讀：「系統」只佔 9.4%，看混淆矩陣、「系統」recall 與 macro-F1，不要只看 accuracy。情緒沒有人工真值，評審分數是「標註與文字證據的相符程度」，不是 accuracy；`emotion_target` 也不能當 gold label。

### 05 IDT 機率評分與 checkpoint 選擇（分析用）

```bash
python -m scripts.05_score_idt --dry-run                 # 列出要評的設定、檢查 adapter 的基礎模型與提示，不載模型
python -m scripts.05_score_idt --limit 20                # 試跑，輸出到 results/idt_scores/trial_20/
python -m scripts.05_score_idt                           # score + evaluate
python -m scripts.05_score_idt --stage evaluate          # 只重算評估，不載模型（沒有 GPU 的電腦也能跑）
python -m scripts.05_score_idt --adapters models/idt_adapter/checkpoints/epoch-1 --no-zero-shot
```

04 的 ⑥ 用貪婪解碼，等於固定在 P(系統) > 0.5 才判「系統」，看不出模型有多確定，也不能比較 checkpoint 或調門檻。05 不生成文字：用與訓練相同的提示與 chat template，讀回答位置上「個人」「系統」兩個標籤第一個 token 的 log 機率，在兩者之間正規化成 P(系統)。

- **score**（要 GPU）：基礎模型只載入一次，掛上 `--adapter-dir` 底下所有 `checkpoints/epoch-N`（最上層的最終 adapter 與某個 epoch 權重相同就不重複評），再加上不掛 adapter 的 zero-shot，對驗證集與測試集逐筆評分。每個「設定 × 資料集」一個進度檔，記錄模型、精度、提示詞雜湊、adapter 權重雜湊與標籤 token；不一致就在載入模型之前拒絕續跑。
- **evaluate**（不載模型）：每個設定算驗證集與測試集的 log-loss、AUROC、AP。只在訓練出的 checkpoint 中挑驗證集 log-loss 最低的；zero-shot 與 TF-IDF 是基準，不參與挑選。每個設定在驗證集找 macro-F1 最高的門檻，報告測試集在 0.5 與該門檻的混淆矩陣與各類別指標。測試集做 bootstrap（預設 2,000 次，所有設定共用同一組重抽樣），給 95% 信賴區間與各設定減 TF-IDF 的配對差距。
- **TF-IDF 基準**：字元 1–3 gram＋邏輯迴歸（C=10、類別平衡權重），用 `--train-data`（預設 01 的原始訓練集，不含增生）訓練，同樣在驗證集調門檻。需要 scikit-learn，`--no-baseline` 可略過。
- **核對**：`--reference`（預設 04 的 `results/inference_predictions.json`）存在時，報告各 adapter 在 0.5 門檻的判斷與 04 的 `idt_pred` 一致幾筆，產生 04 結果的 adapter 應該幾乎全部一致。另報告兩個標籤的機率合計，以及「機率最高的 token 就是標籤」的比例；zero-shot 這兩個數字低，表示貪婪解碼常常不會直接輸出標籤。

輸出到 `results/idt_scores/`：`scores_val.json`、`scores_test.json`（每筆在各設定下的 P(系統)）、`idt_score_evaluation.json`（各設定的指標、選出的 checkpoint、門檻、信賴區間與基準比較）、`progress/score_<設定>_<val|test>.jsonl`。

解讀：驗證集只有 31 筆「系統」，調出的門檻與挑出的 checkpoint 都有不小的抽樣誤差。比較設定時看信賴區間；與 TF-IDF 的差距區間不含 0，才算有差。

## 模型設定

所有呼叫模型的階段（③④⑥⑦⑨）都走 LangChain 的 `BaseChatModel.invoke`，程式在 `utils/llm.py`；03 的 LoRA 訓練直接用 transformers 與 peft。

| 用途 | 常數（`utils/llm.py`） | 預設 | 執行時覆寫 |
|---|---|---|---|
| 基礎模型：03 訓練、04 的 ⑥ | `DEFAULT_MODEL` | `Qwen/Qwen3.5-9B` | 03 `--model`；04 `--model`（須與 adapter 一致） |
| ③ 增生 | `DEFAULT_AUGMENT_MODEL` | `Qwen/Qwen3.5-2B` | 02 `--model` 或 `--augment-model` |
| ④ 訓練集情緒標註、⑦ 測試集情緒推論 | `DEFAULT_EMOTION_MODEL` | `Qwen/Qwen3.5-2B` | 02／04 的 `--model` 或 `--emotion-model` |
| ⑨ 情緒評審 | `DEFAULT_JUDGE_MODEL` | `Qwen/Qwen3.8-27B` | 04 `--judge-model` |

優先順序：階段專屬選項（`--augment-*`、`--emotion-*`、`--judge-*`）> 共用選項（`--model`、`--provider` 等）> 常數。

**Provider**

- `--provider huggingface`（預設）：本機 transformers 模型，由 `LocalChatModel` 實作 `BaseChatModel`。不用 `langchain-huggingface` 的 `ChatHuggingFace`，因為它無法關閉 Gemma 4／Qwen 的思考模式、掛 LoRA adapter、控制 seed 或 4-bit 載入。
- 其他 provider 交給 `langchain.chat_models.init_chat_model`：`openai`、`ollama`、`anthropic`、`google_genai`、`groq`、`deepseek`、`mistralai`、`azure_openai` 等，需安裝對應的 `langchain-<provider>` 套件，缺了會提示要裝哪個。OpenAI 相容的本機服務（Ollama、vLLM、LM Studio）用 `--provider openai --base-url http://localhost:11434/v1 --api-key-env <任一有值的變數>`。
- API key 放專案根目錄的 `.env`（已在 `.gitignore`）或環境變數，`--api-key-env` 指定變數名。
- ⑥ 要掛 adapter，只能是本機模型；③④⑦⑨ 都可切到雲端。
- 本機模型每次呼叫都帶 temperature 與 seed，可重現；雲端 provider 多數不支援 seed，輸出不保證可重現。
- 事件描述是醫療資料，改用雲端 provider 前先確認資料可以外送。

**進度檔與提示詞雜湊**

- 02 與 04 的每個階段逐筆寫 `progress/<stage>.jsonl`，第一行記錄 provider、模型、token 上限、階段參數與提示詞雜湊；重跑時任一項不同就拒絕續跑，要加 `--restart`。
- 四種提示在 `utils/prompts.py`，雜湊為 `AUGMENT_PROMPT_ID`、`EMOTION_PROMPT_ID`、`IDT_PROMPT_ID`、`JUDGE_PROMPT_ID`；改了提示或情緒清單雜湊就變。03 把 `IDT_PROMPT_ID` 寫進 `training_metadata.json`，04 載入 adapter 時比對，不符就中止。

## 資料格式

01 的紀錄（`data/processed_data/*.json`）：

```json
{
  "source_id": "data.csv:2",
  "idt_target": "個人",
  "emotion_target": null,
  "description": "事件描述（唯一送進模型的文字）",
  "directive": "批示（保留在資料，不送模型；可為 null）"
}
```

02 的紀錄（`data/features/train_features.json`）多了 `record_id` 與 `augmented`，並填入 `emotion_target`；改寫的 `record_id` 是 `source_id#augN`，沿用原文的標籤與批示：

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

04 的紀錄（`results/inference_predictions.json`）在測試集紀錄上加三個欄位：

```json
{
  "idt_pred": "個人",
  "emotion_pred": "中性",
  "emotion_judge": {
    "scores": {"emotion_plausibility": 5, "evidence_support": 5, "writer_perspective": 5},
    "overall": 5.0,
    "rationale": "評審理由（不超過 80 字）",
    "model": "huggingface:Qwen/Qwen3.8-27B"
  }
}
```

## 已知問題與下一步

整理自實驗報告第 6 節與版本庫中的檔案狀態。

**IDT 分類**

- 「系統」召回率只有 45.2%：訓練集「系統」佔 9.5%，增生維持同樣比例，訓練沒有重新加權或過採樣，模型偏向多數類（預測 46 筆系統，實際 62 筆）。
- epoch 1 的 val_loss 最低，但最終 adapter 是 epoch 3（與 `checkpoints/epoch-3` 權重相同）。`python -m scripts.05_score_idt` 會用驗證集比較三個 checkpoint、挑 checkpoint 並調門檻。
- LoRA 只匹配 16 個模組（1,114,112 個參數，佔 0.0118%），`--target-modules` 的正規表示式似乎只涵蓋部分層；可擴大 target modules 或提高 rank。
- `val_label_match` 三個 epoch 都在 0.906 到 0.912，幾乎等於驗證集「個人」的佔比，看不出少數類的學習情形；宜改以驗證集「系統」recall 或 macro-F1 監看。
- 04 一定載入 adapter，沒有對照組；05 加入不掛 adapter 的 zero-shot 與 TF-IDF＋邏輯迴歸基準，用來量化 LoRA 的增益。
- 人工標籤有不一致案例（幾乎相同的描述一筆標「系統」、一筆標「個人」），建議抽樣複核「系統」的判準。

**情緒標註與評審**

- 版本庫裡 `train_features.json` 的 `emotion_target` 是用修訂前的情緒清單（含「驚嚇」「委屈」）標的；目前提示用「驚慌」「警覺」，測試集預測也是，兩者分布不能直接比較。02 的進度檔產生於提示詞雜湊功能之前，沒有 `prompt` 欄位，重跑 02 須加 `--restart`，會整個重做。
- 非中性標籤 272 筆平均只有 3.07 分、39.7% 總分 ≤ 2.5，「困惑」106 筆平均 2.47 最低；評審理由多為「客觀陳述、無情緒詞彙、缺乏證據」。可能是 2B 標註器過度賦予情緒，也可能是評審的保守規則系統性懲罰非中性標籤，需人工抽樣才能分辨。
- 同一原文的三個版本只有 53.3% 情緒標籤一致。評審是單一模型、溫度 0、單次，沒有做評審一致性檢驗。

**增生資料**

- 約 15.1% 的改寫是簡體中文，抽查有事實漂移，而 adapter 是用含這些改寫的 6,906 筆訓練的。對照做法：`python -m scripts.03_train --train-features data/processed_data/train_data.json` 只用原始 2,302 筆再訓練一次。
- 改善方向：改寫後以 OpenCC 轉繁或在提示中要求繁體並檢查；對改寫做事實一致性檢查；或改用較大的改寫模型。

## 變更紀錄

- 2026-10-16：改成 `scripts/` + `utils/` 結構，完成 01 到 04 的程式與測試；模型層改用 LangChain chat model 介面；開發環境改為 conda `py313`。
- 2026-10-17：提示詞改回原專案設計（完整 IDT 決策樹、十種情緒清單、改寫輸入附標籤與已產生版本、評審防注入與保守評分）；評審欄位改為 `emotion_plausibility`／`evidence_support`／`writer_perspective`／`rationale`；進度檔與 `training_metadata.json` 記錄提示詞雜湊；③④ 的模型可分開設定。
- 2026-10-17 到 18：第一次完整正式實驗（02 用 Qwen3.5-2B、03 用 Qwen3.5-9B LoRA 訓練 21.3 小時、04 評審用 Qwen3.8-27B）；資料、adapter 與結果進版本庫。
- 2026-10-18：實驗報告 `docs/experiment_report_2026-10-18.docx`（2026-10-27 更新）。
- 2026-10-27：README 重寫。
- 2026-11-20：新增 05 IDT 機率評分：各 epoch、zero-shot 與 TF-IDF 基準的 AUROC／AP，以驗證集挑 checkpoint 與調門檻，bootstrap 信賴區間；requirements 加入 scikit-learn。
- 2026-11-22：CUDA 載入模型改成每個權重先複製再搬上 GPU；在 GB10（vast.ai）上直接 `model.to("cuda")` 會卡住十分鐘以上，改後 17.5 GiB 只要 5 秒。模型權重（`*.pt`、`*.safetensors`）不再進版本庫。
