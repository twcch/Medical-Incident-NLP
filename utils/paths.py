"""專案路徑常數。

一律由本檔案的位置推回專案根目錄，不依賴執行時的工作目錄；
各步驟的輸入輸出檔名集中在此，之後的 script 只從這裡取路徑。
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# 步驟 01 資料準備：原始 CSV → 清洗 → 切分
RAW_CSV = RAW_DATA_DIR / "data.csv"
CLEANED_DATA = PROCESSED_DATA_DIR / "cleaned_data.json"
TRAIN_DATA = PROCESSED_DATA_DIR / "train_data.json"
VAL_DATA = PROCESSED_DATA_DIR / "val_data.json"
TEST_DATA = PROCESSED_DATA_DIR / "test_data.json"
SPLIT_DATA = {"train": TRAIN_DATA, "val": VAL_DATA, "test": TEST_DATA}
PREPARE_SUMMARY = PROCESSED_DATA_DIR / "prepare_summary.json"

# 步驟 02 特徵工程：訓練集增生 → 撰寫者情緒標註（只對訓練集）
TRAIN_AUGMENTED = FEATURES_DIR / "train_augmented.json"
TRAIN_FEATURES = FEATURES_DIR / "train_features.json"
FEATURES_PROGRESS_DIR = FEATURES_DIR / "progress"
FEATURES_SUMMARY = FEATURES_DIR / "features_summary.json"

# 步驟 03 訓練：IDT LoRA adapter（adapter、tokenizer、training_metadata.json、checkpoints/）
IDT_ADAPTER_DIR = MODELS_DIR / "idt_adapter"

# 步驟 04 推論：逐筆預測與評審、整體評估
INFERENCE_PREDICTIONS = RESULTS_DIR / "inference_predictions.json"
INFERENCE_EVALUATION = RESULTS_DIR / "inference_evaluation.json"
RESULTS_PROGRESS_DIR = RESULTS_DIR / "progress"
