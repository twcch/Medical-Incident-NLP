import argparse

from src import data_augmentation, data_preprocessing, feature_engineering, trainer
from src import inference as predictor

# 各階段的輸出路徑
TRAIN_DATA = "data/processed_data/train_data.json"
TRAIN_AUGMENTED = "data/interim/train_augmented.json"
TRAIN_FEATURES = "data/interim/train_features.json"
TEST_DATA = "data/processed_data/test_data.json"
TEST_FEATURES = "data/interim/test_features.json"


def train():
    """訓練流程：前處理 -> 資料增強 -> 特徵工程 -> fine-tune 兩個模型。"""
    # print("=== [train] 1. 資料前處理 ===")
    # data_preprocessing.run(data_preprocessing.TRAIN_SHEETS, TRAIN_DATA)

    # print("=== [train] 2. 資料增強 ===")
    # data_augmentation.run(TRAIN_DATA, TRAIN_AUGMENTED, n=0)

    # print("=== [train] 3. 特徵工程 ===")
    # feature_engineering.run(TRAIN_AUGMENTED, TRAIN_FEATURES)

    print("=== [train] 4. 模型訓練 (fine-tune idt + emotion) ===")
    trainer.run(TRAIN_FEATURES)


def inference():
    """推論流程：前處理 -> 特徵工程 -> 模型預測（idt + emotion）。"""
    # print("=== [inference] 1. 資料前處理 ===")
    # data_preprocessing.run(data_preprocessing.TEST_SHEETS, TEST_DATA)

    # print("=== [inference] 2. 特徵工程（不增強）===")
    # feature_engineering.run(TEST_DATA, TEST_FEATURES)

    print("=== [inference] 3. 模型預測 (idt + emotion) ===")
    predictor.run(TEST_FEATURES)


def main():
    parser = argparse.ArgumentParser(description="醫療事件 NLP 資料流程")
    parser.add_argument("mode", choices=["train", "inference"], help="選擇要執行的流程")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "inference":
        inference()


if __name__ == "__main__":
    main()
