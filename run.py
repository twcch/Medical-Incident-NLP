"""完整資料流程與可重用的分階段命令。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src import data_augmentation, data_preprocessing, feature_engineering, trainer
from src import evaluation, inference as predictor
from src.config import RuntimeConfig, TrainingConfig

TRAIN_DATA = "data/processed_data/train_data.json"
TRAIN_AUGMENTED = "data/interim/train_augmented.json"
TRAIN_FEATURES = "data/interim/train_features.json"
TEST_DATA = "data/processed_data/test_data.json"


def prepare(*, raw_file=data_preprocessing.RAW_FILE, train_data=TRAIN_DATA,
            train_augmented=TRAIN_AUGMENTED, train_features=TRAIN_FEATURES,
            train_sheets=None, augment_n=2, config=None):
    config = config or RuntimeConfig()
    print("=== 1. 訓練資料讀取與清洗 ===")
    data_preprocessing.run(train_sheets or data_preprocessing.TRAIN_SHEETS, train_data, raw_file=raw_file)
    print("=== 2. 保留 y（人工 IDT 標籤）的資料增生 ===")
    data_augmentation.run(train_data, train_augmented, n=augment_n, config=config)
    print("=== 3. Gemma 撰寫者情緒標註 ===")
    return feature_engineering.run(train_augmented, train_features, config=config)


def train(*, reuse_prepared=False, config=None, training_config=None,
          models_dir="models", interim_dir="data/interim", **prepare_options):
    if not reuse_prepared:
        prepare(config=config, **prepare_options)
    train_features = prepare_options.get("train_features", TRAIN_FEATURES)
    print("=== 4. Gemma IDT LoRA fine-tuning ===")
    return trainer.run(
        train_features, config=config, training_config=training_config,
        models_dir=models_dir, interim_dir=interim_dir,
    )


def inference(*, raw_file=data_preprocessing.RAW_FILE, test_data=TEST_DATA,
              reuse_prepared=False, config=None, models_dir="models",
              out_dir="results", test_sheets=None, evaluate_emotion=True):
    if not reuse_prepared:
        print("=== 測試資料讀取與清洗（不增生）===")
        data_preprocessing.run(test_sheets or data_preprocessing.TEST_SHEETS, test_data, raw_file=raw_file)
    print("=== Gemma IDT 推論、情緒標註與 Qwen 評分 ===")
    return predictor.run(
        test_data, out_dir=out_dir, models_dir=models_dir, config=config,
        evaluate_emotion=evaluate_emotion,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Gemma 醫療事件 IDT LoRA 與 Qwen 情緒評分流程")
    parser.add_argument("mode", choices=["prepare", "train", "inference", "evaluate", "all"],
                        help="prepare 資料準備；train 準備+LoRA；inference 推論+評分；evaluate 單獨評分；all 全流程")
    parser.add_argument("--raw-file", default=data_preprocessing.RAW_FILE)
    parser.add_argument("--train-sheets", nargs="+", default=data_preprocessing.TRAIN_SHEETS,
                        help="訓練工作表，可依更新後資料指定月份")
    parser.add_argument("--test-sheets", nargs="+", default=data_preprocessing.TEST_SHEETS,
                        help="測試工作表，可依更新後資料指定月份")
    parser.add_argument("--train-data", default=TRAIN_DATA)
    parser.add_argument("--train-augmented", default=TRAIN_AUGMENTED)
    parser.add_argument("--train-features", default=TRAIN_FEATURES)
    parser.add_argument("--test-data", default=TEST_DATA)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--augment-n", type=int, default=2, help="每筆新增的改寫數，0 表示不增生")
    parser.add_argument("--reuse-prepared", action="store_true", help="train 重用 train-features；inference 重用 test-data")
    parser.add_argument("--skip-emotion-evaluation", action="store_true", help="推論後暫不載入 Qwen，之後以 evaluate 評分")
    parser.add_argument("--evaluation-input", help="evaluate 讀取的 JSON，預設 out-dir/inference_predictions.json")
    parser.add_argument("--prediction-field", choices=["emotion_pred", "emotion_target"], default="emotion_pred")
    parser.add_argument("--gemma-model", default=RuntimeConfig.gemma_model, help="Hugging Face ID 或本機路徑")
    parser.add_argument("--judge-model", default=RuntimeConfig.judge_model, help="Hugging Face ID 或本機路徑")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--local-files-only", action="store_true", help="僅使用已下載的模型")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="列出設定及執行階段，不下載模型或寫入資料")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.augment_n < 0:
        parser.error("--augment-n 不可為負數。")
    if args.mode in {"prepare", "train", "inference", "all"} and set(args.train_sheets) & set(args.test_sheets):
        parser.error("--train-sheets 與 --test-sheets 不可重疊，避免訓練／測試資料洩漏。")
    try:
        config = RuntimeConfig(
            gemma_model=args.gemma_model, judge_model=args.judge_model,
            device=args.device, dtype=args.dtype, max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only, seed=args.seed,
        )
        training_config = TrainingConfig(
            epochs=args.epochs, batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate, max_length=args.max_length,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        )
    except ValueError as exc:
        parser.error(str(exc))

    stages = {
        "prepare": ["讀取／清洗訓練集", "保留 y 增生", "Gemma 情緒標註"],
        "train": ["讀取／清洗訓練集", "保留 y 增生", "Gemma 情緒標註", "IDT LoRA 訓練"],
        "inference": ["讀取／清洗測試集", "IDT LoRA 推論", "Gemma 情緒標註", "Qwen 情緒評分"],
        "evaluate": ["Qwen 情緒評分"],
        "all": ["讀取／清洗訓練集", "保留 y 增生", "Gemma 情緒標註", "IDT LoRA 訓練",
                "讀取／清洗測試集", "IDT LoRA 推論", "Gemma 情緒標註", "Qwen 情緒評分"],
    }
    planned = stages[args.mode]
    if args.reuse_prepared and args.mode in {"train", "inference", "all"}:
        skip = {"讀取／清洗訓練集", "保留 y 增生", "讀取／清洗測試集"}
        planned = [stage for stage in planned if stage not in skip]
        if args.mode in {"train", "all"}:
            planned.remove("Gemma 情緒標註")
    if args.skip_emotion_evaluation and args.mode in {"inference", "all"}:
        planned.remove("Qwen 情緒評分")
    if args.dry_run:
        print(json.dumps({
            "mode": args.mode, "stages": planned, "runtime": asdict(config),
            "training": asdict(training_config), "augment_n": args.augment_n,
            "paths": {
                "raw_file": args.raw_file, "train_data": args.train_data,
                "train_sheets": args.train_sheets, "test_sheets": args.test_sheets,
                "train_augmented": args.train_augmented, "train_features": args.train_features,
                "test_data": args.test_data, "idt_adapter": str(Path(args.models_dir) / "idt_adapter"),
                "out_dir": args.out_dir,
                "evaluation_input": args.evaluation_input or str(Path(args.out_dir) / "inference_predictions.json"),
                "prediction_field": args.prediction_field,
            },
        }, ensure_ascii=False, indent=2))
        return

    prepare_options = {
        "raw_file": args.raw_file, "train_data": args.train_data,
        "train_sheets": args.train_sheets,
        "train_augmented": args.train_augmented, "train_features": args.train_features,
        "augment_n": args.augment_n,
    }
    if args.mode == "prepare":
        return prepare(config=config, **prepare_options)
    if args.mode in {"train", "all"}:
        result = train(
            config=config, training_config=training_config,
            reuse_prepared=args.reuse_prepared, models_dir=args.models_dir,
            interim_dir=str(Path(args.train_features).parent), **prepare_options,
        )
        if args.mode == "train":
            return result
    if args.mode in {"inference", "all"}:
        return inference(
            raw_file=args.raw_file, test_data=args.test_data,
            test_sheets=args.test_sheets,
            reuse_prepared=args.reuse_prepared, config=config,
            models_dir=args.models_dir, out_dir=args.out_dir,
            evaluate_emotion=not args.skip_emotion_evaluation,
        )
    if args.mode == "evaluate":
        return evaluation.run(
            args.evaluation_input or str(Path(args.out_dir) / "inference_predictions.json"),
            out_dir=args.out_dir, config=config, prediction_field=args.prediction_field,
        )


if __name__ == "__main__":
    main()
