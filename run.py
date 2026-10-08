"""醫療事件 NLP 的命令入口，串接資料準備、IDT 訓練、推論與情緒評分。

各階段以 JSON 檔案交接，方便更新資料、重用中間結果或單獨重試評分。
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from src import data_augmentation, data_preprocessing, evaluation, feature_engineering, trainer
from src import inference as predictor
from src.config import RuntimeConfig, TrainingConfig

# 清洗、增生與情緒標註各自保留輸出，便於核對哪個階段改變了資料。
TRAIN_DATA = "data/processed_data/train_data.json"
TRAIN_AUGMENTED = "data/interim/train_augmented.json"
TRAIN_FEATURES = "data/interim/train_features.json"
TEST_DATA = "data/processed_data/test_data.json"


def prepare(
    *,
    raw_file: str | Path = data_preprocessing.RAW_FILE,
    train_data: str | Path = TRAIN_DATA,
    train_augmented: str | Path = TRAIN_AUGMENTED,
    train_features: str | Path = TRAIN_FEATURES,
    train_sheets: Sequence[str] | None = None,
    augment_n: int = 2,
    config: RuntimeConfig | None = None,
) -> list:
    """依序產生清洗資料、改寫資料與帶情緒標註的訓練特徵。

    增生沿用人工 IDT 標籤；新增的情緒欄位只記錄模型標註，不是人工真值。
    回傳最後的紀錄陣列，同時保留三個階段的 JSON 輸出。
    """
    config = config or RuntimeConfig()
    # None 表示使用預設月份；空清單交由資料層回報，避免靜默改讀預設資料。
    sheets = data_preprocessing.TRAIN_SHEETS if train_sheets is None else train_sheets
    print("=== 1. 訓練資料讀取與清洗 ===")
    data_preprocessing.run(sheets, train_data, raw_file=raw_file)
    print("=== 2. 保留 y（人工 IDT 標籤）的資料增生 ===")
    data_augmentation.run(train_data, train_augmented, n=augment_n, config=config)
    print("=== 3. Gemma 撰寫者情緒標註 ===")
    return feature_engineering.run(train_augmented, train_features, config=config)


def train(
    *,
    reuse_prepared: bool = False,
    config: RuntimeConfig | None = None,
    training_config: TrainingConfig | None = None,
    models_dir: str | Path = "models",
    interim_dir: str | Path = "data/interim",
    **prepare_options,
) -> str:
    """訓練 IDT adapter 並回傳其目錄；prepare_options 沿用 prepare 的參數。

    reuse_prepared=True 時直接讀取 train_features，省去資料準備與情緒標註。
    """
    if not reuse_prepared:
        prepare(config=config, **prepare_options)
    train_features = prepare_options.get("train_features", TRAIN_FEATURES)
    print("=== 4. Gemma IDT LoRA fine-tuning ===")
    return trainer.run(
        train_features,
        config=config,
        training_config=training_config,
        models_dir=models_dir,
        interim_dir=interim_dir,
    )


def inference(
    *,
    raw_file: str | Path = data_preprocessing.RAW_FILE,
    test_data: str | Path = TEST_DATA,
    reuse_prepared: bool = False,
    config: RuntimeConfig | None = None,
    models_dir: str | Path = "models",
    out_dir: str | Path = "results",
    test_sheets: Sequence[str] | None = None,
    evaluate_emotion: bool = True,
) -> list:
    """測試集只清洗與推論，不增生；回傳 IDT 與情緒的評估摘要。

    重用資料只省去 Excel 清洗，仍重新產生預測；可將 Qwen 評分留到之後。
    """
    if not reuse_prepared:
        sheets = data_preprocessing.TEST_SHEETS if test_sheets is None else test_sheets
        print("=== 測試資料讀取與清洗（不增生）===")
        data_preprocessing.run(sheets, test_data, raw_file=raw_file)
    print("=== Gemma IDT 推論、情緒標註與 Qwen 評分 ===")
    return predictor.run(
        test_data,
        out_dir=out_dir,
        models_dir=models_dir,
        config=config,
        evaluate_emotion=evaluate_emotion,
    )


def build_parser() -> argparse.ArgumentParser:
    """CLI 預設值直接取自設定類別，避免兩處設定隨維護而分歧。"""
    runtime_defaults = RuntimeConfig()
    training_defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Gemma 醫療事件 IDT LoRA 與 Qwen 情緒評分流程")
    parser.add_argument(
        "mode",
        choices=["prepare", "train", "inference", "evaluate", "all"],
        help="prepare 資料準備；train 準備+LoRA；inference 推論+評分；evaluate 單獨評分；all 全流程",
    )
    # 資料來源與各階段輸出可分別指定，讓更新月份及多次實驗各自保留結果。
    parser.add_argument("--raw-file", default=data_preprocessing.RAW_FILE)
    parser.add_argument(
        "--train-sheets",
        nargs="+",
        default=data_preprocessing.TRAIN_SHEETS,
        help="訓練工作表，可依更新後資料指定月份",
    )
    parser.add_argument(
        "--test-sheets",
        nargs="+",
        default=data_preprocessing.TEST_SHEETS,
        help="測試工作表，可依更新後資料指定月份",
    )
    parser.add_argument("--train-data", default=TRAIN_DATA)
    parser.add_argument("--train-augmented", default=TRAIN_AUGMENTED)
    parser.add_argument("--train-features", default=TRAIN_FEATURES)
    parser.add_argument("--test-data", default=TEST_DATA)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--augment-n", type=int, default=2, help="每筆新增的改寫數，0 表示不增生")
    # 重用資料與略過評分是不同控制：前者省去準備，後者延後載入 Qwen。
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="train 重用 train-features；inference 重用 test-data",
    )
    parser.add_argument(
        "--skip-emotion-evaluation",
        action="store_true",
        help="推論後暫不載入 Qwen，之後以 evaluate 評分",
    )
    parser.add_argument(
        "--evaluation-input", help="evaluate 讀取的 JSON，預設 out-dir/inference_predictions.json"
    )
    parser.add_argument(
        "--prediction-field", choices=["emotion_pred", "emotion_target"], default="emotion_pred"
    )
    # 執行設定由所有模型階段共用；本機模型路徑與 Hugging Face ID 使用相同介面。
    parser.add_argument(
        "--gemma-model", default=runtime_defaults.gemma_model, help="Hugging Face ID 或本機路徑"
    )
    parser.add_argument(
        "--judge-model", default=runtime_defaults.judge_model, help="Hugging Face ID 或本機路徑"
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default=runtime_defaults.device
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default=runtime_defaults.dtype,
    )
    parser.add_argument("--local-files-only", action="store_true", help="僅使用已下載的模型")
    parser.add_argument("--seed", type=int, default=runtime_defaults.seed)
    parser.add_argument("--max-new-tokens", type=int, default=runtime_defaults.max_new_tokens)
    # 下列參數只用於 IDT LoRA；情緒標註與 Qwen 評分不進行 fine-tuning。
    parser.add_argument("--epochs", type=float, default=training_defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=training_defaults.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=training_defaults.gradient_accumulation_steps,
    )
    parser.add_argument("--learning-rate", type=float, default=training_defaults.learning_rate)
    parser.add_argument("--max-length", type=int, default=training_defaults.max_length)
    parser.add_argument("--lora-rank", type=int, default=training_defaults.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=training_defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=training_defaults.lora_dropout)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="列出設定及執行階段，不下載模型或寫入資料"
    )
    return parser


def planned_stages(args: argparse.Namespace) -> list[str]:
    """依實際執行順序組合預覽，避免靠刪除同名階段混淆兩次情緒標註。"""
    stages = []
    if args.mode in {"prepare", "train", "all"}:
        if args.mode == "prepare" or not args.reuse_prepared:
            stages.extend(["讀取／清洗訓練集", "保留 y 增生", "Gemma 情緒標註"])
        if args.mode in {"train", "all"}:
            stages.append("IDT LoRA 訓練")
    if args.mode in {"inference", "all"}:
        if not args.reuse_prepared:
            stages.append("讀取／清洗測試集")
        # 重用清洗後的測試集，仍需重新產生 IDT 與情緒預測。
        stages.extend(["IDT LoRA 推論", "Gemma 情緒標註"])
        if not args.skip_emotion_evaluation:
            stages.append("Qwen 情緒評分")
    if args.mode == "evaluate":
        stages.append("Qwen 情緒評分")
    return stages


def main(argv: Sequence[str] | None = None) -> list | str | None:
    """先驗證設定，再分派階段；argv=None 時讀取命令列。

    prepare 回傳紀錄、train 回傳 adapter 路徑，其餘執行模式回傳評估摘要；
    dry-run 只列出設定，在任何資料或模型操作之前返回 None。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.augment_n < 0:
        parser.error("--augment-n 不可為負數。")
    # 在讀取 Excel 之前檢查月份交集，避免同一工作表同時進入訓練與測試。
    if args.mode in {"prepare", "train", "inference", "all"} and set(args.train_sheets) & set(
        args.test_sheets
    ):
        parser.error("--train-sheets 與 --test-sheets 不可重疊，避免訓練／測試資料洩漏。")
    try:
        config = RuntimeConfig(
            gemma_model=args.gemma_model,
            judge_model=args.judge_model,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only,
            seed=args.seed,
        )
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        )
    except ValueError as exc:
        # 將設定類別的驗證錯誤轉成 CLI 用法錯誤，讓命令列回傳一致的失敗狀態。
        parser.error(str(exc))

    if args.dry_run:
        # 此處只輸出設定值，不檢查檔案、GPU 或模型權限，也不建立任何輸出目錄。
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "stages": planned_stages(args),
                    "runtime": asdict(config),
                    "training": asdict(training_config),
                    "augment_n": args.augment_n,
                    "paths": {
                        "raw_file": args.raw_file,
                        "train_data": args.train_data,
                        "train_sheets": args.train_sheets,
                        "test_sheets": args.test_sheets,
                        "train_augmented": args.train_augmented,
                        "train_features": args.train_features,
                        "test_data": args.test_data,
                        "idt_adapter": str(Path(args.models_dir) / "idt_adapter"),
                        "out_dir": args.out_dir,
                        "evaluation_input": args.evaluation_input
                        or str(Path(args.out_dir) / "inference_predictions.json"),
                        "prediction_field": args.prediction_field,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    prepare_options = {
        "raw_file": args.raw_file,
        "train_data": args.train_data,
        "train_sheets": args.train_sheets,
        "train_augmented": args.train_augmented,
        "train_features": args.train_features,
        "augment_n": args.augment_n,
    }
    if args.mode == "prepare":
        return prepare(config=config, **prepare_options)
    if args.mode in {"train", "all"}:
        # all 必須先完成 adapter 訓練，再讓下方推論使用相同 models_dir。
        result = train(
            config=config,
            training_config=training_config,
            reuse_prepared=args.reuse_prepared,
            models_dir=args.models_dir,
            interim_dir=str(Path(args.train_features).parent),
            **prepare_options,
        )
        if args.mode == "train":
            return result
    if args.mode in {"inference", "all"}:
        return inference(
            raw_file=args.raw_file,
            test_data=args.test_data,
            test_sheets=args.test_sheets,
            reuse_prepared=args.reuse_prepared,
            config=config,
            models_dir=args.models_dir,
            out_dir=args.out_dir,
            evaluate_emotion=not args.skip_emotion_evaluation,
        )
    if args.mode == "evaluate":
        # 直接讀取既有情緒標註或預測；獨立評分不需要重新載入 Gemma。
        return evaluation.run(
            args.evaluation_input or str(Path(args.out_dir) / "inference_predictions.json"),
            out_dir=args.out_dir,
            config=config,
            prediction_field=args.prediction_field,
        )


if __name__ == "__main__":
    main()
