"""步驟 03 IDT LoRA 訓練，對應流程圖的 ⑤。

以 02 產生的 train_features.json（description + 人工 idt_target；情緒標註不參與）對基礎模型做
LoRA supervised fine-tuning，輸出「個人」或「系統」。每個 epoch 結束以 val_data.json 計算
驗證損失，只供觀察過擬合，不用來挑 checkpoint。

在專案根目錄執行：
    python -m scripts.03_train --dry-run                # 只載 tokenizer，檢查標籤與 token 長度
    python -m scripts.03_train --limit 8 --epochs 1     # 試跑，輸出到 models/idt_adapter/trial_8/
    python -m scripts.03_train                          # 正式訓練
    python -m scripts.03_train --resume                 # 從最後一個 epoch 的 checkpoint 續跑
    python -m scripts.03_train --train-features data/processed_data/train_data.json   # 不用增生資料

輸出（預設 models/idt_adapter/）：
    adapter_config.json、adapter_model.safetensors    LoRA adapter（最後一個 epoch）
    tokenizer 檔案                                    推論時與 adapter 一起載入
    training_metadata.json                            設定、資料、匹配的模組、每個 epoch 的損失
    checkpoints/epoch-N/                              每個 epoch 的 adapter 與 optimizer 狀態
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from utils import io, paths
from utils.llm import (
    DEFAULT_MODEL,
    DEVICE_CHOICES,
    DTYPE_CHOICES,
    ModelConfig,
    load_base_model,
    load_tokenizer,
    release,
)
from utils.preprocessing import count_labels
from utils.prompts import IDT_PROMPT_ID
from utils.training import (
    CHECKPOINT_DIR,
    DEFAULT_TARGET_MODULES,
    TrainingConfig,
    attach_lora,
    build_examples,
    latest_checkpoint,
    length_stats,
    load_checkpoint_adapter,
    load_state,
    matched_modules,
    prepare_for_training,
    train_adapter,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IDT LoRA 訓練：事件描述 → 個人／系統")
    parser.add_argument(
        "--train-features",
        type=Path,
        default=paths.TRAIN_FEATURES,
        help=f"訓練資料，預設 {paths.TRAIN_FEATURES.relative_to(paths.PROJECT_ROOT)}；"
        "也可直接用 01 的 train_data.json",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=paths.VAL_DATA,
        help=f"驗證資料，預設 {paths.VAL_DATA.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.IDT_ADAPTER_DIR,
        help=f"adapter 輸出目錄，預設 {paths.IDT_ADAPTER_DIR.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--limit", type=int, help="訓練與驗證各只取前 N 筆試跑，輸出改到 out-dir/trial_N/"
    )
    parser.add_argument(
        "--resume", action="store_true", help="從 out-dir/checkpoints/ 最後一個 epoch 續跑"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只載入 tokenizer 檢查資料與設定，不載入模型"
    )

    model = parser.add_argument_group("模型")
    model.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face 模型 ID 或本機目錄")
    model.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    model.add_argument("--dtype", choices=DTYPE_CHOICES, default="auto")
    model.add_argument("--local-files-only", action="store_true", help="只用已下載的權重")
    model.add_argument("--seed", type=int, default=42, help="LoRA 初始化與每個 epoch 資料順序")

    training = parser.add_argument_group("訓練")
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument(
        "--gradient-accumulation-steps", type=int, default=8, help="每筆單獨前向，累積 N 筆更新一次"
    )
    training.add_argument("--learning-rate", type=float, default=2e-4)
    training.add_argument("--max-length", type=int, default=4096, help="超過就報錯，不截斷")
    training.add_argument("--warmup-ratio", type=float, default=0.03)
    training.add_argument("--weight-decay", type=float, default=0.0)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument("--no-gradient-checkpointing", action="store_true")
    training.add_argument("--log-every", type=int, default=10, help="每 N 個優化步印一次")

    lora = parser.add_argument_group("LoRA")
    lora.add_argument("--lora-rank", type=int, default=8)
    lora.add_argument("--lora-alpha", type=int, default=16)
    lora.add_argument("--lora-dropout", type=float, default=0.05)
    lora.add_argument(
        "--target-modules",
        default=DEFAULT_TARGET_MODULES,
        help="掛 LoRA 的模組（正規表示式，整個模組名稱要匹配）；預設只掛文字模型的 q_proj／v_proj",
    )
    return parser


def describe(records: list[dict]) -> str:
    return "／".join(f"{label} {count}" for label, count in count_labels(records).items())


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # 導到檔案或 tee 時進度也能即時看到
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必須大於 0。")
    try:
        model_config = ModelConfig(
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
        config = TrainingConfig(
            epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            log_every=args.log_every,
        )
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = args.out_dir / f"trial_{args.limit}" if args.limit else args.out_dir
    checkpoint_dir = out_dir / CHECKPOINT_DIR
    for path, what in ((args.train_features, "訓練資料"), (args.val_data, "驗證資料")):
        if not path.is_file():
            sys.exit(f"找不到{what} {path}。")
    train_records = io.load_json(args.train_features)
    val_records = io.load_json(args.val_data)
    if args.limit:
        train_records, val_records = train_records[: args.limit], val_records[: args.limit]
        print(f"試跑模式：訓練與驗證各取前 {args.limit} 筆，輸出到 {out_dir}")
    if not train_records or not val_records:
        sys.exit("訓練或驗證資料是空的。")
    print(
        f"訓練 {len(train_records)} 筆（{describe(train_records)}）、"
        f"驗證 {len(val_records)} 筆（{describe(val_records)}）"
    )

    print(f"=== 載入 tokenizer：{model_config.model} ===")
    tokenizer = load_tokenizer(model_config.model, model_config.local_files_only)
    try:
        train_examples = build_examples(tokenizer, train_records, config.max_length, "訓練資料")
        val_examples = build_examples(tokenizer, val_records, config.max_length, "驗證資料")
    except ValueError as exc:
        sys.exit(f"中止：{exc}")
    train_stats, val_stats = length_stats(train_examples), length_stats(val_examples)
    print(
        f"token 長度：訓練 max {train_stats['max']}／p95 {train_stats['p95']}／"
        f"mean {train_stats['mean']}，驗證 max {val_stats['max']}；"
        f"標籤 token 數 {train_stats['label_tokens']}（max_length {config.max_length}）"
    )

    settings = {
        "train_features": str(args.train_features),
        "val_data": str(args.val_data),
        "out_dir": str(out_dir),
        "limit": args.limit,
        "resume": args.resume,
        "model": model_config.to_dict(),
        "seed": args.seed,
        "training": config.to_dict(),
        "token_lengths": {"train": train_stats, "val": val_stats},
    }
    if args.dry_run:
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        return

    state = checkpoint = None
    if args.resume:
        checkpoint = latest_checkpoint(checkpoint_dir)
        if checkpoint is None:
            sys.exit(f"--resume 找不到 checkpoint：{checkpoint_dir}")
        state = load_state(checkpoint)
        if state["config"] != config.to_dict():
            sys.exit(
                "中止：checkpoint 的訓練設定與本次不同，續跑必須用相同設定。\n"
                f"checkpoint：{state['config']}\n本次：{config.to_dict()}"
            )
        if state["epoch"] >= config.epochs:
            sys.exit(f"checkpoint 已完成 {state['epoch']} 個 epoch，沒有可續跑的 epoch。")
        print(f"從 {checkpoint} 續跑（已完成 {state['epoch']} 個 epoch）")

    import torch

    torch.manual_seed(args.seed)  # LoRA 初始化可重現
    print("=== 載入模型 ===")
    started = time.perf_counter()
    model, device, dtype = load_base_model(model_config)
    print(f"裝置 {device}、精度 {dtype}，{time.perf_counter() - started:.0f} 秒")
    model = prepare_for_training(model, config.gradient_checkpointing)
    peft_model = None
    try:
        names = matched_modules(model, config.target_modules)
        if checkpoint is not None:
            peft_model = load_checkpoint_adapter(model, checkpoint)
        else:
            peft_model, names = attach_lora(model, config)
        trainable, total = peft_model.get_nb_trainable_parameters()
        print(
            f"LoRA 掛在 {len(names)} 個模組（例如 {names[:2]}）；"
            f"可訓練參數 {trainable:,}／{total:,}（{trainable / total:.3%}）"
        )
        print(
            f"=== 5. IDT LoRA 訓練：{config.epochs} 個 epoch，"
            f"每 {config.gradient_accumulation_steps} 筆更新一次 ==="
        )
        history = train_adapter(
            model=peft_model,
            train_examples=train_examples,
            val_examples=val_examples,
            config=config,
            device=device,
            seed=args.seed,
            checkpoint_dir=checkpoint_dir,
            state=state,
        )
        peft_model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        metadata = {
            "base_model": model_config.model,
            "idt_prompt": IDT_PROMPT_ID,  # 04 載入 adapter 時會比對，提示改了就要重新訓練
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "device": device,
            "dtype": dtype,
            "data": {
                "train_features": str(args.train_features),
                "train_records": len(train_records),
                "train_label_counts": count_labels(train_records),
                "val_data": str(args.val_data),
                "val_records": len(val_records),
                "val_label_counts": count_labels(val_records),
                "limit": args.limit,
                "token_lengths": {"train": train_stats, "val": val_stats},
            },
            "seed": args.seed,
            "training": config.to_dict(),
            "lora": {
                "matched_modules": len(names),
                "examples": names[:4],
                "trainable_params": trainable,
                "total_params": total,
            },
            "history": history,
            "duration_minutes": round((time.perf_counter() - started) / 60, 1),
        }
        metadata_path = io.save_json(metadata, out_dir / "training_metadata.json")
    except ValueError as exc:
        sys.exit(f"中止：{exc}")
    finally:
        release(peft_model)
        release(model)
    last = history[-1]
    print(
        f"完成 {last['epoch']} 個 epoch：train_loss {last['train_loss']:.4f}、"
        f"val_loss {last['val_loss']:.4f}、val_label_match {last['val_label_match']:.3f}"
    )
    print(f"adapter → {out_dir}\n→ {metadata_path}")


if __name__ == "__main__":
    main()
