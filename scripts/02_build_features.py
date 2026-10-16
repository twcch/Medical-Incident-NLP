"""步驟 02 特徵工程：訓練集增生 → 撰寫者情緒標註，對應流程圖的 ③④。

只對訓練集進行；驗證與測試集不增生、不標註。兩個階段用同一個模型，透過 LangChain 的
chat model 介面呼叫：--provider 預設 huggingface（本機 transformers，載入一次、用完釋放），
也可切到 openai、ollama 等服務。每筆結果都寫進進度檔，中斷後重跑同樣的指令即可續跑；
換 provider、模型或參數時進度檔會拒絕混用，需加 --restart。

在專案根目錄執行：
    python -m scripts.02_build_features
    python -m scripts.02_build_features --limit 5          # 先用 5 筆試跑，輸出到 trial_5/
    python -m scripts.02_build_features --stage emotion    # 只重跑情緒標註（讀 train_augmented.json）
    python -m scripts.02_build_features --dry-run          # 只列出設定，不載入模型
    python -m scripts.02_build_features --provider openai --model gpt-4o-mini   # 雲端，需 OPENAI_API_KEY
    python -m scripts.02_build_features --provider ollama --model qwen3.5:9b    # 需 pip install langchain-ollama

輸出（預設 data/features/）：
    train_augmented.json                               原文 + 改寫（③）
    train_features.json                                再加上 emotion_target（④），供 03 訓練使用
    progress/augment.jsonl、progress/emotion.jsonl     進度檔
    features_summary.json                              本次設定、呼叫次數、缺額與情緒分布
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from utils import io, paths
from utils.augmentation import ON_FAILURE_CHOICES, AugmentConfig, augment_records
from utils.cli import add_model_arguments, model_config_from_args
from utils.emotion import EmotionConfig, EmotionLabelError, label_records
from utils.llm import (
    DEFAULT_MODEL,
    build_chat_model,
    chat_generate,
    close_chat_model,
    describe_chat_model,
)
from utils.progress import Progress

STAGES = ("all", "augment", "emotion")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="特徵工程：訓練集增生（③）與撰寫者情緒標註（④）；只處理訓練集"
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=paths.TRAIN_DATA,
        help=f"01 產生的訓練集，預設 {paths.TRAIN_DATA.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.FEATURES_DIR,
        help=f"輸出目錄，預設 {paths.FEATURES_DIR.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="all 增生後標註；augment 只增生；emotion 只標註（讀 out-dir 的 train_augmented.json）",
    )
    parser.add_argument(
        "--limit", type=int, help="只處理前 N 筆做試跑，輸出改到 out-dir/trial_N/，不會動到正式結果"
    )
    parser.add_argument(
        "--restart", action="store_true", help="忽略既有進度檔重新開始（換模型或參數時需要）"
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出設定與階段，不載入模型")

    model = add_model_arguments(
        parser,
        title="模型（③④ 共用，不套 adapter）",
        default_model=DEFAULT_MODEL,
        default_max_new_tokens=1024,
    )
    model.add_argument("--seed", type=int, default=42, help="增生抽樣的基礎 seed（本機模型）")

    augment = parser.add_argument_group("③ 訓練集增生")
    augment.add_argument("--augment-n", type=int, default=2, help="每筆的改寫數，0 表示不增生")
    augment.add_argument("--augment-max-attempts", type=int, default=4, help="每個改寫最多嘗試次數")
    augment.add_argument("--augment-temperature", type=float, default=0.8)
    augment.add_argument(
        "--augment-on-failure",
        choices=ON_FAILURE_CHOICES,
        default="keep-partial",
        help="湊不滿時：keep-partial 保留已取得的版本並記錄缺額（預設）；abort 立即中止",
    )

    emotion = parser.add_argument_group("④ 撰寫者情緒標註（溫度 0）")
    emotion.add_argument(
        "--emotion-max-attempts", type=int, default=3, help="輸出無效的總嘗試次數，用完即中止"
    )
    return parser


def load_input(args: argparse.Namespace, out_dir: Path) -> tuple[Path, list[dict]]:
    """all／augment 讀 01 的訓練集；emotion 讀本目錄已增生的結果。"""
    if args.stage == "emotion":
        path = out_dir / paths.TRAIN_AUGMENTED.name
        if not path.is_file():
            sys.exit(f"找不到 {path}；--stage emotion 需要先執行過增生。")
    else:
        path = args.train_data
        if not path.is_file():
            sys.exit(f"找不到訓練集 {path}；請先執行 python -m scripts.01_prepare_data。")
    records = io.load_json(path)
    if args.limit:
        records = records[: args.limit]
    if not records:
        sys.exit(f"{path} 沒有任何紀錄。")
    return path, records


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # 導到檔案或 tee 時進度也能即時看到
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必須大於 0。")
    try:
        model_config = model_config_from_args(args)
        augment_config = AugmentConfig(
            n=args.augment_n,
            max_attempts=args.augment_max_attempts,
            temperature=args.augment_temperature,
            on_failure=args.augment_on_failure,
        )
        emotion_config = EmotionConfig(max_attempts=args.emotion_max_attempts)
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = args.out_dir / f"trial_{args.limit}" if args.limit else args.out_dir
    input_path, records = load_input(args, out_dir)
    run_augment = args.stage in ("all", "augment")
    run_emotion = args.stage in ("all", "emotion")
    settings = {
        "stage": args.stage,
        "input": str(input_path),
        "records_in": len(records),
        "limit": args.limit,
        "out_dir": str(out_dir),
        "model": model_config.to_dict(),
        "seed": args.seed,
        "augment": augment_config.to_dict() if run_augment else None,
        "emotion": emotion_config.to_dict() if run_emotion else None,
    }
    if args.dry_run:
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        return
    if args.limit:
        print(f"試跑模式：只處理前 {args.limit} 筆，輸出到 {out_dir}")

    # 增生 0 個改寫時不需要模型；其他情況建立一次，兩階段共用（本機模型只載入一次）。
    needs_model = run_emotion or augment_config.n > 0
    augment_chat = emotion_chat = None
    if needs_model:
        print(f"=== 載入模型 {model_config.provider}:{model_config.model} ===")
        started = time.perf_counter()
        first_temperature = (
            augment_config.temperature if run_augment else emotion_config.temperature
        )
        first = build_chat_model(model_config, temperature=first_temperature)
        print(f"{describe_chat_model(first)}，{time.perf_counter() - started:.0f} 秒")
        if run_augment:
            augment_chat = first
            if run_emotion:
                emotion_chat = build_chat_model(
                    model_config, temperature=emotion_config.temperature, share=first
                )
        else:
            emotion_chat = first
    progress_dir = out_dir / paths.FEATURES_PROGRESS_DIR.name
    summary: dict = {"settings": settings}
    total = len(records)

    try:
        if run_augment:
            print(f"=== 3. 訓練集增生：{total} 筆，每筆 {augment_config.n} 個改寫 ===")
            progress = Progress(
                progress_dir / "augment.jsonl",
                meta={**model_config.to_meta(), **augment_config.to_meta(), "seed": args.seed},
                restart=args.restart,
            )
            print(f"進度檔已有 {len(progress)} 筆：{progress.path}")

            def on_augmented(position: int, record: dict, rewrites: list[str], calls: int) -> None:
                detail = f"呼叫 {calls} 次" if calls else "沿用進度"
                print(
                    f"[{position}/{total}] {record['source_id']} "
                    f"改寫 {len(rewrites)}/{augment_config.n}（{detail}）"
                )

            records, augment_report = augment_records(
                records,
                generate=chat_generate(augment_chat) if augment_chat else None,
                config=augment_config,
                progress=progress,
                seed=args.seed,
                on_record=on_augmented,
            )
            augmented_path = io.save_json(records, out_dir / paths.TRAIN_AUGMENTED.name)
            summary["augment"] = augment_report.to_dict()
            print(
                f"原文 {augment_report.records} + 改寫 {augment_report.rewrites_obtained} "
                f"= {len(records)} 筆；缺額 {len(augment_report.shortfalls)} 筆 → {augmented_path}"
            )
            total = len(records)

        if run_emotion:
            print(f"=== 4. 撰寫者情緒標註：{total} 筆 ===")
            progress = Progress(
                progress_dir / "emotion.jsonl",
                meta={**model_config.to_meta(), **emotion_config.to_meta()},
                restart=args.restart,
            )
            print(f"進度檔已有 {len(progress)} 筆：{progress.path}")

            def on_labeled(position: int, record: dict, label: str, calls: int) -> None:
                detail = f"呼叫 {calls} 次" if calls else "沿用進度"
                key = record.get("record_id") or record["source_id"]
                print(f"[{position}/{total}] {key} → {label}（{detail}）")

            records, emotion_report = label_records(
                records,
                generate=chat_generate(emotion_chat),
                config=emotion_config,
                progress=progress,
                on_record=on_labeled,
            )
            features_path = io.save_json(records, out_dir / paths.TRAIN_FEATURES.name)
            summary["emotion"] = emotion_report.to_dict()
            counts = "、".join(f"{k} {v}" for k, v in emotion_report.label_counts.most_common())
            print(f"情緒分布：{counts} → {features_path}")
    except (RuntimeError, EmotionLabelError, ValueError) as exc:
        # 進度檔已保存到中止的那一筆；修正後重跑同一指令即可續跑。
        sys.exit(f"中止：{exc}")
    finally:
        close_chat_model(augment_chat)
        if emotion_chat is not augment_chat:
            close_chat_model(emotion_chat)

    summary_path = io.save_json(summary, out_dir / paths.FEATURES_SUMMARY.name)
    print(f"→ {summary_path}")


if __name__ == "__main__":
    main()
