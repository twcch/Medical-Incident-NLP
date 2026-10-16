"""步驟 02 特徵工程：訓練集增生 → 撰寫者情緒標註，對應流程圖的 ③④。

只對訓練集進行；驗證與測試集不增生、不標註。兩個階段都透過 LangChain 的 chat model 介面
呼叫模型，而且可以分開設定：預設模型是 utils/llm.py 的 DEFAULT_AUGMENT_MODEL（③）與
DEFAULT_EMOTION_MODEL（④），改那兩行就能固定用不同模型；執行時 --provider／--model 等是
③④ 共用的覆寫（--model 同時蓋掉兩階段的預設），各階段再可用 --augment-*／--emotion-* 覆寫，
沒指定的選項沿用共用設定。
同一份本機權重只載入一次、兩階段共用；③④ 是不同本機模型時依序載入、用完釋放，一次只在
記憶體裡一個。每筆結果都寫進進度檔，中斷後重跑同樣的指令即可續跑；換 provider、模型或
參數時進度檔會拒絕混用，需加 --restart。

在專案根目錄執行：
    python -m scripts.02_build_features
    python -m scripts.02_build_features --limit 5          # 先用 5 筆試跑，輸出到 trial_5/
    python -m scripts.02_build_features --stage emotion    # 只重跑情緒標註（讀 train_augmented.json）
    python -m scripts.02_build_features --dry-run          # 只列出設定，不載入模型
    python -m scripts.02_build_features --provider openai --model gpt-4o-mini   # ③④ 都用雲端，需 OPENAI_API_KEY
    python -m scripts.02_build_features --provider ollama --model qwen3.5:9b    # 需 pip install langchain-ollama
    python -m scripts.02_build_features --augment-model Qwen/Qwen3.5-9B --emotion-model Qwen/Qwen3.5-4B
        # ③④ 用不同本機模型，依序載入
    python -m scripts.02_build_features --augment-provider openai --augment-model gpt-4o-mini
        # 只有 ③ 走雲端，④ 用共用設定的本機模型

輸出（預設 data/features/）：
    train_augmented.json                               原文 + 改寫（③）
    train_features.json                                再加上 emotion_target（④），供 03 訓練使用
    progress/augment.jsonl、progress/emotion.jsonl     進度檔
    features_summary.json                              本次設定（含兩階段的模型）、呼叫次數、缺額與情緒分布
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from utils import io, paths
from utils.augmentation import ON_FAILURE_CHOICES, AugmentConfig, augment_records
from utils.cli import (
    LOCAL_OPTIONS,
    PER_MODEL_LOCAL_OPTIONS,
    add_model_arguments,
    first_given,
    model_config_from_args,
)
from utils.emotion import EmotionConfig, EmotionLabelError, label_records
from utils.llm import (
    DEFAULT_AUGMENT_MODEL,
    DEFAULT_EMOTION_MODEL,
    ModelConfig,
    build_chat_model,
    chat_generate,
    close_chat_model,
    describe_chat_model,
)
from utils.progress import Progress

STAGES = ("all", "augment", "emotion")
MODEL_HELP = "Hugging Face ID、本機目錄，或該 provider 上的模型名"


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

    # 不帶前綴的是 ③④ 共用的預設；下面兩個群組只覆寫有指定的選項。
    add_model_arguments(
        parser,
        title="模型（③④ 共用的預設，不套 adapter；各階段可用 --augment-*／--emotion-* 覆寫）",
        default_max_new_tokens=1024,
        local_options=LOCAL_OPTIONS,
        model_help=f"{MODEL_HELP}；③④ 都用它，沒指定就各用 utils/llm.py 的 "
        f"DEFAULT_AUGMENT_MODEL（{DEFAULT_AUGMENT_MODEL}）與 DEFAULT_EMOTION_MODEL（{DEFAULT_EMOTION_MODEL}）",
    )

    augment = add_model_arguments(
        parser,
        title="③ 訓練集增生（模型選項沒指定就沿用共用設定）",
        prefix="augment",
        inherit=True,
        local_options=PER_MODEL_LOCAL_OPTIONS,
        model_help=f"{MODEL_HELP}；預設沿用 --model，都沒指定就用 utils/llm.py 的 "
        f"DEFAULT_AUGMENT_MODEL（{DEFAULT_AUGMENT_MODEL}）",
    )
    augment.add_argument("--seed", type=int, default=42, help="增生抽樣的基礎 seed（本機模型）")
    augment.add_argument("--augment-n", type=int, default=2, help="每筆的改寫數，0 表示不增生")
    augment.add_argument("--augment-max-attempts", type=int, default=4, help="每個改寫最多嘗試次數")
    augment.add_argument("--augment-temperature", type=float, default=0.8)
    augment.add_argument(
        "--augment-on-failure",
        choices=ON_FAILURE_CHOICES,
        default="keep-partial",
        help="湊不滿時：keep-partial 保留已取得的版本並記錄缺額（預設）；abort 立即中止",
    )

    emotion = add_model_arguments(
        parser,
        title="④ 撰寫者情緒標註（溫度 0；模型選項沒指定就沿用共用設定）",
        prefix="emotion",
        inherit=True,
        local_options=PER_MODEL_LOCAL_OPTIONS,
        model_help=f"{MODEL_HELP}；預設沿用 --model，都沒指定就用 utils/llm.py 的 "
        f"DEFAULT_EMOTION_MODEL（{DEFAULT_EMOTION_MODEL}）",
    )
    emotion.add_argument(
        "--emotion-max-attempts", type=int, default=3, help="輸出無效的總嘗試次數，用完即中止"
    )
    return parser


def model_configs_from_args(args: argparse.Namespace) -> tuple[ModelConfig, ModelConfig]:
    """回傳 (③ 增生, ④ 情緒標註) 的模型設定。

    模型名稱依序取 --augment-model／--emotion-model、--model、utils/llm.py 的
    DEFAULT_AUGMENT_MODEL／DEFAULT_EMOTION_MODEL；其他選項是 --augment-*／--emotion-* 蓋在共用設定上。
    """
    shared = model_config_from_args(args)
    augment = model_config_from_args(
        args,
        "augment",
        base=shared,
        model=first_given(args.augment_model, args.model, DEFAULT_AUGMENT_MODEL),
    )
    emotion = model_config_from_args(
        args,
        "emotion",
        base=shared,
        model=first_given(args.emotion_model, args.model, DEFAULT_EMOTION_MODEL),
    )
    return augment, emotion


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


def load_chat_model(stage: str, config: ModelConfig, temperature: float) -> BaseChatModel:
    """建立某個階段的 chat model（本機模型在這裡載入權重）並印出耗時。"""
    print(f"=== 載入{stage}模型 {config.provider}:{config.model} ===")
    started = time.perf_counter()
    chat = build_chat_model(config, temperature=temperature)
    print(f"{describe_chat_model(chat)}，{time.perf_counter() - started:.0f} 秒")
    return chat


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # 導到檔案或 tee 時進度也能即時看到
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必須大於 0。")
    try:
        augment_model, emotion_model = model_configs_from_args(args)
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
        "augment": {
            "model": augment_model.to_dict(),
            "seed": args.seed,
            **augment_config.to_dict(),
        }
        if run_augment
        else None,
        "emotion": {"model": emotion_model.to_dict(), **emotion_config.to_dict()}
        if run_emotion
        else None,
    }
    if args.dry_run:
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        return
    if args.limit:
        print(f"試跑模式：只處理前 {args.limit} 筆，輸出到 {out_dir}")

    progress_dir = out_dir / paths.FEATURES_PROGRESS_DIR.name
    summary: dict = {"settings": settings}
    total = len(records)
    augment_chat = emotion_chat = None

    try:
        if run_augment:
            if augment_config.n > 0:  # 增生 0 個改寫時不需要模型
                augment_chat = load_chat_model("增生", augment_model, augment_config.temperature)
            print(f"=== 3. 訓練集增生：{total} 筆，每筆 {augment_config.n} 個改寫 ===")
            progress = Progress(
                progress_dir / "augment.jsonl",
                meta={**augment_model.to_meta(), **augment_config.to_meta(), "seed": args.seed},
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
            if augment_chat is not None and augment_model.same_weights(emotion_model):
                # 同一份本機權重只載入一次：④ 以自己的溫度與 token 上限共用 ③ 的模型。
                emotion_chat = build_chat_model(
                    emotion_model, temperature=emotion_config.temperature, share=augment_chat
                )
                print(f"=== 情緒標註沿用已載入的增生模型 {emotion_model.model} ===")
            else:
                # 本機模型一次只在記憶體裡一個：先釋放 ③ 的模型，再載入 ④ 的。
                close_chat_model(augment_chat)
                augment_chat = None
                emotion_chat = load_chat_model(
                    "情緒標註", emotion_model, emotion_config.temperature
                )
            print(f"=== 4. 撰寫者情緒標註：{total} 筆 ===")
            progress = Progress(
                progress_dir / "emotion.jsonl",
                meta={**emotion_model.to_meta(), **emotion_config.to_meta()},
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
        # 共用權重時兩個 chat model 都指著同一份模型，兩個都關才會真正釋放。
        close_chat_model(emotion_chat)
        close_chat_model(augment_chat)

    summary_path = io.save_json(summary, out_dir / paths.FEATURES_SUMMARY.name)
    print(f"→ {summary_path}")


if __name__ == "__main__":
    main()
