"""步驟 04 推論與評估，對應流程圖的 ⑥⑦⑧⑨。

對測試集：⑥ 載入本機基礎模型 + IDT adapter 預測 idt_pred（溫度 0，只接受「個人」或「系統」），
釋放後 ⑦ 以情緒模型（不套 adapter；預設與 02 ④ 同一個 utils/llm.py 的 DEFAULT_EMOTION_MODEL，
--model 或 --emotion-model 可覆寫）預測 emotion_pred（提示與訓練集標註相同），⑧ 比對人工
idt_target 算 accuracy 與混淆矩陣，⑨ 再用評審模型對 (description, emotion_pred) 評分。
模型透過 LangChain 的 chat model 介面呼叫：⑥ 一定是本機模型（要掛 adapter）；⑦ 與 ⑨ 可用
--emotion-provider／--judge-provider 切到 openai、ollama 等服務。本機模型一次只載入一個。
每筆預測與評分都先寫進進度檔與 inference_predictions.json，中斷或評審失敗都能續跑；
記成 None 的紀錄重跑時會再試一次。

在專案根目錄執行：
    python -m scripts.04_inference --dry-run
    python -m scripts.04_inference --limit 5                 # 試跑，輸出到 results/trial_5/
    python -m scripts.04_inference                           # ⑥⑦⑧⑨ 全部
    python -m scripts.04_inference --skip-judge              # 只做 ⑥⑦⑧，評審之後再補
    python -m scripts.04_inference --stage judge             # 只對既有預測做 ⑨
    python -m scripts.04_inference --stage evaluate          # 只重算 ⑧ 與評審彙整，不載模型
    python -m scripts.04_inference --judge-load-in-4bit      # 12 GB 顯卡跑 9B 評審模型
    python -m scripts.04_inference --judge-provider openai --judge-model gpt-4o   # 雲端評審
    python -m scripts.04_inference --stage judge --predictions data/features/train_features.json \\
        --emotion-field emotion_target --out-dir results/train_emotion_judge   # 評訓練集的模型標註

輸出（預設 results/）：
    inference_predictions.json   每筆的 idt_pred、emotion_pred、emotion_judge
    inference_evaluation.json    IDT accuracy／混淆矩陣／各類別 recall；情緒分布與評審平均分
    progress/{idt,emotion,judge}.jsonl   進度檔
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from utils import io, paths
from utils.cli import add_model_arguments, first_given, model_config_from_args
from utils.emotion import record_key
from utils.evaluation import idt_metrics, judge_summary, label_distribution
from utils.inference import (
    EMOTION_JUDGE,
    EMOTION_PREDICTION,
    IDT_PREDICTION,
    RetryConfig,
    judge_emotion,
    predict_emotion,
    predict_idt,
)
from utils.llm import (
    DEFAULT_EMOTION_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    DEVICE_CHOICES,
    DTYPE_CHOICES,
    ModelConfig,
    build_chat_model,
    chat_generate,
    close_chat_model,
    describe_chat_model,
    load_base_model,
    load_tokenizer,
)
from utils.preprocessing import LABEL_COLUMN, count_labels
from utils.progress import Progress

STAGES = ("all", "idt", "emotion", "judge", "evaluate")
PREDICTION_FIELDS = (IDT_PREDICTION, EMOTION_PREDICTION, EMOTION_JUDGE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IDT 推論、情緒推論、評估與情緒評審（測試集）")
    parser.add_argument(
        "--test-data",
        type=Path,
        default=paths.TEST_DATA,
        help=f"01 切出的測試集，預設 {paths.TEST_DATA.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="既有的預測檔；--stage judge／evaluate 預設讀 out-dir 的 inference_predictions.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.RESULTS_DIR,
        help=f"輸出目錄，預設 {paths.RESULTS_DIR.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="all ⑥⑦⑧⑨；idt 只 ⑥⑧；emotion 只 ⑦；judge 只 ⑨；evaluate 只重算 ⑧ 與彙整",
    )
    parser.add_argument(
        "--skip-judge", action="store_true", help="all 時略過 ⑨，之後用 --stage judge 補"
    )
    parser.add_argument("--limit", type=int, help="只處理前 N 筆試跑，輸出改到 out-dir/trial_N/")
    parser.add_argument("--restart", action="store_true", help="忽略既有進度檔重新開始")
    parser.add_argument(
        "--dry-run", action="store_true", help="只列出設定與檢查 adapter，不載入模型"
    )

    base = parser.add_argument_group("⑥ IDT 推論：本機基礎模型 + adapter（一定在本機）")
    base.add_argument(
        "--model",
        help="基礎模型（Hugging Face ID 或本機目錄），須與訓練 adapter 時相同；"
        f"預設 utils/llm.py 的 DEFAULT_MODEL（{DEFAULT_MODEL}）。有指定時 ⑦ 也沿用它",
    )
    base.add_argument(
        "--adapter",
        type=Path,
        default=paths.IDT_ADAPTER_DIR,
        help=f"03 輸出的 adapter 目錄，預設 {paths.IDT_ADAPTER_DIR.relative_to(paths.PROJECT_ROOT)}",
    )
    base.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="本機模型共用")
    base.add_argument("--dtype", choices=DTYPE_CHOICES, default="auto")
    base.add_argument("--local-files-only", action="store_true", help="本機模型只用已下載的權重")
    base.add_argument("--idt-max-new-tokens", type=int, default=16, help="⑥ 只需輸出兩個字")
    base.add_argument("--max-attempts", type=int, default=3, help="⑥⑦ 輸出無效的總嘗試次數")

    add_model_arguments(
        parser,
        title="⑦ 情緒推論（不套 adapter；預設與 02 ④ 用同一個模型）",
        prefix="emotion",
        default_max_new_tokens=256,
        local_options=(),
        model_help="Hugging Face ID、本機目錄，或該 provider 上的模型名；預設沿用 --model，"
        f"都沒指定就用 utils/llm.py 的 DEFAULT_EMOTION_MODEL（{DEFAULT_EMOTION_MODEL}）",
    )
    judge = add_model_arguments(
        parser,
        title="⑨ 情緒評審",
        prefix="judge",
        default_model=DEFAULT_JUDGE_MODEL,
        default_max_new_tokens=512,
        local_options=("dtype", "load_in_4bit"),
    )
    judge.add_argument("--judge-max-attempts", type=int, default=3)
    judge.add_argument(
        "--emotion-field",
        choices=(EMOTION_PREDICTION, "emotion_target"),
        default=EMOTION_PREDICTION,
        help="被評的欄位：測試集預測用 emotion_pred；評訓練集的模型標註用 emotion_target",
    )
    return parser


def model_configs_from_args(
    args: argparse.Namespace,
) -> tuple[ModelConfig, ModelConfig, ModelConfig]:
    """回傳 (⑥ IDT 基礎模型, ⑦ 情緒模型, ⑨ 評審模型) 的設定。

    ⑥ 用 --model，沒給就是 utils/llm.py 的 DEFAULT_MODEL；⑦ 依序取 --emotion-model、--model、
    DEFAULT_EMOTION_MODEL（與 02 ④ 相同）；device 與 local_files_only 三個模型共用。
    """
    idt = ModelConfig(
        model=first_given(args.model, DEFAULT_MODEL),
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.idt_max_new_tokens,
        local_files_only=args.local_files_only,
    )
    emotion = model_config_from_args(
        args,
        "emotion",
        model=first_given(args.emotion_model, args.model, DEFAULT_EMOTION_MODEL),
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    judge = model_config_from_args(
        args, "judge", device=args.device, local_files_only=args.local_files_only
    )
    return idt, emotion, judge


def check_adapter(adapter: Path, model_name: str) -> dict:
    """adapter 目錄要有 adapter_config.json；基礎模型必須與訓練時相同。"""
    if not (adapter / "adapter_config.json").is_file():
        sys.exit(
            f"找不到 adapter：{adapter}/adapter_config.json；請先執行 python -m scripts.03_train。"
        )
    metadata_path = adapter / "training_metadata.json"
    if metadata_path.is_file():
        trained_on = io.load_json(metadata_path).get("base_model")
    else:
        trained_on = io.load_json(adapter / "adapter_config.json").get("base_model_name_or_path")
    if trained_on and trained_on != model_name:
        sys.exit(
            f"中止：adapter 是用 {trained_on} 訓練的，本次 --model 是 {model_name}；基礎模型必須一致。"
        )
    return {"path": str(adapter), "base_model": trained_on}


def overlay_previous(records: list[dict], previous_path: Path) -> int:
    """把既有預測檔的 idt_pred／emotion_pred／emotion_judge 疊回同一筆，保留其他階段的結果。"""
    if not previous_path.is_file():
        return 0
    previous = {record_key(record): record for record in io.load_json(previous_path)}
    merged = 0
    for record in records:
        earlier = previous.get(record_key(record))
        if earlier is None:
            continue
        for field_name in PREDICTION_FIELDS:
            if field_name in earlier:
                record[field_name] = earlier[field_name]
                merged += 1
    return merged


def describe(records: list[dict]) -> str:
    return "／".join(f"{label} {count}" for label, count in count_labels(records).items())


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # 導到檔案或 tee 時進度也能即時看到
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必須大於 0。")
    try:
        idt_config, emotion_config, judge_config = model_configs_from_args(args)
        retry = RetryConfig(max_attempts=args.max_attempts)
        judge_retry = RetryConfig(max_attempts=args.judge_max_attempts)
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = args.out_dir / f"trial_{args.limit}" if args.limit else args.out_dir
    progress_dir = out_dir / paths.RESULTS_PROGRESS_DIR.name
    run_idt = args.stage in ("all", "idt")
    run_emotion = args.stage in ("all", "emotion")
    run_judge = (args.stage == "all" and not args.skip_judge) or args.stage == "judge"

    # 輸入：⑥⑦ 從測試集出發並疊回既有預測；judge／evaluate 直接讀預測檔。
    if args.stage in ("judge", "evaluate"):
        input_path = args.predictions or out_dir / paths.INFERENCE_PREDICTIONS.name
        if not input_path.is_file():
            sys.exit(f"找不到預測檔 {input_path}；請先執行 ⑥⑦，或用 --predictions 指定。")
        records = io.load_json(input_path)
        output_path = out_dir / input_path.name
        merged = 0
    else:
        input_path = args.test_data
        if not input_path.is_file():
            sys.exit(f"找不到測試集 {input_path}；請先執行 python -m scripts.01_prepare_data。")
        records = io.load_json(input_path)
        output_path = out_dir / paths.INFERENCE_PREDICTIONS.name
        merged = overlay_previous(records, output_path)
    if args.limit:
        records = records[: args.limit]
        print(f"試跑模式：只處理前 {args.limit} 筆，輸出到 {out_dir}")
    if not records:
        sys.exit(f"{input_path} 沒有任何紀錄。")
    has_targets = all(record.get(LABEL_COLUMN) for record in records)
    print(
        f"輸入 {len(records)} 筆（{describe(records) if has_targets else '無人工標籤'}）：{input_path}"
    )
    if merged:
        print(f"疊回既有預測 {merged} 個欄位：{output_path}")

    adapter_info = check_adapter(args.adapter, idt_config.model) if run_idt else None
    settings = {
        "stage": args.stage,
        "skip_judge": args.skip_judge,
        "input": str(input_path),
        "records_in": len(records),
        "limit": args.limit,
        "out_dir": str(out_dir),
        "adapter": adapter_info,
        "idt_model": idt_config.to_dict() if run_idt else None,
        "emotion_model": emotion_config.to_dict() if run_emotion else None,
        "retry": retry.to_dict(),
        "judge": {
            **judge_config.to_dict(),
            "retry": judge_retry.to_dict(),
            "emotion_field": args.emotion_field,
        }
        if run_judge
        else None,
    }
    if args.dry_run:
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        return

    reports: dict[str, dict] = {}
    total = len(records)

    def run_model_stage(stage: str, meta: dict, build, runner) -> None:
        """建立 chat model → 逐筆 → 存檔 → 釋放；本機模型一次只在記憶體裡一個。"""
        nonlocal records
        started = time.perf_counter()
        chat = build()
        print(f"模型：{describe_chat_model(chat)}，{time.perf_counter() - started:.0f} 秒")
        progress = Progress(progress_dir / f"{stage}.jsonl", meta=meta, restart=args.restart)
        print(f"進度檔已有 {len(progress)} 筆：{progress.path}")
        try:
            records, report = runner(chat_generate(chat), progress)
        finally:
            close_chat_model(chat)
        reports[stage] = report.to_dict()
        path = io.save_json(records, output_path)
        failed = f"，失敗 {len(report.failed)} 筆" if report.failed else ""
        counts = "、".join(f"{key} {value}" for key, value in report.counts.most_common())
        print(f"{stage}：{counts}{failed}（呼叫 {report.calls} 次）→ {path}")

    def on_record(position: int, record: dict, value, calls: int) -> None:
        shown = value if not isinstance(value, dict) else f"總分 {value['overall']}"
        detail = f"呼叫 {calls} 次" if calls else "沿用進度"
        print(f"[{position}/{total}] {record_key(record)} → {shown}（{detail}）")

    def build_idt_chat():
        """基礎模型掛上 adapter；tokenizer 優先用 adapter 目錄裡訓練時存的那份。"""
        from peft import PeftModel

        base, device, dtype = load_base_model(idt_config)
        model = PeftModel.from_pretrained(base, str(args.adapter))
        source = (
            args.adapter if (args.adapter / "tokenizer_config.json").is_file() else idt_config.model
        )
        tokenizer = load_tokenizer(str(source), idt_config.local_files_only)
        return build_chat_model(
            idt_config,
            temperature=0.0,
            hf_model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )

    if run_idt:
        print(f"=== 6. IDT 推論（{idt_config.model} + adapter）：{total} 筆 ===")
        run_model_stage(
            "idt",
            {
                **idt_config.to_meta(),
                "stage": "idt",
                "adapter": str(args.adapter),
                **retry.to_dict(),
            },
            build_idt_chat,
            lambda generate, progress: predict_idt(
                records, generate=generate, config=retry, progress=progress, on_record=on_record
            ),
        )
    if run_emotion:
        print(
            f"=== 7. 情緒推論（{emotion_config.provider}:{emotion_config.model}，不套 adapter）：{total} 筆 ==="
        )
        run_model_stage(
            "emotion",
            {**emotion_config.to_meta(), "stage": "emotion", **retry.to_dict()},
            lambda: build_chat_model(emotion_config, temperature=0.0),
            lambda generate, progress: predict_emotion(
                records, generate=generate, config=retry, progress=progress, on_record=on_record
            ),
        )
    if run_judge:
        print(
            f"=== 9. 情緒評審（{judge_config.provider}:{judge_config.model}，評 {args.emotion_field}）：{total} 筆 ==="
        )
        run_model_stage(
            "judge",
            {
                **judge_config.to_meta(),
                "stage": "judge",
                "emotion_field": args.emotion_field,
                **judge_retry.to_dict(),
            },
            lambda: build_chat_model(judge_config, temperature=0.0),
            lambda generate, progress: judge_emotion(
                records,
                generate=generate,
                config=judge_retry,
                progress=progress,
                emotion_field=args.emotion_field,
                model_name=f"{judge_config.provider}:{judge_config.model}",
                on_record=on_record,
            ),
        )

    # ⑧ 評估：IDT 有人工標籤就算指標；情緒只能看分布與評審分數。
    print("=== 8. 評估 ===")
    evaluation: dict = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "predictions": str(output_path),
        "records": total,
        "idt": None,
        "emotion": {
            "field": args.emotion_field,
            "distribution": label_distribution(records, args.emotion_field),
            "judge": None,
        },
        "settings": settings,
        "reports": reports,
    }
    if has_targets and any(IDT_PREDICTION in record for record in records):
        evaluation["idt"] = idt_metrics(records)
        metrics = evaluation["idt"]
        labels = metrics["confusion_matrix"]["labels"]
        print(
            f"IDT accuracy {metrics['accuracy']:.4f}（{metrics['correct']}/{metrics['records']}，"
            f"無效 {metrics['invalid_predictions']}）、macro F1 {metrics['macro_f1']:.4f}"
        )
        print(f"混淆矩陣（列 = 人工，欄 = 預測 {labels}）")
        for label, row in zip(labels, metrics["confusion_matrix"]["matrix"], strict=True):
            print(f"  {label}: {row}")
        for label, item in metrics["per_label"].items():
            print(
                f"  {label}: recall {item['recall']:.4f}、precision {item['precision']:.4f}、"
                f"F1 {item['f1']:.4f}"
            )
    else:
        print("沒有 idt_pred 或人工標籤，略過 IDT 指標。")
    distribution = evaluation["emotion"]["distribution"]
    print(
        f"情緒分布（{args.emotion_field}）：{distribution['counts']}，無效 {distribution['invalid']}"
    )
    if any(EMOTION_JUDGE in record for record in records):
        evaluation["emotion"]["judge"] = judge_summary(records)
        summary = evaluation["emotion"]["judge"]
        print(
            f"評審模型 {summary['models']}；平均分 {summary['mean_scores']}，"
            f"總分平均 {summary['mean_overall_score']}（{summary['scored']} 筆有分數，"
            f"{summary['unscored']} 筆沒有；這是品質分數，不是 accuracy）"
        )
    evaluation_path = io.save_json(evaluation, out_dir / paths.INFERENCE_EVALUATION.name)
    print(f"→ {evaluation_path}")


if __name__ == "__main__":
    main()
