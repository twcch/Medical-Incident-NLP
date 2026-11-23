"""步驟 05 IDT 機率評分與 checkpoint 選擇（分析用），對應流程圖的 ⑩⑪。

04 的 ⑥ 用貪婪解碼，等於固定在 P(系統) > 0.5 才判「系統」，無法比較 checkpoint，也不能調門檻。
這一步不生成文字，直接讀回答第一個 token 的 logit 算 P(系統)：

    score     載入基礎模型一次，依序掛上 03 存下的每個 epoch 的 adapter，再加上不掛 adapter 的
              zero-shot，對驗證集與測試集逐筆評分（要 GPU；逐筆寫進度檔，可中斷續跑）
    evaluate  只讀分數檔（不載模型）：先把只差浮點誤差的分數合併成同分，再算各設定的
              log-loss／AUROC／AP；以驗證集 log-loss 挑 checkpoint；
              在驗證集找 macro-F1 最高的門檻；比較測試集在 0.5 與該門檻的指標與 bootstrap 信賴區間；
              加上 TF-IDF＋邏輯迴歸基準（需要 scikit-learn）

在專案根目錄執行：
    python -m scripts.05_score_idt --dry-run              # 列出要評的設定、檢查 adapter，不載模型
    python -m scripts.05_score_idt --limit 20             # 試跑，輸出到 results/idt_scores/trial_20/
    python -m scripts.05_score_idt                        # score + evaluate
    python -m scripts.05_score_idt --stage evaluate       # 只重算評估，不載模型
    python -m scripts.05_score_idt --adapters models/idt_adapter/checkpoints/epoch-1 --no-zero-shot

輸出（預設 results/idt_scores/）：
    scores_val.json、scores_test.json      每筆在各設定下的 P(系統)
    idt_score_evaluation.json              各設定的指標、選出的 checkpoint、門檻、信賴區間與基準比較
    progress/score_<設定>_<val|test>.jsonl  進度檔
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from utils import io, paths
from utils.baseline import TFIDF_SETTINGS, tfidf_scores
from utils.emotion import record_key
from utils.evaluation import summarize_scores
from utils.llm import (
    DEFAULT_MODEL,
    DEVICE_CHOICES,
    DTYPE_CHOICES,
    ModelConfig,
    load_base_model,
    load_tokenizer,
    release,
)
from utils.preprocessing import LABEL_COLUMN, TEXT_COLUMN, VALID_IDT_LABELS, count_labels
from utils.progress import Progress
from utils.prompts import IDT_PROMPT_ID
from utils.scoring import (
    SCORE_METHOD,
    ZERO_SHOT,
    adapter_metadata,
    check_adapter_compatible,
    discover_adapters,
    label_token_ids,
    score_record,
    score_records,
)
from utils.training import supports_logits_to_keep

STAGES = ("all", "score", "evaluate")
BASELINE = "tfidf"
REPORT_EVERY = 50  # 評分時每幾筆印一次進度


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IDT 機率評分：比較各 checkpoint、zero-shot 與 TF-IDF，挑 checkpoint、調門檻"
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=paths.VAL_DATA,
        help=f"驗證集（挑 checkpoint 與調門檻），預設 {paths.VAL_DATA.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=paths.TEST_DATA,
        help=f"測試集，預設 {paths.TEST_DATA.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.IDT_SCORES_DIR,
        help=f"輸出目錄，預設 {paths.IDT_SCORES_DIR.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="all：score + evaluate；score 只評分（要 GPU）；evaluate 只讀分數檔重算（不載模型）",
    )
    parser.add_argument(
        "--limit", type=int, help="驗證集與測試集各只取前 N 筆試跑，輸出改到 out-dir/trial_N/"
    )
    parser.add_argument("--restart", action="store_true", help="忽略既有進度檔重新評分")
    parser.add_argument(
        "--dry-run", action="store_true", help="只列出要評的設定與檢查 adapter，不載入模型"
    )

    model = parser.add_argument_group("評分模型：本機基礎模型 + 03 的 adapter")
    model.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"基礎模型，須與訓練 adapter 時相同；預設 utils/llm.py 的 DEFAULT_MODEL（{DEFAULT_MODEL}）",
    )
    model.add_argument(
        "--adapter-dir",
        type=Path,
        default=paths.IDT_ADAPTER_DIR,
        help="03 的輸出目錄：評它底下的 checkpoints/epoch-N；最上層的 adapter 與某個 epoch 相同就"
        f"不重複評。預設 {paths.IDT_ADAPTER_DIR.relative_to(paths.PROJECT_ROOT)}",
    )
    model.add_argument(
        "--adapters",
        type=Path,
        nargs="+",
        help="改評指定的 adapter 目錄（設定名稱取目錄名），不自動搜尋 --adapter-dir",
    )
    model.add_argument("--no-zero-shot", action="store_true", help="不評不掛 adapter 的基礎模型")
    model.add_argument("--zero-shot-only", action="store_true", help="只評不掛 adapter 的基礎模型")
    model.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    model.add_argument("--dtype", choices=DTYPE_CHOICES, default="auto")
    model.add_argument("--local-files-only", action="store_true", help="只用已下載的權重")

    evaluation = parser.add_argument_group("評估")
    evaluation.add_argument(
        "--train-data",
        type=Path,
        default=paths.TRAIN_DATA,
        help="TF-IDF 基準的訓練資料，預設 01 的原始訓練集（不含增生）",
    )
    evaluation.add_argument(
        "--no-baseline", action="store_true", help="不算 TF-IDF 基準（不需要 scikit-learn）"
    )
    evaluation.add_argument(
        "--bootstrap", type=int, default=2000, help="測試集 bootstrap 次數，0 表示不算信賴區間"
    )
    evaluation.add_argument("--seed", type=int, default=42, help="bootstrap 的亂數種子")
    evaluation.add_argument(
        "--reference",
        type=Path,
        default=paths.INFERENCE_PREDICTIONS,
        help="04 的預測檔，用來核對 0.5 門檻與貪婪解碼的一致率；檔案不存在就略過",
    )
    return parser


def describe(records: list[dict]) -> str:
    return "／".join(f"{label} {count}" for label, count in count_labels(records).items())


def load_datasets(args: argparse.Namespace) -> dict[str, list[dict]]:
    """讀驗證集與測試集（--limit 時各取前 N 筆）；每筆都要有人工 IDT 標籤。"""
    datasets = {}
    for split, path, name in (("val", args.val_data, "驗證集"), ("test", args.test_data, "測試集")):
        if not path.is_file():
            sys.exit(f"找不到{name} {path}；請先執行 python -m scripts.01_prepare_data。")
        records = io.load_json(path)
        if args.limit:
            records = records[: args.limit]
        if not records:
            sys.exit(f"{path} 沒有任何紀錄。")
        if any(record.get(LABEL_COLUMN) not in VALID_IDT_LABELS for record in records):
            sys.exit(f"{path} 有紀錄缺少人工 IDT 標籤，無法評估。")
        datasets[split] = records
    return datasets


def resolve_configs(args: argparse.Namespace) -> list[dict]:
    """要評的設定：zero-shot（不掛 adapter）與各個 adapter；基礎模型或 IDT 提示不符就中止。"""
    if args.zero_shot_only:
        adapters = []
    elif args.adapters:
        adapters = [(path.name, path) for path in args.adapters]
    else:
        adapters = discover_adapters(args.adapter_dir)
        if not adapters:
            sys.exit(
                f"{args.adapter_dir} 底下找不到 adapter；請先執行 python -m scripts.03_train，"
                "或用 --adapters 指定、--zero-shot-only 只評基礎模型。"
            )
    names = [name for name, _ in adapters]
    reserved = {ZERO_SHOT, BASELINE}
    if len(set(names)) < len(names) or reserved & set(names):
        sys.exit(f"adapter 名稱（取目錄名）重複，或與保留名稱 {sorted(reserved)} 相同：{names}")
    configs = [] if args.no_zero_shot else [{"name": ZERO_SHOT, "adapter": None}]
    for name, path in adapters:
        try:
            info = adapter_metadata(path)
            check_adapter_compatible(info, args.model)
        except ValueError as exc:
            sys.exit(f"中止：{exc}")
        if not info["idt_prompt"]:
            print(
                f"注意：{path} 找不到訓練時的 training_metadata.json，無法確認 IDT 提示是否相同。"
            )
        configs.append({"name": name, "adapter": info})
    return configs


def scores_path(out_dir: Path, split: str) -> Path:
    return out_dir / f"scores_{split}.json"


def save_scores(
    out_dir: Path,
    split: str,
    records: list[dict],
    values: dict[str, list[dict]],
    configs: list[dict],
    label_ids: dict[str, int],
) -> Path:
    """一筆一列整理各設定的分數；values 是 {設定: 與 records 同序的評分結果}。"""
    rows = [
        {
            "key": record_key(record),
            LABEL_COLUMN: record[LABEL_COLUMN],
            "p_system": {name: items[index]["p_system"] for name, items in values.items()},
            "label_mass": {name: items[index]["label_mass"] for name, items in values.items()},
            "top_is_label": {name: items[index]["top_is_label"] for name, items in values.items()},
        }
        for index, record in enumerate(records)
    ]
    return io.save_json(
        {
            "split": split,
            "method": SCORE_METHOD,
            "idt_prompt": IDT_PROMPT_ID,
            "label_tokens": label_ids,
            "configs": [config for config in configs if config["name"] in values],
            "records": rows,
        },
        scores_path(out_dir, split),
    )


def score_meta(model_config: ModelConfig, config: dict, split: str, label_ids: dict) -> dict:
    """進度檔第一行：模型、精度、提示、評分方式、adapter 權重或標籤 token 不同就不能續跑。"""
    adapter = config["adapter"]
    return {
        "stage": "idt_score",
        "method": SCORE_METHOD,
        "model": model_config.model,
        "dtype": model_config.dtype,
        "prompt": IDT_PROMPT_ID,
        "config": config["name"],
        "adapter_sha256": adapter["sha256"] if adapter else None,
        "split": split,
        "label_tokens": label_ids,
    }


def run_scoring(
    args: argparse.Namespace,
    model_config: ModelConfig,
    configs: list[dict],
    datasets: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    """基礎模型只載入一次，所有 adapter 一起掛上，再逐一切換；zero-shot 時暫時停用 adapter。"""
    adapter_dir = args.adapter_dir
    source = (
        adapter_dir if (adapter_dir / "tokenizer_config.json").is_file() else model_config.model
    )
    tokenizer = load_tokenizer(str(source), model_config.local_files_only)
    try:
        label_ids = label_token_ids(tokenizer, datasets["val"][0][TEXT_COLUMN])
    except ValueError as exc:
        sys.exit(f"中止：{exc}")
    print(f"標籤 token：{label_ids}（tokenizer：{source}）")
    # 先開好所有進度檔：設定與既有進度不同就在載入模型之前中止，不必白等模型載入
    progress: dict[tuple[str, str], Progress] = {}
    for config in configs:
        for split in datasets:
            try:
                progress[config["name"], split] = Progress(
                    out_dir / "progress" / f"score_{config['name']}_{split}.jsonl",
                    meta=score_meta(model_config, config, split, label_ids),
                    restart=args.restart,
                )
            except ValueError as exc:
                sys.exit(f"中止：{exc}")
    done = sum(len(item) for item in progress.values())
    if done:
        print(f"進度檔已有 {done} 筆分數，會直接沿用")

    print(f"=== 載入模型：{model_config.model} ===")
    started = time.perf_counter()
    base, device, dtype = load_base_model(model_config)
    model = base
    adapters = [config for config in configs if config["adapter"]]
    try:
        if adapters:
            from peft import PeftModel

            first, *others = adapters
            model = PeftModel.from_pretrained(
                base, first["adapter"]["path"], adapter_name=first["name"]
            )
            for config in others:
                model.load_adapter(config["adapter"]["path"], adapter_name=config["name"])
        model.eval()
        names = "、".join(config["name"] for config in adapters) or "無"
        print(
            f"裝置 {device}、精度 {dtype}，{time.perf_counter() - started:.0f} 秒；adapter：{names}"
        )
        keep = supports_logits_to_keep(model)
        print(
            "只計算最後一個位置的 logits：" + ("是" if keep else "否（模型不支援 logits_to_keep）")
        )

        total = len(configs) * sum(len(records) for records in datasets.values())
        finished = scored = 0
        clock = time.perf_counter()

        def show(position: int, record: dict, value: dict, reused: bool, *, name, split, size):
            nonlocal finished, scored
            finished += 1
            scored += not reused
            if position % REPORT_EVERY and position != size:
                return
            elapsed = time.perf_counter() - clock
            remaining = elapsed / scored * (total - finished) if scored else 0.0
            print(
                f"[{name}／{split} {position}/{size}] 全部 {finished}/{total}，"
                f"P(系統) {value['p_system']:.3f}；已 {elapsed / 60:.1f} 分，約剩 {remaining / 60:.1f} 分"
            )

        values: dict[str, dict[str, list[dict]]] = {split: {} for split in datasets}
        for config in configs:
            name = config["name"]
            if config["adapter"]:
                model.set_adapter(name)
            # zero-shot：暫時停用所有 adapter，等於基礎模型本身
            context = (
                model.disable_adapter() if adapters and not config["adapter"] else nullcontext()
            )
            where = config["adapter"]["path"] if config["adapter"] else "不掛 adapter"
            print(f"=== {name}（{where}） ===")
            with context:
                for split, records in datasets.items():
                    items, report = score_records(
                        records,
                        score=lambda description: score_record(
                            model, tokenizer, description, label_ids, device, logits_to_keep=keep
                        ),
                        progress=progress[name, split],
                        on_record=partial(show, name=name, split=split, size=len(records)),
                    )
                    values[split][name] = items
                    path = save_scores(out_dir, split, records, values[split], configs, label_ids)
                    print(
                        f"{name}／{split}：評分 {report.scored} 筆、沿用進度 {report.reused} 筆 → {path}"
                    )
    finally:
        model = base = None
        release()


def load_scores(out_dir: Path, datasets: dict[str, list[dict]]) -> dict[str, dict]:
    """讀分數檔，並確認紀錄與資料一致、IDT 提示與目前相同。"""
    loaded = {}
    for split, records in datasets.items():
        path = scores_path(out_dir, split)
        if not path.is_file():
            sys.exit(
                f"找不到分數檔 {path}；請先執行 python -m scripts.05_score_idt --stage score"
                "（試跑要加同樣的 --limit）。"
            )
        data = io.load_json(path)
        if [row["key"] for row in data["records"]] != [record_key(record) for record in records]:
            sys.exit(
                f"{path} 的紀錄與 {split} 資料不一致（筆數或順序不同）；"
                "請用同樣的資料與 --limit 重跑 --stage score。"
            )
        if data.get("idt_prompt") != IDT_PROMPT_ID:
            sys.exit(
                f"{path} 是用 IDT 提示 {data.get('idt_prompt')} 評的，目前是 {IDT_PROMPT_ID}；"
                "請重跑 --stage score --restart。"
            )
        loaded[split] = data
    return loaded


def load_reference(path: Path | None, records: list[dict]) -> list[str | None] | None:
    """04 的 idt_pred，依測試集順序對齊；檔案不存在或沒有任何一筆對得上就回傳 None。"""
    if path is None or not path.is_file():
        return None
    predictions = {record_key(item): item.get("idt_pred") for item in io.load_json(path)}
    aligned = [predictions.get(record_key(record)) for record in records]
    return aligned if any(value is not None for value in aligned) else None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_report(evaluation: dict) -> None:
    system = VALID_IDT_LABELS[1]
    configs = evaluation["configs"]
    selected = (evaluation["selected"] or {}).get("config")
    print(
        f"{'設定':<10}{'val_logloss':>12}{'val_AUROC':>10}{'test_AUROC':>11}{'test_AP':>9} │"
        f"{'F1@0.5':>8}{'R@0.5':>7}{'P@0.5':>7} │{'thr*':>9}{'F1@thr*':>9}{'R@thr*':>8}{'P@thr*':>8}"
    )
    for name, item in configs.items():
        half, tuned = item["test_at_0.5"], item["test_at_threshold"]
        print(
            f"{name + (' *' if name == selected else ''):<12}"
            f"{_fmt(item['val']['log_loss']):>12}{_fmt(item['val']['auroc']):>10}"
            f"{_fmt(item['test']['auroc']):>11}{_fmt(item['test']['ap']):>9} │"
            f"{_fmt(half['macro_f1']):>8}{_fmt(half['per_label'][system]['recall'], 3):>7}"
            f"{_fmt(half['per_label'][system]['precision'], 3):>7} │"
            f"{item['threshold']['value']:>9.4g}{_fmt(tuned['macro_f1']):>9}"
            f"{_fmt(tuned['per_label'][system]['recall'], 3):>8}"
            f"{_fmt(tuned['per_label'][system]['precision'], 3):>8}"
        )
    print(
        "欄位：val／test = 驗證集／測試集；F1 = macro-F1；R、P = 「系統」的 recall、precision；"
        "@0.5 = P(系統) ≥ 0.5 才判系統（與 04 貪婪解碼的判斷相同）；thr* = 驗證集上 macro-F1 "
        "最高的門檻；* = 選出的 checkpoint。"
    )
    if evaluation["selected"]:
        chosen = evaluation["selected"]
        print(
            f"選出的 checkpoint：{chosen['config']}"
            f"（{chosen['criterion']}，log-loss {chosen['val_log_loss']}）"
        )
    bootstrap = evaluation["bootstrap"]
    if bootstrap:
        print(
            f"測試集 95% 信賴區間（bootstrap {bootstrap['resamples']} 次；門檻相關指標用 thr*）："
        )
        for name, intervals in bootstrap["intervals"].items():
            item, tuned = configs[name], configs[name]["test_at_threshold"]
            print(
                f"  {name:<10} macro-F1 {_fmt(tuned['macro_f1'])} {intervals['macro_f1']}，"
                f"系統 recall {_fmt(tuned['per_label'][system]['recall'], 3)} "
                f"{intervals['system_recall']}，AUROC {_fmt(item['test']['auroc'])} {intervals['auroc']}"
            )
            if "vs_baseline" in intervals:
                difference = intervals["vs_baseline"]
                print(
                    f"  {'':<10} 減 {bootstrap['baseline']} 的差距：macro-F1 {difference['macro_f1']}，"
                    f"AUROC {difference['auroc']}（區間不含 0 才算有差）"
                )
    agreements = [
        f"{name} {item['agreement_with_reference']['matched']}/"
        f"{item['agreement_with_reference']['compared']}"
        for name, item in configs.items()
        if "agreement_with_reference" in item and name not in (ZERO_SHOT, BASELINE)
    ]
    if agreements:
        print("0.5 門檻與 04 預測檔（--reference）的一致筆數：" + "、".join(agreements))
    for name, item in evaluation["diagnostics"].items():
        print(
            f"診斷 {name}：兩個標籤的機率合計平均 {item['label_mass_mean']:.3f}，"
            f"機率最高的 token 就是標籤的比例 {item['top_is_label_rate']:.1%}"
        )


def run_evaluation(
    args: argparse.Namespace, datasets: dict[str, list[dict]], out_dir: Path, settings: dict
) -> None:
    print("=== 評估 ===")
    loaded = load_scores(out_dir, datasets)
    val_names = {config["name"] for config in loaded["val"]["configs"]}
    names = [config["name"] for config in loaded["test"]["configs"] if config["name"] in val_names]
    system = VALID_IDT_LABELS[1]
    targets = {
        split: [int(record[LABEL_COLUMN] == system) for record in records]
        for split, records in datasets.items()
    }
    scores = {
        split: {name: [row["p_system"][name] for row in loaded[split]["records"]] for name in names}
        for split in datasets
    }
    if not args.no_baseline:
        if not args.train_data.is_file():
            sys.exit(f"找不到 TF-IDF 基準的訓練資料 {args.train_data}；或加 --no-baseline 略過。")
        try:
            baseline = tfidf_scores(io.load_json(args.train_data), datasets)
        except (RuntimeError, ValueError) as exc:
            sys.exit(f"中止：{exc}")
        for split, items in scores.items():
            items[BASELINE] = baseline[split]
    reference = load_reference(args.reference, datasets["test"])
    summary = summarize_scores(
        targets["val"],
        scores["val"],
        targets["test"],
        scores["test"],
        candidates=[name for name in names if name != ZERO_SHOT],
        bootstrap=args.bootstrap,
        seed=args.seed,
        baseline=None if args.no_baseline else BASELINE,
        reference=reference,
    )
    rows = loaded["val"]["records"] + loaded["test"]["records"]
    diagnostics = {
        name: {
            "label_mass_mean": round(statistics.fmean(row["label_mass"][name] for row in rows), 4),
            "top_is_label_rate": round(
                statistics.fmean(bool(row["top_is_label"][name]) for row in rows), 4
            ),
        }
        for name in names
    }
    evaluation = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": {
            **settings,
            "scored_configs": loaded["test"]["configs"],
            "reference": str(args.reference) if reference else None,
        },
        "label_tokens": loaded["test"]["label_tokens"],
        "diagnostics": diagnostics,
        **summary,
    }
    path = io.save_json(evaluation, out_dir / paths.IDT_SCORE_EVALUATION.name)
    print_report(evaluation)
    print(f"→ {path}")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # 導到檔案或 tee 時進度也能即時看到
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必須大於 0。")
    if args.bootstrap < 0:
        parser.error("--bootstrap 不可為負數。")
    if args.no_zero_shot and args.zero_shot_only:
        parser.error("--no-zero-shot 與 --zero-shot-only 不能同時使用。")
    try:
        model_config = ModelConfig(
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = args.out_dir / f"trial_{args.limit}" if args.limit else args.out_dir
    datasets = load_datasets(args)
    if args.limit:
        print(f"試跑模式：驗證集與測試集各取前 {args.limit} 筆，輸出到 {out_dir}")
    for split, records in datasets.items():
        print(f"{split} {len(records)} 筆（{describe(records)}）")
    run_score = args.stage in ("all", "score")
    configs = resolve_configs(args) if run_score or args.dry_run else None
    settings = {
        "stage": args.stage,
        "limit": args.limit,
        "out_dir": str(out_dir),
        "data": {
            split: {
                "path": str(path),
                "records": len(datasets[split]),
                "label_counts": count_labels(datasets[split]),
            }
            for split, path in (("val", args.val_data), ("test", args.test_data))
        },
        "model": model_config.to_dict(),
        "method": SCORE_METHOD,
        "idt_prompt": IDT_PROMPT_ID,
        "configs": configs,
        "baseline": None
        if args.no_baseline
        else {"name": BASELINE, "train_data": str(args.train_data), **TFIDF_SETTINGS},
        "bootstrap": args.bootstrap,
        "seed": args.seed,
    }
    if args.dry_run:
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        return
    if run_score:
        run_scoring(args, model_config, configs, datasets, out_dir)
    if args.stage in ("all", "evaluate"):
        run_evaluation(args, datasets, out_dir, settings)


if __name__ == "__main__":
    main()
