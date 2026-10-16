"""步驟 01 資料準備：原始 CSV → 清洗 → 依比例切分，對應流程圖的 ①②。

在專案根目錄執行：
    python -m scripts.01_prepare_data
    python -m scripts.01_prepare_data --split-ratio 7 1 2 --split-seed 42
    python -m scripts.01_prepare_data --raw-file data/raw_data/other.csv --encoding cp950

輸出（預設 data/processed_data/）：
    cleaned_data.json                              清洗後全部紀錄
    train_data.json / val_data.json / test_data.json   切分結果
    prepare_summary.json                           本次輸入、剔除原因、各集合筆數與切分設定

增生與情緒標註只對訓練集進行，屬於下一步特徵工程，不在此執行。
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from utils import io, paths
from utils.preprocessing import DROP_REASONS, clean_records, count_labels
from utils.split import (
    DEFAULT_RATIO,
    DEFAULT_SEED,
    SPLIT_NAMES,
    shared_groups,
    split_records,
    validate_ratio,
    validate_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="資料準備：清洗原始 CSV，並依 IDT 標籤分層切分訓練／驗證／測試集"
    )
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=paths.RAW_CSV,
        help=f"原始 CSV，預設 {paths.RAW_CSV.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="原始 CSV 編碼；Excel 另存的「CSV UTF-8」用預設即可，舊版 Excel 存的 Big5 改用 cp950",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.PROCESSED_DATA_DIR,
        help=f"輸出目錄，預設 {paths.PROCESSED_DATA_DIR.relative_to(paths.PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--split-ratio",
        nargs=3,
        type=float,
        default=list(DEFAULT_RATIO),
        metavar=("TRAIN", "VAL", "TEST"),
        help="訓練、驗證、測試的比例，預設 7 1 2",
    )
    # 切分種子與之後訓練的 seed 分開，換訓練種子比較結果時資料集不會跟著變。
    parser.add_argument(
        "--split-seed", type=int, default=DEFAULT_SEED, help="切分的隨機種子，預設 42"
    )
    return parser


def format_counts(records: list[dict]) -> str:
    return "／".join(f"{label} {count}" for label, count in count_labels(records).items())


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # 導到檔案或 tee 時進度也能即時看到
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ratio = validate_ratio(args.split_ratio)
        validate_seed(args.split_seed)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.raw_file.is_file():
        parser.error(f"找不到原始資料 {args.raw_file}")

    print(f"=== 1. 讀取與清洗：{args.raw_file} ===")
    try:
        rows = io.read_csv_records(args.raw_file, encoding=args.encoding)
        records, report = clean_records(rows, source_name=args.raw_file.name)
    except (ValueError, UnicodeDecodeError) as exc:
        sys.exit(f"讀取失敗：{exc}")
    print(f"輸入 {report.input_rows} 列")
    for key, reason in DROP_REASONS.items():
        print(f"  剔除 {reason}：{report.dropped[key]} 列")
    print(f"輸出 {report.output_rows} 筆（{format_counts(records)}）")
    if not records:
        sys.exit("清洗後沒有可用的紀錄，請檢查原始資料。")
    cleaned_path = io.save_json(records, args.out_dir / paths.CLEANED_DATA.name)
    print(f"→ {cleaned_path}")

    ratio_text = ":".join(f"{value:.0%}" for value in ratio)
    print(f"=== 2. 依比例切分 {ratio_text}（seed {args.split_seed}）===")
    splits = split_records(records, ratio=ratio, seed=args.split_seed)
    shared = shared_groups(splits)
    if shared:
        sys.exit(f"切分錯誤：有 {len(shared)} 組描述同時出現在多個集合。")
    sets_summary = {}
    for name in SPLIT_NAMES:
        path = io.save_json(splits[name], args.out_dir / paths.SPLIT_DATA[name].name)
        print(f"{name:<5} {len(splits[name]):>5} 筆（{format_counts(splits[name])}）→ {path}")
        sets_summary[name] = {
            "file": path.name,
            "rows": len(splits[name]),
            "label_counts": count_labels(splits[name]),
        }
    print("描述跨集合：0 組")

    summary_path = io.save_json(
        {
            "clean": report.to_dict(),
            "split": {"ratio": list(ratio), "seed": args.split_seed, "sets": sets_summary},
        },
        args.out_dir / paths.PREPARE_SUMMARY.name,
    )
    print(f"→ {summary_path}")


if __name__ == "__main__":
    main()
