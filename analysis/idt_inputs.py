"""只讀事件描述夠不夠判斷 IDT：以 TF-IDF＋邏輯迴歸做的探索性分析（實驗報告 4.6 節用）。

模型只讀 description，批示（directive）不送模型。這支腳本用 05 的 TF-IDF 基準（同樣的設定）
回答幾個問題，結果只是資訊量的粗略估計，不是正式實驗：

    fields          同一個模型換輸入欄位（description／directive／兩者），測試集的指標
    cross_validation 全部資料的分組 5-fold（相同描述同組、三個亂數種子），含只用部分訓練資料的學習曲線
    near_duplicates 描述幾乎相同（字元 2–3 gram 的 cosine 相似度）的紀錄，標籤不同的有幾對
    same_text       描述一字不差、標籤卻不同的紀錄
    directive_cues  批示裡的流程用語（敬會、PDCA 等）在兩種標籤出現的比例
    event_types     以關鍵字粗分事件類型，各類型的「系統」比例
    augmentation    訓練資料加入 02 的改寫後，同一個測試集的指標

在專案根目錄執行（不載模型，約 2 分鐘）：
    python -m analysis.idt_inputs                         # → results/analysis/idt_inputs.json
    python -m analysis.idt_inputs <輸出目錄>
"""

import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from utils import io
from utils.baseline import TFIDF_SETTINGS
from utils.evaluation import roc_auc, threshold_metrics
from utils.preprocessing import DIRECTIVE_COLUMN, LABEL_COLUMN, TEXT_COLUMN

ROOT = Path.cwd()
DATA = ROOT / "data/processed_data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/analysis"
SYSTEM = "系統"
SEEDS = (0, 1, 2)
FIELDS = {
    "description": lambda r: r[TEXT_COLUMN],
    "directive": lambda r: r.get(DIRECTIVE_COLUMN) or "",
    "description+directive": lambda r: r[TEXT_COLUMN] + " ||| " + (r.get(DIRECTIVE_COLUMN) or ""),
}
DIRECTIVE_CUES = {
    "敬會": r"敬會",
    "PDCA": r"(?i)pdca",
    "RCA／根本原因": r"(?i)\brca\b|根本原因",
    "資訊室／總務室／工務／廠商": r"資訊室|總務室|工務|廠商",
}
CUE_PATTERN = re.compile(r"敬會|pdca|\brca\b|根本原因|資訊室|總務室|工務|廠商", flags=re.IGNORECASE)
EVENT_TYPES = {
    "跌倒": r"跌倒|跌落|滑倒",
    "自拔管路": r"自拔|自行拔",
    "給藥": r"給錯|劑量|給藥|藥物錯|錯藥",
    "檢驗／檢體": r"檢驗|檢體|採檢",
    "傳送": r"傳送",
    "設備／資訊": r"設備|儀器|故障|當機|系統|電腦|資訊",
    "輸血": r"輸血|血品|領血",
}


def targets(records):
    return np.array([int(r[LABEL_COLUMN] == SYSTEM) for r in records])


def fit_predict(train_texts, train_y, eval_texts):
    """05 的 TF-IDF 基準：以 train 訓練，回傳 eval 每筆的 P(系統)。"""
    options = TFIDF_SETTINGS["vectorizer"]
    vectorizer = TfidfVectorizer(
        analyzer=options["analyzer"],
        ngram_range=tuple(options["ngram_range"]),
        min_df=options["min_df"],
        sublinear_tf=options["sublinear_tf"],
    )
    classifier = LogisticRegression(**TFIDF_SETTINGS["classifier"])
    classifier.fit(vectorizer.fit_transform(train_texts), train_y)
    return classifier.predict_proba(vectorizer.transform(eval_texts))[:, 1]


def metrics(y, scores, threshold=0.5):
    predicted = (scores >= threshold).astype(int)
    precision, recall, _, _ = precision_recall_fscore_support(y, predicted, labels=[1], zero_division=0)
    return {
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "system_precision": float(precision[0]),
        "system_recall": float(recall[0]),
        "auroc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
    }


def cross_validate(texts, y, groups, fraction=1.0):
    """分組 5-fold 的 out-of-fold 分數算指標，三個亂數種子取平均；fraction < 1 時每折只用部分訓練資料。"""
    rows = []
    for seed in SEEDS:
        scores = np.zeros(len(texts))
        rng = np.random.default_rng(seed)
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, test_index in folds.split(texts, y, groups):
            if fraction < 1:
                train_index = rng.choice(train_index, int(len(train_index) * fraction), replace=False)
            scores[test_index] = fit_predict(
                [texts[i] for i in train_index], y[train_index], [texts[i] for i in test_index]
            )
        rows.append(metrics(y, scores))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main():
    sets = {name: io.load_json(DATA / f"{name}_data.json") for name in ("train", "val", "test")}
    everything = io.load_json(DATA / "cleaned_data.json")
    y_all, y_train, y_test = targets(everything), targets(sets["train"]), targets(sets["test"])
    result = {"records": len(everything), "system": int(y_all.sum()), "tfidf": TFIDF_SETTINGS, "seeds": list(SEEDS)}

    # 同一個模型換輸入欄位；信賴區間與配對差距用 05 的重抽樣（random.Random(42)、2,000 次）
    variants = {**FIELDS, "directive（去掉流程用語）": lambda r: CUE_PATTERN.sub("", r.get(DIRECTIVE_COLUMN) or "")}
    field_scores = {
        name: fit_predict([get(r) for r in sets["train"]], y_train, [get(r) for r in sets["test"]]).tolist()
        for name, get in variants.items()
    }
    truth = y_test.tolist()
    rng = random.Random(42)
    samples = {name: {"macro_f1": [], "auroc": [], "system_recall": []} for name in field_scores}
    for _ in range(2000):
        weights = [0] * len(truth)
        for _ in range(len(truth)):
            weights[rng.randrange(len(truth))] += 1
        if not 0 < sum(w for w, t in zip(weights, truth, strict=True) if t) < len(truth):
            continue
        for name, scores in field_scores.items():
            at_half = threshold_metrics(truth, scores, 0.5, weights)
            samples[name]["macro_f1"].append(at_half["macro_f1"])
            samples[name]["system_recall"].append(at_half["system_recall"])
            samples[name]["auroc"].append(roc_auc(truth, scores, weights))

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    result["fields"] = {}
    for name, scores in field_scores.items():
        item = metrics(y_test, np.array(scores))
        item["ci"] = {key: interval(values) for key, values in samples[name].items()}
        if name != "description":
            item["minus_description"] = {
                key: {
                    "diff": item[key] - result["fields"]["description"][key],
                    "ci": interval([a - b for a, b in zip(values, samples["description"][key], strict=True)]),
                }
                for key, values in samples[name].items()
            }
        result["fields"][name] = item

    # 批示裡有「原因說明」的比例（規劃「調查後輔助判定」時參考）
    has_reason = np.array(["原因說明" in (r.get(DIRECTIVE_COLUMN) or "") for r in everything])
    result["reason_coverage"] = {
        "all": float(has_reason.mean()),
        "system": float(has_reason[y_all == 1].mean()),
        "person": float(has_reason[y_all == 0].mean()),
    }

    # 分組交叉驗證：相同描述同組（組別編號用第一次出現的位置，每次執行都一樣）
    first_seen: dict[str, int] = {}
    groups = np.array([first_seen.setdefault(r[TEXT_COLUMN], len(first_seen)) for r in everything])
    result["cross_validation"] = {
        name: cross_validate([get(r) for r in everything], y_all, groups) for name, get in FIELDS.items()
    }
    descriptions = [r[TEXT_COLUMN] for r in everything]
    result["learning_curve"] = {
        f"{fraction:.2f}": cross_validate(descriptions, y_all, groups, fraction) for fraction in (0.25, 0.5, 0.75)
    }
    result["learning_curve"]["1.00"] = result["cross_validation"]["description"]

    # 描述幾乎相同的紀錄，標籤不同的有幾對
    matrix = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), sublinear_tf=True).fit_transform(descriptions)
    similarity = (matrix @ matrix.T).tocoo()
    result["near_duplicates"] = {}
    for level in (0.95, 0.9, 0.85, 0.8):
        mask = (similarity.row < similarity.col) & (similarity.data >= level)
        differ = y_all[similarity.row[mask]] != y_all[similarity.col[mask]]
        result["near_duplicates"][f"{level:.2f}"] = {"pairs": int(mask.sum()), "label_conflicts": int(differ.sum())}

    # 描述一字不差、標籤卻不同
    by_text: dict[str, list[dict]] = {}
    for record in everything:
        by_text.setdefault(record[TEXT_COLUMN].strip(), []).append(record)
    result["same_text"] = {
        "groups": sum(len(rows) > 1 for rows in by_text.values()),
        "conflicting": [
            {"description": text, "records": [[r["source_id"], r[LABEL_COLUMN]] for r in rows]}
            for text, rows in by_text.items()
            if len({r[LABEL_COLUMN] for r in rows}) > 1
        ],
    }

    # 批示裡的流程用語、事件類型的「系統」比例
    system_rows = [r for r in everything if r[LABEL_COLUMN] == SYSTEM]
    person_rows = [r for r in everything if r[LABEL_COLUMN] != SYSTEM]
    result["directive_cues"] = {
        name: {
            "system_rate": sum(bool(re.search(pattern, r.get(DIRECTIVE_COLUMN) or "")) for r in system_rows) / len(system_rows),
            "person_rate": sum(bool(re.search(pattern, r.get(DIRECTIVE_COLUMN) or "")) for r in person_rows) / len(person_rows),
        }
        for name, pattern in DIRECTIVE_CUES.items()
    }
    result["event_types"] = {}
    for name, pattern in EVENT_TYPES.items():
        rows = [r for r in everything if re.search(pattern, r[TEXT_COLUMN])]
        system = sum(r[LABEL_COLUMN] == SYSTEM for r in rows)
        result["event_types"][name] = {"records": len(rows), "system": system, "system_rate": system / len(rows)}

    # 訓練資料加入 02 的改寫
    features_path = ROOT / "data/features/train_features.json"
    if features_path.is_file():
        features = io.load_json(features_path)
        test_texts = [r[TEXT_COLUMN] for r in sets["test"]]
        result["augmentation"] = {
            name: {"records": len(rows), **metrics(y_test, fit_predict([r[TEXT_COLUMN] for r in rows], targets(rows), test_texts))}
            for name, rows in (
                ("originals", sets["train"]),
                ("originals+paraphrases", features),
                ("paraphrases", [r for r in features if r.get("augmented")]),
            )
        }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "idt_inputs.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print("== 換輸入欄位（測試集，門檻 0.5）｜分組 5-fold ==")
    for name, test in result["fields"].items():
        cv = result["cross_validation"].get(name)
        extra = f"｜5-fold macro-F1 {cv['macro_f1']:.3f} AUROC {cv['auroc']:.3f}" if cv else ""
        diff = test.get("minus_description")
        gap = (
            f"｜減 description：macro-F1 {diff['macro_f1']['diff']:+.3f} [{diff['macro_f1']['ci'][0]:+.3f}, {diff['macro_f1']['ci'][1]:+.3f}]、"
            f"AUROC {diff['auroc']['diff']:+.3f} [{diff['auroc']['ci'][0]:+.3f}, {diff['auroc']['ci'][1]:+.3f}]"
            if diff
            else ""
        )
        print(
            f"  {name:22s} macro-F1 {test['macro_f1']:.3f} R {test['system_recall']:.3f} P {test['system_precision']:.3f} "
            f"AUROC {test['auroc']:.3f} [{test['ci']['auroc'][0]:.3f}, {test['ci']['auroc'][1]:.3f}]{extra}{gap}"
        )
    print("== 批示含「原因說明」的比例 ==", {k: round(v, 3) for k, v in result["reason_coverage"].items()})
    print("== 學習曲線（description，5-fold macro-F1）==", {k: round(v["macro_f1"], 3) for k, v in result["learning_curve"].items()})
    print("== 近似重複 ==", result["near_duplicates"])
    print("== 同文不同標籤 ==", [(c["description"][:20], c["records"]) for c in result["same_text"]["conflicting"]])
    print("== 批示用語 ==", {k: (round(v["system_rate"], 3), round(v["person_rate"], 3)) for k, v in result["directive_cues"].items()})
    print("== 事件類型 ==", {k: (v["records"], round(v["system_rate"], 3)) for k, v in result["event_types"].items()})
    if "augmentation" in result:
        print("== 加入改寫 ==", {k: round(v["macro_f1"], 4) for k, v in result["augmentation"].items()})
    print(f"→ {path}")


if __name__ == "__main__":
    main()
