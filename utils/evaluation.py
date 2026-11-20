"""步驟 ⑧⑨ 的計算部分：IDT 分類指標與情緒評審分數的彙整，不呼叫模型。

IDT 有人工標籤，算 accuracy、混淆矩陣（列 = 人工、欄 = 預測）與各類別的
precision／recall／F1；「系統」類別少，看混淆矩陣與召回率比看 accuracy 有意義。
情緒沒有人工真值，只能彙整評審模型給的品質分數，不是 accuracy。

05 的機率評分另外用到不依賴門檻的 AUROC、AP、log-loss，驗證集調門檻與 bootstrap 信賴區間。
這些函式的 targets 是 1 =「系統」、0 =「個人」；weights 是重抽樣時每筆被抽中的次數，
不給就是每筆 1 次。
"""

import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import pairwise

from utils.preprocessing import LABEL_COLUMN, VALID_IDT_LABELS
from utils.prompts import JUDGE_DIMENSIONS

IDT_PREDICTION = "idt_pred"
EMOTION_PREDICTION = "emotion_pred"
EMOTION_JUDGE = "emotion_judge"


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def idt_metrics(
    records: list[dict],
    *,
    target: str = LABEL_COLUMN,
    prediction: str = IDT_PREDICTION,
    labels: tuple[str, ...] = VALID_IDT_LABELS,
) -> dict:
    """預測為 None 或不在合法標籤內的紀錄列為 invalid，計入分母、算作錯誤，不猜標籤。"""
    labels = tuple(labels)
    matrix = [[0] * len(labels) for _ in labels]
    invalid_by_label = Counter()
    for record in records:
        row = labels.index(record[target])
        predicted = record.get(prediction)
        if predicted in labels:
            matrix[row][labels.index(predicted)] += 1
        else:
            invalid_by_label[record[target]] += 1
    correct = sum(matrix[i][i] for i in range(len(labels)))
    per_label = {}
    for i, label in enumerate(labels):
        true_positive = matrix[i][i]
        support = sum(matrix[i]) + invalid_by_label[label]
        predicted = sum(matrix[j][i] for j in range(len(labels)))
        precision = _ratio(true_positive, predicted)
        recall = _ratio(true_positive, support)
        per_label[label] = {
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall),
        }
    return {
        "records": len(records),
        "correct": correct,
        "accuracy": _ratio(correct, len(records)),
        "invalid_predictions": sum(invalid_by_label.values()),
        "macro_f1": _ratio(sum(item["f1"] for item in per_label.values()), len(labels)),
        "confusion_matrix": {
            "labels": list(labels),
            "rows": f"{target}（人工）",
            "cols": f"{prediction}（預測）",
            "matrix": matrix,
            "invalid_by_label": {label: invalid_by_label[label] for label in labels},
        },
        "per_label": per_label,
    }


def label_distribution(records: list[dict], field: str) -> dict:
    """某個標籤欄位的分布；None 另計為 invalid。"""
    counts = Counter(record.get(field) for record in records)
    invalid = counts.pop(None, 0)
    return {"counts": dict(counts.most_common()), "invalid": invalid}


def judge_summary(records: list[dict], field: str = EMOTION_JUDGE) -> dict:
    """評審分數的平均；只算有分數的紀錄，沒評到的另計。"""
    scored = [record[field] for record in records if record.get(field)]
    mean_scores = {
        key: _ratio(sum(item["scores"][key] for item in scored), len(scored))
        for key in JUDGE_DIMENSIONS
    }
    overall = [item["overall"] for item in scored]
    return {
        "records": len(records),
        "scored": len(scored),
        "unscored": len(records) - len(scored),
        "models": sorted({item["model"] for item in scored if item.get("model")}),
        "mean_scores": mean_scores,
        "mean_overall_score": _ratio(sum(overall), len(overall)),
        "overall_distribution": dict(sorted(Counter(round(value) for value in overall).items())),
    }


# ---- 05 機率評分：不依賴門檻的指標、門檻選擇與 bootstrap ----

BOOTSTRAP_METRICS = ("macro_f1", "system_precision", "system_recall", "system_f1", "auroc", "ap")
PAIRED_METRICS = ("macro_f1", "auroc")


def _weights(targets: Sequence[int], weights: Sequence[int] | None) -> Sequence[int]:
    return [1] * len(targets) if weights is None else weights


def _class_totals(targets, weights) -> tuple[float, float]:
    positives = sum(weight for target, weight in zip(targets, weights, strict=True) if target)
    return positives, sum(weights) - positives


def roc_auc(
    targets: Sequence[int], scores: Sequence[float], weights: Sequence[int] | None = None
) -> float | None:
    """「系統」的分數高於「個人」的機率，同分算一半；只有單一類別時無法計算，回傳 None。"""
    weights = _weights(targets, weights)
    positives, negatives = _class_totals(targets, weights)
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    wins = negatives_below = 0.0
    start = 0
    while start < len(order):
        end, tied_positives, tied_negatives = start, 0, 0
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            if targets[order[end]]:
                tied_positives += weights[order[end]]
            else:
                tied_negatives += weights[order[end]]
            end += 1
        wins += tied_positives * (negatives_below + tied_negatives / 2)
        negatives_below += tied_negatives
        start = end
    return wins / (positives * negatives)


def average_precision(
    targets: Sequence[int], scores: Sequence[float], weights: Sequence[int] | None = None
) -> float | None:
    """分數由高到低逐一當門檻，累加「召回率增量 × 精確率」；同分視為同一個門檻，與 scikit-learn
    相同。沒有「系統」時無法計算，回傳 None。"""
    weights = _weights(targets, weights)
    positives, _ = _class_totals(targets, weights)
    if not positives:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    total = true_positives = false_positives = previous_recall = 0.0
    start = 0
    while start < len(order):
        end = start
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            if targets[order[end]]:
                true_positives += weights[order[end]]
            else:
                false_positives += weights[order[end]]
            end += 1
        if true_positives + false_positives:
            recall = true_positives / positives
            precision = true_positives / (true_positives + false_positives)
            total += (recall - previous_recall) * precision
            previous_recall = recall
        start = end
    return total


def binary_log_loss(
    targets: Sequence[int], probabilities: Sequence[float], eps: float = 1e-12
) -> float:
    """平均交叉熵（nats）；機率夾在 [eps, 1 − eps]，避免 log(0)。"""
    total = 0.0
    for target, probability in zip(targets, probabilities, strict=True):
        probability = min(max(probability, eps), 1 - eps)
        total -= math.log(probability if target else 1 - probability)
    return total / len(targets)


def _f1(true_positive: float, false_positive: float, false_negative: float) -> float:
    if not true_positive:
        return 0.0
    return 2 * true_positive / (2 * true_positive + false_positive + false_negative)


def threshold_metrics(
    targets: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    weights: Sequence[int] | None = None,
) -> dict:
    """分數 ≥ 門檻判「系統」；回傳不四捨五入的 macro-F1 與「系統」的 precision／recall／F1。"""
    weights = _weights(targets, weights)
    tp = fp = fn = tn = 0
    for target, score, weight in zip(targets, scores, weights, strict=True):
        if score >= threshold:
            tp, fp = (tp + weight, fp) if target else (tp, fp + weight)
        else:
            fn, tn = (fn + weight, tn) if target else (fn, tn + weight)
    return {
        # 「個人」當正類時，TN 是它的 TP、FN 是它的 FP、FP 是它的 FN
        "macro_f1": (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2,
        "system_precision": tp / (tp + fp) if tp + fp else 0.0,
        "system_recall": tp / (tp + fn) if tp + fn else 0.0,
        "system_f1": _f1(tp, fp, fn),
    }


def best_threshold(targets: Sequence[int], scores: Sequence[float]) -> dict:
    """在驗證集找 macro-F1 最高的門檻（分數 ≥ 門檻判「系統」），候選是相鄰分數的中點；
    同分時取最接近 0.5 的。驗證集只有單一類別時無從比較，沿用 0.5。"""
    positives = sum(targets)
    if positives in (0, len(targets)):
        return {
            "threshold": 0.5,
            "macro_f1": None,
            "note": "驗證集只有單一類別，無法調門檻，沿用 0.5",
        }
    distinct = sorted(set(scores))
    candidates = [distinct[0]] + [(low + high) / 2 for low, high in pairwise(distinct)]
    best_key, best = None, None
    for threshold in candidates:
        value = threshold_metrics(targets, scores, threshold)["macro_f1"]
        key = (value, -abs(threshold - 0.5))
        if best_key is None or key > best_key:
            best_key, best = key, {"threshold": threshold, "macro_f1": value, "note": None}
    return best


def _percentile(values: list[float], q: float) -> float:
    """線性內插的百分位數（與 numpy 預設相同）。"""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_intervals(
    targets: Sequence[int],
    systems: dict[str, tuple[Sequence[float], float]],
    *,
    n: int,
    seed: int,
    baseline: str | None = None,
    confidence: float = 0.95,
) -> dict:
    """測試集重抽樣 n 次（每次抽同樣筆數、可重複），所有設定共用同一組重抽樣；systems 是
    {名稱: (分數, 門檻)}。回傳各指標的百分位信賴區間；給 baseline 時另附配對差距（設定 − 基準）。
    抽到只有單一類別的那次略過。"""
    rng = random.Random(seed)
    size = len(targets)
    samples = {name: defaultdict(list) for name in systems}
    differences = {name: defaultdict(list) for name in systems if baseline and name != baseline}
    for _ in range(n):
        weights = [0] * size
        for _ in range(size):
            weights[rng.randrange(size)] += 1
        positives, negatives = _class_totals(targets, weights)
        if not positives or not negatives:
            continue
        values = {}
        for name, (scores, threshold) in systems.items():
            metrics = threshold_metrics(targets, scores, threshold, weights)
            metrics["auroc"] = roc_auc(targets, scores, weights)
            metrics["ap"] = average_precision(targets, scores, weights)
            values[name] = metrics
            for key, value in metrics.items():
                samples[name][key].append(value)
        for name, items in differences.items():
            for key in PAIRED_METRICS:
                items[key].append(values[name][key] - values[baseline][key])
    tail = (1 - confidence) / 2

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        return [round(_percentile(values, tail), 4), round(_percentile(values, 1 - tail), 4)]

    result = {}
    for name in systems:
        result[name] = {key: interval(samples[name][key]) for key in BOOTSTRAP_METRICS}
        if name in differences:
            result[name]["vs_baseline"] = {
                key: interval(differences[name][key]) for key in PAIRED_METRICS
            }
    return result


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _labelled(targets: Sequence[int], scores: Sequence[float], threshold: float) -> list[dict]:
    """把 0／1 與分數轉回 idt_metrics 用的紀錄：分數 ≥ 門檻判「系統」。"""
    person, system = VALID_IDT_LABELS
    return [
        {
            LABEL_COLUMN: system if target else person,
            IDT_PREDICTION: system if score >= threshold else person,
        }
        for target, score in zip(targets, scores, strict=True)
    ]


def summarize_scores(
    val_targets: Sequence[int],
    val_scores: dict[str, Sequence[float]],
    test_targets: Sequence[int],
    test_scores: dict[str, Sequence[float]],
    *,
    candidates: Sequence[str],
    bootstrap: int,
    seed: int,
    baseline: str | None = None,
    reference: Sequence[str | None] | None = None,
) -> dict:
    """05 的彙整：每個設定的驗證／測試 log-loss、AUROC、AP，驗證集調出的門檻，以及測試集在
    0.5 與調整後門檻的指標（同 ⑧ 的 idt_metrics）。

    checkpoint 只從 candidates 中挑驗證集 log-loss 最低的；zero-shot 與 TF-IDF 是基準，不參與挑選。
    reference 是 04 對測試集的 idt_pred（與 test_targets 同序，None 表示沒有），用來核對 0.5 門檻
    的判斷與貪婪解碼是否一致。bootstrap 是重抽樣次數，0 表示不算信賴區間。
    """
    configs: dict[str, dict] = {}
    val_log_loss: dict[str, float] = {}
    for name, test in test_scores.items():
        val = val_scores[name]
        val_log_loss[name] = binary_log_loss(val_targets, val)
        threshold = best_threshold(val_targets, val)
        item = {
            "val": {
                "log_loss": round(val_log_loss[name], 4),
                "auroc": _rounded(roc_auc(val_targets, val)),
                "ap": _rounded(average_precision(val_targets, val)),
            },
            "test": {
                "log_loss": round(binary_log_loss(test_targets, test), 4),
                "auroc": _rounded(roc_auc(test_targets, test)),
                "ap": _rounded(average_precision(test_targets, test)),
            },
            "threshold": {
                "value": threshold["threshold"],
                "val_macro_f1": _rounded(threshold["macro_f1"]),
                "note": threshold["note"],
            },
            "test_at_0.5": idt_metrics(_labelled(test_targets, test, 0.5)),
            "test_at_threshold": idt_metrics(_labelled(test_targets, test, threshold["threshold"])),
        }
        if reference is not None:
            pairs = [
                (row[IDT_PREDICTION], expected)
                for row, expected in zip(_labelled(test_targets, test, 0.5), reference, strict=True)
                if expected is not None
            ]
            item["agreement_with_reference"] = {
                "matched": sum(predicted == expected for predicted, expected in pairs),
                "compared": len(pairs),
            }
        configs[name] = item
    pool = [name for name in candidates if name in configs]
    selected = None
    if pool:
        best = min(pool, key=val_log_loss.__getitem__)
        selected = {
            "config": best,
            "criterion": "只在訓練出的 checkpoint 中挑驗證集 log-loss 最低的",
            "val_log_loss": round(val_log_loss[best], 4),
        }
    intervals = None
    if bootstrap:
        systems = {
            name: (test_scores[name], configs[name]["threshold"]["value"]) for name in configs
        }
        intervals = {
            "resamples": bootstrap,
            "seed": seed,
            "confidence": 0.95,
            "baseline": baseline,
            "operating_point": "門檻相關指標用各設定在驗證集調出的門檻",
            "intervals": bootstrap_intervals(
                test_targets, systems, n=bootstrap, seed=seed, baseline=baseline
            ),
        }
    return {"configs": configs, "selected": selected, "bootstrap": intervals}
