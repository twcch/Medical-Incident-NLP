"""步驟 ⑧⑨ 的計算部分：IDT 分類指標與情緒評審分數的彙整，不呼叫模型。

IDT 有人工標籤，算 accuracy、混淆矩陣（列 = 人工、欄 = 預測）與各類別的
precision／recall／F1；「系統」類別少，看混淆矩陣與召回率比看 accuracy 有意義。
情緒沒有人工真值，只能彙整評審模型給的品質分數，不是 accuracy。
"""

from collections import Counter

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
