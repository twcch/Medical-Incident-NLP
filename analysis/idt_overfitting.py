"""選定的 checkpoint 有沒有過度擬合：論文稿 3.11、4.5 節的數字都在這裡算。

只讀 05 的分數檔、03 的訓練紀錄與資料檔，不載模型。四組結果（論文稿用前三組；第四組的校準沒有寫進論文稿，
留在輸出檔備查，審稿人問到校準時可以直接取用）：

    training        03 記下的每個 epoch 訓練損失與驗證損失；選定的 checkpoint 是不是驗證損失最低的那個
    generalization  各設定在驗證集與測試集的 log-loss／AUROC／AP，以及門檻固定在驗證集調出的 thr* 時的
                    macro-F1 與召回率；「測試集 − 驗證集」的差與 95% 區間（兩個集合各自獨立重抽樣）
    similarity      每筆測試通報與訓練集裡最像的那一筆的相似度（字元 2–3 gram TF-IDF 的 cosine，與
                    analysis.idt_inputs 的近似描述分析相同）；拿掉高度相似的測試通報後的 AUROC；
                    依相似度的中位數分成兩半後的 AUROC，以及微調相對於 zero-shot 的配對增益
    calibration     校準斜率（logit(P) 對標籤的邏輯迴歸係數；1 最理想，小於 1 表示機率太極端）、斜率固定為 1 時
                    的截距、觀察／預期比、ECE、Brier；另把驗證集配適的 logistic 再校準套到測試集

沒有的東西：訓練集上的分數。05 只評了驗證集與測試集，所以這裡沒有「訓練集對測試集」的落差；
訓練損失只有 03 記下的每個 epoch 平均（訓練模式、權重還在變），不能和驗證損失直接相減。

測試集的重抽樣與 05、analysis.idt_comparison 相同（random.Random(42)、2,000 次、單一類別的那次略過），
驗證集另外用 random.Random(43) 抽；分數也和 05 一樣先合併同分。

在專案根目錄執行（約 2 分鐘）：
    python -m analysis.idt_overfitting                    # → results/analysis/idt_overfitting.json
    python -m analysis.idt_overfitting <輸出目錄>
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import io
from utils.baseline import tfidf_scores
from utils.emotion import record_key
from utils.evaluation import (
    _percentile,
    average_precision,
    binary_log_loss,
    merge_near_ties,
    roc_auc,
    threshold_metrics,
)
from utils.preprocessing import LABEL_COLUMN, TEXT_COLUMN

ROOT = Path.cwd()
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/analysis"
DATA = ROOT / "data/processed_data"
SYSTEM = "系統"
RUNS = {"v1": ROOT / "results/idt_scores", "v2": ROOT / "results/idt_scores_v2"}
ADAPTERS = {"v1": ROOT / "models/idt_adapter", "v2": ROOT / "models/idt_adapter_v2"}
CONFIGS = ("v1-zero-shot", "v1-epoch-1", "v1-epoch-2", "v1-epoch-3", "v2-epoch-1", "v2-epoch-2", "tfidf")
RESAMPLES = 2000
NEAR_DUPLICATE = 0.8  # 與 analysis.idt_inputs 的「高度相似」同一個門檻
ECE_BINS = 10

# ---------------- 讀資料與分數（與 analysis.idt_comparison 相同的整理方式）----------------
datasets = {s: io.load_json(DATA / f"{s}_data.json") for s in ("train", "val", "test")}
targets = {s: [int(r[LABEL_COLUMN] == SYSTEM) for r in datasets[s]] for s in ("val", "test")}
scores = {"val": {}, "test": {}}
thr, selected = {}, {}
for run, out in RUNS.items():
    evaluation = io.load_json(out / "idt_score_evaluation.json")
    selected[run] = f"{run}-{evaluation['selected']['config']}"
    for s in ("val", "test"):
        loaded = io.load_json(out / f"scores_{s}.json")
        assert [r["key"] for r in loaded["records"]] == [record_key(r) for r in datasets[s]], (run, s)
        for name in loaded["records"][0]["p_system"]:
            scores[s][f"{run}-{name}"] = [row["p_system"][name] for row in loaded["records"]]
    for name, config in evaluation["configs"].items():
        thr[name if name == "tfidf" else f"{run}-{name}"] = config["threshold"]["value"]
tf = tfidf_scores(datasets["train"], {s: datasets[s] for s in ("val", "test")})
for s in ("val", "test"):
    scores[s]["tfidf"] = tf[s]
for key in CONFIGS:  # 與 05 相同：驗證集與測試集一起合併同分
    size = len(scores["val"][key])
    merged = merge_near_ties([*scores["val"][key], *scores["test"][key]])
    scores["val"][key], scores["test"][key] = merged[:size], merged[size:]
SELECTED = (selected["v1"], selected["v2"])
assert SELECTED == ("v1-epoch-1", "v2-epoch-2"), SELECTED


def ci(values):
    values = list(values)  # 也接受 generator
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def draw(size, seed, labels):
    """重抽樣的權重（每筆被抽到幾次）；只抽到單一類別的那次略過，與 05 相同。"""
    rng = random.Random(seed)
    out = []
    for _ in range(RESAMPLES):
        w = [0] * size
        for _ in range(size):
            w[rng.randrange(size)] += 1
        if 0 < sum(wi for wi, t in zip(w, labels, strict=True) if t) < size:
            out.append(w)
    return out


RT = draw(len(targets["test"]), 42, targets["test"])  # 與 idt_comparison 的 resamples 相同
RV = draw(len(targets["val"]), 43, targets["val"])
PAIRS = list(zip(RV, RT, strict=False))  # 兩個集合各自獨立抽，依序配成一對算差


def weighted_log_loss(labels, probabilities, weights, eps=1e-12):
    total = 0.0
    for t, p, w in zip(labels, probabilities, weights, strict=True):
        if w:
            p = min(max(p, eps), 1 - eps)
            total -= w * np.log(p if t else 1 - p)
    return float(total / sum(weights))


# ---------------- 1. 訓練紀錄 ----------------
training = {}
for run, adapter in ADAPTERS.items():
    meta = io.load_json(adapter / "training_metadata.json")
    history = meta["history"]
    best = min(history, key=lambda h: h["val_loss"])
    chosen = int(selected[run].rsplit("-", 1)[1])
    training[run] = {
        "trainable_params": meta["lora"]["trainable_params"],
        "trainable_share": meta["lora"]["trainable_params"] / meta["lora"]["total_params"],
        "train_records": meta["data"]["train_records"],
        "answer_tokens": meta["data"]["token_lengths"]["train"]["label_tokens"],
        "lora_dropout": meta["training"]["lora_dropout"],
        "history": [
            {k: h[k] for k in ("epoch", "global_step", "train_loss", "val_loss")}
            | {
                "selected": h["epoch"] == chosen,
                # 式 (2) 是每個答案 token 的平均；乘上 token 數應接近 05 以標籤機率算的 log-loss
                "val_log_loss": binary_log_loss(targets["val"], scores["val"][f"{run}-epoch-{h['epoch']}"]),
                "test_log_loss": binary_log_loss(targets["test"], scores["test"][f"{run}-epoch-{h['epoch']}"]),
            }
            for h in history
        ],
        "selected_epoch": chosen,
        "selected_step": next(h["global_step"] for h in history if h["epoch"] == chosen),
        "selected_is_lowest_val_loss": best["epoch"] == chosen,
        "val_loss_rises_after_selected": any(
            h["val_loss"] > best["val_loss"] * 1.2 for h in history if h["epoch"] > chosen
        ),
    }

# ---------------- 2. 驗證集 → 測試集 ----------------


def split_metrics(split, key, weights=None):
    t, s = targets[split], scores[split][key]
    at_thr = threshold_metrics(t, s, thr[key], weights)
    at_half = threshold_metrics(t, s, 0.5, weights)
    return {
        "log_loss": binary_log_loss(t, s) if weights is None else weighted_log_loss(t, s, weights),
        "auroc": roc_auc(t, s, weights),
        "ap": average_precision(t, s, weights),
        "macro_f1_thr": at_thr["macro_f1"],
        "recall_thr": at_thr["system_recall"],
        "macro_f1_0.5": at_half["macro_f1"],
    }


generalization = {}
for key in CONFIGS:
    val, test = split_metrics("val", key), split_metrics("test", key)
    boot_val = [split_metrics("val", key, w) for w in RV]
    boot_test = [split_metrics("test", key, w) for w in RT]
    generalization[key] = {
        "threshold": thr[key],
        "val": val | {"ci": {m: ci(b[m] for b in boot_val) for m in val}},
        "test": test | {"ci": {m: ci(b[m] for b in boot_test) for m in test}},
        "test_minus_val": {
            m: {"diff": test[m] - val[m], "ci": ci(bt[m] - bv[m] for bv, bt in zip(boot_val, boot_test, strict=False))}
            for m in val
        },
    }

# ---------------- 3. 與訓練資料的相似度 ----------------
texts = {s: [r[TEXT_COLUMN] for r in datasets[s]] for s in datasets}
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), sublinear_tf=True)
vectorizer.fit(texts["train"] + texts["val"] + texts["test"])  # 與 idt_inputs 相同：全部描述一起算 IDF，不用標籤
train_matrix = vectorizer.transform(texts["train"])
nearest = np.asarray((vectorizer.transform(texts["test"]) @ train_matrix.T).max(axis=1).todense()).ravel()
# LoRA-S 的訓練資料另有改寫：一併算與「原始＋改寫」最像的那一筆
augmented = io.load_json(ROOT / "data/features/train_features.json")
augmented_matrix = vectorizer.transform([r[TEXT_COLUMN] for r in augmented])
nearest_augmented = np.asarray((vectorizer.transform(texts["test"]) @ augmented_matrix.T).max(axis=1).todense()).ravel()
T = targets["test"]
SIM_MODELS = ("v1-epoch-1", "v2-epoch-2", "tfidf", "v1-zero-shot")


def subset_auroc(indices, key, weights=None):
    w = None if weights is None else [weights[i] for i in indices]
    return roc_auc([T[i] for i in indices], [scores["test"][key][i] for i in indices], w)


def subset_summary(indices):
    """子集合的 AUROC 與區間（沿用整個測試集的重抽樣，子集合只剩單一類別的那次略過），以及配對差。"""
    indices = [int(i) for i in indices]
    labels = [T[i] for i in indices]
    usable = [w for w in RT if 0 < sum(w[i] for i in indices if T[i]) < sum(w[i] for i in indices)]
    boot = {key: [subset_auroc(indices, key, w) for w in usable] for key in SIM_MODELS}
    point = {key: subset_auroc(indices, key) for key in SIM_MODELS}
    out = {
        "records": len(indices),
        "system": sum(labels),
        "system_rate": sum(labels) / len(indices),
        "resamples": len(usable),
        "auroc": {key: {"value": point[key], "ci": ci(boot[key])} for key in SIM_MODELS},
        "paired": {},
    }
    for a, b in (("v1-epoch-1", "v1-zero-shot"), ("v2-epoch-2", "v1-zero-shot"), ("v1-epoch-1", "tfidf"), ("v2-epoch-2", "tfidf")):
        out["paired"][f"{a}_minus_{b}"] = {
            "diff": point[a] - point[b],
            "ci": ci(x - y for x, y in zip(boot[a], boot[b], strict=True)),
        }
    return out


median = float(np.median(nearest))
everything = np.arange(len(T))
similarity = {
    "measure": "與訓練集（原始 2,302 筆）最相近通報的 cosine 相似度；字元 2–3 gram、sublinear TF、IDF 以全部描述計算",
    "nearest_train": {
        "median": median,
        "q1": float(np.percentile(nearest, 25)),
        "q3": float(np.percentile(nearest, 75)),
        "max": float(nearest.max()),
        **{f"ge_{level:.1f}": int((nearest >= level).sum()) for level in (0.9, 0.8, 0.6, 0.5)},
        "ge_0.8_system": int(sum(T[i] for i in np.where(nearest >= NEAR_DUPLICATE)[0])),
    },
    "nearest_train_with_paraphrases": {
        "median": float(np.median(nearest_augmented)),
        **{f"ge_{level:.1f}": int((nearest_augmented >= level).sum()) for level in (0.9, 0.8, 0.6, 0.5)},
    },
    "all": subset_summary(everything),
    "exclude_near_duplicates": {
        f"{level:.1f}": subset_summary(np.where(nearest < level)[0]) for level in (0.8, 0.6)
    },
    "exclude_near_duplicates_with_paraphrases": subset_summary(np.where(nearest_augmented < NEAR_DUPLICATE)[0]),
    "low_half": subset_summary(np.where(nearest <= median)[0]),
    "high_half": subset_summary(np.where(nearest > median)[0]),
}

# ---------------- 4. 校準 ----------------


def logit(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_logistic(x, y, w=None, slope_fixed=False):
    """y 對 x 的邏輯迴歸（不加正則化），回傳（截距, 斜率）；slope_fixed 時斜率固定為 1，只估截距。"""
    w = np.ones_like(x) if w is None else w

    def loss(params):
        eta = params[0] + (1.0 if slope_fixed else params[1]) * x
        residual = w * (1 / (1 + np.exp(-eta)) - y)
        gradient = [residual.sum()] if slope_fixed else [residual.sum(), (residual * x).sum()]
        return float(np.sum(w * (np.logaddexp(0, eta) - y * eta))), np.array(gradient)

    start = np.array([0.0]) if slope_fixed else np.array([0.0, 1.0])
    result = minimize(loss, start, jac=True, method="BFGS")
    return (float(result.x[0]), 1.0) if slope_fixed else (float(result.x[0]), float(result.x[1]))


def ece(p, y, bins=ECE_BINS, equal_width=False):
    """期望校準誤差：各組「平均預測機率 − 實際比例」的絕對值，依組內筆數加權。預設依預測機率排序後等筆數分組。"""
    if equal_width:
        groups = [np.where((p >= lo) & ((p < hi) | (hi == 1)))[0] for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:], strict=True)]
    else:
        groups = np.array_split(np.argsort(p, kind="stable"), bins)
    return float(sum(len(g) / len(p) * abs(p[g].mean() - y[g].mean()) for g in groups if len(g)))


def calibration_summary(p, y, weights):
    x = logit(p)
    intercept, slope = fit_logistic(x, y)
    in_the_large, _ = fit_logistic(x, y, slope_fixed=True)
    slopes, larges, eces = [], [], []
    for w in weights:
        w = np.asarray(w, dtype=float)
        slopes.append(fit_logistic(x, y, w)[1])
        larges.append(fit_logistic(x, y, w, slope_fixed=True)[0])
        index = np.repeat(np.arange(len(p)), w.astype(int))
        eces.append(ece(p[index], y[index]))
    low = p < 0.05
    return {
        "slope": slope,
        "slope_ci": ci(slopes),
        "intercept": intercept,
        "calibration_in_the_large": in_the_large,
        "calibration_in_the_large_ci": ci(larges),
        "observed_over_expected": float(y.sum() / p.sum()),
        "mean_predicted": float(p.mean()),
        "prevalence": float(y.mean()),
        "ece": ece(p, y),
        "ece_ci": ci(eces),
        "ece_equal_width": ece(p, y, equal_width=True),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": binary_log_loss([int(v) for v in y], [float(v) for v in p]),
        "below_0.05": {"records": int(low.sum()), "mean_predicted": float(p[low].mean()), "observed": float(y[low].mean()), "system": int(y[low].sum())},
    }


calibration = {}
for key in CONFIGS:
    pv, pt = np.array(scores["val"][key]), np.array(scores["test"][key])
    yv, yt = np.array(targets["val"], dtype=float), np.array(targets["test"], dtype=float)
    item = {"val": calibration_summary(pv, yv, RV), "test": calibration_summary(pt, yt, RT)}
    # 再校準：截距與斜率只用驗證集估，套到測試集；單調轉換，不改變排序，也不改變驗證集調出的門檻下的判定
    a, b = fit_logistic(logit(pv), yv)
    assert b > 0, key
    item["recalibrated_on_val"] = {"intercept": a, "slope": b, "test": calibration_summary(1 / (1 + np.exp(-(a + b * logit(pt)))), yt, RT)}
    calibration[key] = item

result = {
    "selected": {"v1": selected["v1"], "v2": selected["v2"]},
    "resamples": {"val": len(RV), "test": len(RT), "seed_val": 43, "seed_test": 42},
    "records": {s: {"all": len(targets[s]), "system": sum(targets[s])} for s in ("val", "test")},
    "training": training,
    "generalization": generalization,
    "similarity": similarity,
    "calibration": calibration,
    "ece_bins": ECE_BINS,
    "near_duplicate_level": NEAR_DUPLICATE,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "idt_overfitting.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def d(item, digits=3):
    return f"{item['diff']:+.{digits}f} [{item['ci'][0]:+.{digits}f}, {item['ci'][1]:+.{digits}f}]"


print("== 訓練紀錄 ==")
for run, item in training.items():
    for h in item["history"]:
        mark = "  ← 選定" if h["selected"] else ""
        print(
            f"  {run} step {h['global_step']:5d}  train {h['train_loss']:.4f}  val {h['val_loss']:.4f}  "
            f"val log-loss {h['val_log_loss']:.4f}（{h['val_log_loss'] / h['val_loss']:.2f} 倍）  test log-loss {h['test_log_loss']:.4f}{mark}"
        )
    print(f"     選定的是驗證損失最低的：{item['selected_is_lowest_val_loss']}；之後驗證損失上升超過兩成：{item['val_loss_rises_after_selected']}")
print("== 驗證集 → 測試集（測試 − 驗證）==")
for key in CONFIGS:
    g = generalization[key]
    print(f"  {key:13s} " + "  ".join(f"{m} {g['val'][m]:.4f}→{g['test'][m]:.4f} {d(g['test_minus_val'][m])}" for m in ("log_loss", "auroc", "macro_f1_thr")))
print("== 相似度 ==")
print("  最相近訓練通報：", {k: round(v, 3) if isinstance(v, float) else v for k, v in similarity["nearest_train"].items()})
print("  含改寫：", {k: round(v, 3) if isinstance(v, float) else v for k, v in similarity["nearest_train_with_paraphrases"].items()})
for label, item in (
    ("全部", similarity["all"]),
    ("拿掉 ≥0.8", similarity["exclude_near_duplicates"]["0.8"]),
    ("拿掉 ≥0.6", similarity["exclude_near_duplicates"]["0.6"]),
    ("拿掉 ≥0.8（含改寫）", similarity["exclude_near_duplicates_with_paraphrases"]),
    ("相似度較低的一半", similarity["low_half"]),
    ("相似度較高的一半", similarity["high_half"]),
):
    print(f"  {label}：{item['records']} 筆、系統 {item['system']}（{item['resamples']} 次重抽樣）")
    print("     " + "  ".join(f"{k} {v['value']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" for k, v in item["auroc"].items()))
    print("     " + "  ".join(f"{k} {d(v)}" for k, v in item["paired"].items()))
print("== 校準（測試集）==")
for key in CONFIGS:
    c, r = calibration[key]["test"], calibration[key]["recalibrated_on_val"]["test"]
    print(
        f"  {key:13s} 斜率 {c['slope']:.3f} [{c['slope_ci'][0]:.3f}, {c['slope_ci'][1]:.3f}]  截距(斜率=1) {c['calibration_in_the_large']:+.3f}  "
        f"O/E {c['observed_over_expected']:.2f}  ECE {c['ece']:.4f} [{c['ece_ci'][0]:.4f}, {c['ece_ci'][1]:.4f}]（等寬 {c['ece_equal_width']:.4f}）  Brier {c['brier']:.4f}  "
        f"｜再校準後 斜率 {r['slope']:.3f} [{r['slope_ci'][0]:.3f}, {r['slope_ci'][1]:.3f}] log-loss {c['log_loss']:.4f}→{r['log_loss']:.4f} ECE {r['ece']:.4f}"
    )
print(OUT / "idt_overfitting.json")
