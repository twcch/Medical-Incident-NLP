"""v1、v2 與 TF-IDF 的比較：05 的輸出檔沒有的數字都在這裡算（實驗報告第 4 節用）。

讀 results/idt_scores（v1 與 zero-shot）與 results/idt_scores_v2（v2）的分數檔與評估檔，算出：
各操作點（門檻 0.5 與驗證集調出的 thr*）的指標與信賴區間、設定之間的配對差距、與上一輪 04 輸出的
比較、被壓到很低的「系統」筆數、在測試集上挑門檻的上限、各模型漏判案例的重疊、分數的相關與平均。
配對 bootstrap 與 05 用同一套重抽樣（random.Random(42)、2,000 次、單一類別的那次略過），分數也和 05
一樣先合併同分。自行重算的數字會和 05 的評估檔核對，不一致就中止。

在專案根目錄執行（不載模型，約 30 秒）：
    python -m analysis.idt_comparison                     # → results/analysis/idt_comparison.json
    python -m analysis.idt_comparison <輸出目錄>
"""

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

from utils import io
from utils.baseline import tfidf_scores
from utils.emotion import record_key
from utils.evaluation import (
    _percentile,
    average_precision,
    best_threshold,
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

datasets = {s: io.load_json(DATA / f"{s}_data.json") for s in ("train", "val", "test")}
targets = {s: [int(r[LABEL_COLUMN] == SYSTEM) for r in datasets[s]] for s in ("val", "test")}
scores = {"val": {}, "test": {}}
thr = {}
evals = {}
for run, out in RUNS.items():
    evals[run] = io.load_json(out / "idt_score_evaluation.json")
    for s in ("val", "test"):
        loaded = io.load_json(out / f"scores_{s}.json")
        assert [r["key"] for r in loaded["records"]] == [record_key(r) for r in datasets[s]], (run, s)
        for name in loaded["records"][0]["p_system"]:
            scores[s][f"{run}-{name}"] = [row["p_system"][name] for row in loaded["records"]]
    for name, config in evals[run]["configs"].items():
        thr[name if name == "tfidf" else f"{run}-{name}"] = config["threshold"]["value"]
tf = tfidf_scores(datasets["train"], {s: datasets[s] for s in ("val", "test")})
for s in ("val", "test"):
    scores[s]["tfidf"] = tf[s]
for key in list(scores["val"]):  # 與 05 相同：驗證集與測試集一起合併同分
    size = len(scores["val"][key])
    merged = merge_near_ties([*scores["val"][key], *scores["test"][key]])
    scores["val"][key], scores["test"][key] = merged[:size], merged[size:]

T, S = targets["test"], scores["test"]
keys_test = [record_key(r) for r in datasets["test"]]

# 上一輪 04 的實際輸出（v1 epoch-3、貪婪解碼）：硬標籤當成 0／1 分數、門檻 0.5
ref = {record_key(r): r for r in io.load_json(ROOT / "results/inference_predictions.json")}
S["04"] = [1.0 if ref[k]["idt_pred"] == SYSTEM else 0.0 for k in keys_test]
thr["04"] = 0.5
eval04 = io.load_json(ROOT / "results/inference_evaluation.json")["idt"]

CONFIGS = [
    ("v1-zero-shot", "v1", "zero-shot"),
    ("v1-epoch-1", "v1", "epoch-1"),
    ("v1-epoch-2", "v1", "epoch-2"),
    ("v1-epoch-3", "v1", "epoch-3"),
    ("v2-epoch-1", "v2", "epoch-1"),
    ("v2-epoch-2", "v2", "epoch-2"),
    ("tfidf", "v2", "tfidf"),
]


def counts(name, t):
    tp = sum(1 for p, y in zip(S[name], T, strict=True) if y and p >= t)
    fp = sum(1 for p, y in zip(S[name], T, strict=True) if not y and p >= t)
    return {"tp": tp, "fp": fp, "fn": sum(T) - tp, "tn": len(T) - sum(T) - fp}


def point_metrics(name, t):
    m = threshold_metrics(T, S[name], t)
    c = counts(name, t)
    return {
        "threshold": t,
        "accuracy": (c["tp"] + c["tn"]) / len(T),
        "macro_f1": m["macro_f1"],
        "system_precision": m["system_precision"],
        "system_recall": m["system_recall"],
        "system_f1": m["system_f1"],
        **c,
    }


# ---- 重抽樣（與 05 相同）----
rng = random.Random(42)
resamples = []
for _ in range(2000):
    w = [0] * len(T)
    for _ in range(len(T)):
        w[rng.randrange(len(T))] += 1
    if 0 < sum(wi for wi, t in zip(w, T, strict=True) if t) < len(T):
        resamples.append(w)

POINTS = {
    "04": ("04", 0.5),
    "v1-zero-shot@thr": ("v1-zero-shot", thr["v1-zero-shot"]),
    "v1-zero-shot@0.5": ("v1-zero-shot", 0.5),
    "v1-epoch-1@thr": ("v1-epoch-1", thr["v1-epoch-1"]),
    "v1-epoch-1@0.5": ("v1-epoch-1", 0.5),
    "v1-epoch-2@thr": ("v1-epoch-2", thr["v1-epoch-2"]),
    "v1-epoch-2@0.5": ("v1-epoch-2", 0.5),
    "v1-epoch-3@thr": ("v1-epoch-3", thr["v1-epoch-3"]),
    "v1-epoch-3@0.5": ("v1-epoch-3", 0.5),
    "v2-epoch-1@thr": ("v2-epoch-1", thr["v2-epoch-1"]),
    "v2-epoch-1@0.5": ("v2-epoch-1", 0.5),
    "v2-epoch-2@thr": ("v2-epoch-2", thr["v2-epoch-2"]),
    "v2-epoch-2@0.5": ("v2-epoch-2", 0.5),
    "tfidf@thr": ("tfidf", thr["tfidf"]),
    "tfidf@0.5": ("tfidf", 0.5),
}
TK = ("macro_f1", "system_recall", "system_precision", "system_f1")
samples = {p: defaultdict(list) for p in POINTS}
for w in resamples:
    ranking = {}
    for p, (name, t) in POINTS.items():
        m = threshold_metrics(T, S[name], t, w)
        for k in TK:
            samples[p][k].append(m[k])
        if name not in ranking:
            ranking[name] = (roc_auc(T, S[name], w), average_precision(T, S[name], w))
        samples[p]["auroc"].append(ranking[name][0])
        samples[p]["ap"].append(ranking[name][1])


def ci(values):
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


points = {}
for p, (name, t) in POINTS.items():
    pm = point_metrics(name, t)
    pm["auroc"] = None if name == "04" else roc_auc(T, S[name])
    pm["ap"] = None if name == "04" else average_precision(T, S[name])
    pm["ci"] = {k: ci(samples[p][k]) for k in (*TK, "auroc", "ap")}
    points[p] = pm


def paired(a, b, auroc_from=None):
    """a − b 的點估計與 95% 區間；auroc_from＝(a 的分數名, b 的分數名) 可指定 AUROC 用哪兩組分數。"""
    out = {}
    for k in ("macro_f1", "system_recall", "system_precision"):
        d = [x - y for x, y in zip(samples[a][k], samples[b][k], strict=True)]
        out[k] = {"diff": points[a][k] - points[b][k], "ci": ci(d)}
    pa, pb = auroc_from or (a, b)
    for k in ("auroc", "ap"):
        if points[pa][k] is None or points[pb][k] is None:
            out[k] = None
            continue
        d = [x - y for x, y in zip(samples[pa][k], samples[pb][k], strict=True)]
        out[k] = {"diff": points[pa][k] - points[pb][k], "ci": ci(d)}
    return out


comparisons = {
    "v2e2_thr_minus_04": paired("v2-epoch-2@thr", "04", ("v2-epoch-2@thr", "v1-epoch-3@0.5")),
    "v1e1_thr_minus_04": paired("v1-epoch-1@thr", "04", ("v1-epoch-1@thr", "v1-epoch-3@0.5")),
    "v2e2_05_minus_04": paired("v2-epoch-2@0.5", "04", ("v2-epoch-2@0.5", "v1-epoch-3@0.5")),
    "v2e2_thr_minus_v1e1_thr": paired("v2-epoch-2@thr", "v1-epoch-1@thr"),
    "v2e1_thr_minus_v1e1_thr": paired("v2-epoch-1@thr", "v1-epoch-1@thr"),
    "v2e2_05_minus_v1e1_05": paired("v2-epoch-2@0.5", "v1-epoch-1@0.5"),
    "v2e2_thr_minus_tfidf_05": paired("v2-epoch-2@thr", "tfidf@0.5"),
    "v2e2_thr_minus_tfidf_thr": paired("v2-epoch-2@thr", "tfidf@thr"),
    "v1e1_thr_minus_tfidf_05": paired("v1-epoch-1@thr", "tfidf@0.5"),
    "v1e3_05_minus_tfidf_05": paired("v1-epoch-3@0.5", "tfidf@0.5"),
    "v2e2_thr_minus_zs_thr": paired("v2-epoch-2@thr", "v1-zero-shot@thr"),
    "v1e1_thr_minus_zs_thr": paired("v1-epoch-1@thr", "v1-zero-shot@thr"),
    "zs_thr_minus_tfidf_05": paired("v1-zero-shot@thr", "tfidf@0.5"),
    "v1e1_thr_minus_tfidf_thr": paired("v1-epoch-1@thr", "tfidf@thr"),
    "zs_thr_minus_tfidf_thr": paired("v1-zero-shot@thr", "tfidf@thr"),
    "v2e2_thr_minus_v1e3_05": paired("v2-epoch-2@thr", "v1-epoch-3@0.5"),
    "v1e3_thr_minus_tfidf_05": paired("v1-epoch-3@thr", "tfidf@0.5"),
    "v1e3_thr_minus_tfidf_thr": paired("v1-epoch-3@thr", "tfidf@thr"),
}

# ---- 各設定的 val／test 排序指標（直接取 05 的輸出，另自行重算核對）----
config_rows = []
for key, run, name in CONFIGS:
    c = evals[run]["configs"][name]
    row = {
        "key": key,
        "val": c["val"],
        "test": c["test"],
        "threshold": c["threshold"],
        "test_at_0.5": {k: c["test_at_0.5"][k] for k in ("accuracy", "macro_f1", "confusion_matrix", "per_label")},
        "test_at_threshold": {
            k: c["test_at_threshold"][k] for k in ("accuracy", "macro_f1", "confusion_matrix", "per_label")
        },
        "ci": evals[run]["bootstrap"]["intervals"][name],
        "agreement_with_reference": c.get("agreement_with_reference"),
    }
    # 核對：自行重算與 05 輸出一致
    V = scores["val"][key]
    assert abs(binary_log_loss(targets["val"], V) - c["val"]["log_loss"]) < 6e-5, key
    assert abs(roc_auc(T, S[key]) - c["test"]["auroc"]) < 6e-5, key
    assert abs(average_precision(T, S[key]) - c["test"]["ap"]) < 6e-5, key
    mine = points[f"{key}@thr"]
    assert c["test_at_threshold"]["confusion_matrix"]["matrix"] == [[mine["tn"], mine["fp"]], [mine["fn"], mine["tp"]]], key
    for k in ("macro_f1", "auroc"):
        lo, hi = mine["ci"][k]
        assert abs(lo - row["ci"][k][0]) < 6e-5 and abs(hi - row["ci"][k][1]) < 6e-5, (key, k, lo, hi, row["ci"][k])
    config_rows.append(row)
assert eval04["confusion_matrix"]["matrix"] == [[points["04"]["tn"], points["04"]["fp"]], [points["04"]["fn"], points["04"]["tp"]]]

# ---- 過度自信 ----
overconf = {}
for key, _, _ in CONFIGS:
    for s in ("val", "test"):
        pos = [p for p, t in zip(scores[s][key], targets[s], strict=True) if t]
        neg = [p for p, t in zip(scores[s][key], targets[s], strict=True) if not t]
        overconf[f"{key}/{s}"] = {
            "pos_n": len(pos),
            "pos_lt_0.01": sum(p < 0.01 for p in pos),
            "pos_lt_0.05": sum(p < 0.05 for p in pos),
            "pos_median": statistics.median(pos),
            "neg_ge_0.5": sum(p >= 0.5 for p in neg),
        }

# ---- 門檻 ----
ceiling = {}
for key, _, _ in CONFIGS:
    b = best_threshold(T, S[key])
    ceiling[key] = {"test_best_threshold": b["threshold"], "test_best_macro_f1": b["macro_f1"]}
sensitivity = []
for t in (0.03, 0.05, 0.08, thr["v2-epoch-2"], 0.15, 0.2, 0.3, 0.4, 0.5):
    row = {"threshold": t}
    for s in ("val", "test"):
        sc, tg = scores[s]["v2-epoch-2"], targets[s]
        m = threshold_metrics(tg, sc, t)
        row[s] = {
            "tp": sum(1 for p, y in zip(sc, tg, strict=True) if y and p >= t),
            "fp": sum(1 for p, y in zip(sc, tg, strict=True) if not y and p >= t),
            "macro_f1": m["macro_f1"],
            "recall": m["system_recall"],
            "precision": m["system_precision"],
        }
    sensitivity.append(row)

# ---- 錯誤重疊 ----
pos_idx = [i for i, t in enumerate(T) if t]
miss = {
    "04": {i for i in pos_idx if S["04"][i] < 0.5},
    "v1e1": {i for i in pos_idx if S["v1-epoch-1"][i] < thr["v1-epoch-1"]},
    "v2e2": {i for i in pos_idx if S["v2-epoch-2"][i] < thr["v2-epoch-2"]},
    "tfidf": {i for i in pos_idx if S["tfidf"][i] < 0.5},
}
all3 = miss["v1e1"] & miss["v2e2"] & miss["tfidf"]
old16 = miss["v1e1"] & miss["tfidf"]
overlap = {
    "missed": {k: len(v) for k, v in miss.items()},
    "v1e1_and_tfidf": len(old16),
    "old16_recovered_by_v2e2": len(old16 - miss["v2e2"]),
    "all3": len(all3),
    "v2e2_new_miss_vs_v1e1": len(miss["v2e2"] - miss["v1e1"]),
    "v2e2_new_hit_vs_v1e1": len(miss["v1e1"] - miss["v2e2"]),
    "v1e1_and_v2e2": len(miss["v1e1"] & miss["v2e2"]),
    "any_of_3_hit": 62 - len(all3),
    "04_and_tfidf": len(miss["04"] & miss["tfidf"]),
}
all3_cases = [
    {
        "key": keys_test[i],
        "p_v1e1": S["v1-epoch-1"][i],
        "p_v2e2": S["v2-epoch-2"][i],
        "p_tfidf": S["tfidf"][i],
        "chars": len(datasets["test"][i][TEXT_COLUMN]),
        "description": datasets["test"][i][TEXT_COLUMN],
        "directive": datasets["test"][i].get("directive"),
    }
    for i in sorted(all3, key=lambda i: S["v2-epoch-2"][i])
]
texts3 = [c["description"] for c in all3_cases]
overlap["all3_distinct_texts"] = len(set(texts3))
# 驗證集的對應
VT, VS = targets["val"], scores["val"]
vpos = [i for i, t in enumerate(VT) if t]
vmiss = {
    "v1e1": {i for i in vpos if VS["v1-epoch-1"][i] < thr["v1-epoch-1"]},
    "v2e2": {i for i in vpos if VS["v2-epoch-2"][i] < thr["v2-epoch-2"]},
    "tfidf": {i for i in vpos if VS["tfidf"][i] < 0.5},
}
overlap["val"] = {**{k: len(v) for k, v in vmiss.items()}, "all3": len(vmiss["v1e1"] & vmiss["v2e2"] & vmiss["tfidf"])}

# ---- 相關與集成 ----
corr = {
    f"{a}|{b}": spearmanr(S[a], S[b]).statistic
    for a, b in (
        ("v2-epoch-2", "v1-epoch-1"),
        ("v2-epoch-2", "tfidf"),
        ("v1-epoch-1", "tfidf"),
        ("v2-epoch-1", "v2-epoch-2"),
    )
}


def avg(s, *names):
    return [sum(v) / len(v) for v in zip(*(scores[s][n] for n in names), strict=True)]


ens = {}
base = [roc_auc(T, S["v2-epoch-2"], w) for w in resamples]
for label, parts in (("v2e2+tfidf", ("v2-epoch-2", "tfidf")), ("v1e1+v2e2+tfidf", ("v1-epoch-1", "v2-epoch-2", "tfidf"))):
    test = avg("test", *parts)
    d = [roc_auc(T, test, w) - b for w, b in zip(resamples, base, strict=True)]
    ens[label] = {
        "val_auroc": roc_auc(VT, avg("val", *parts)),
        "test_auroc": roc_auc(T, test),
        "diff_vs_v2e2": roc_auc(T, test) - roc_auc(T, S["v2-epoch-2"]),
        "ci": ci(d),
    }

# ---- 當成篩檢工具：依 P(系統) 由高到低複核，前 k% 的通報涵蓋多少「系統」----
S["v2e2+tfidf"] = avg("test", "v2-epoch-2", "tfidf")


def gain(key, fraction, weights=None):
    """前 fraction 的通報（依分數由高到低）涵蓋的「系統」比例；同分的紀錄按比例計入。"""
    w = weights or [1] * len(T)
    budget = fraction * sum(w)
    positives = sum(wi for wi, t in zip(w, T, strict=True) if t)
    order = sorted(range(len(T)), key=S[key].__getitem__, reverse=True)
    taken = captured = 0.0
    i = 0
    while i < len(order) and taken < budget:
        j, group, group_positive = i, 0, 0
        while j < len(order) and S[key][order[j]] == S[key][order[i]]:
            group += w[order[j]]
            group_positive += w[order[j]] * T[order[j]]
            j += 1
        share = min(1.0, (budget - taken) / group) if group else 0.0
        captured += share * group_positive
        taken += share * group
        i = j
    return captured / positives


GAIN_KEYS = ("v2-epoch-2", "v1-epoch-1", "tfidf", "v1-zero-shot", "v2e2+tfidf")
gains = {}
for key in GAIN_KEYS:
    gains[key] = {}
    for fraction in (0.1, 0.2, 0.3, 0.5):
        value = gain(key, fraction)
        boot = [gain(key, fraction, w) for w in resamples] if key in ("v2-epoch-2", "tfidf") else None
        gains[key][f"{fraction:.2f}"] = {
            "reviewed": round(fraction * len(T)),
            "recall": value,
            "captured": value * sum(T),
            "precision": value * sum(T) / (fraction * len(T)),
            "ci": ci(boot) if boot else None,
        }
grid = [i / 100 for i in range(0, 101)]
gain_curve = {key: [[f, gain(key, f)] for f in grid] for key in ("v2-epoch-2", "tfidf", "v1-zero-shot")}
review_needed = {
    key: {f"{target:.2f}": next(f for f, r in gain_curve[key] if r >= target - 1e-9) for target in (0.8, 0.9)}
    for key in gain_curve
}

# ---- 其他 ----
by_key = {k: i for i, k in enumerate(keys_test)}
pair_762_888 = {
    k: {
        "target": datasets["test"][by_key[k]][LABEL_COLUMN],
        "p_v1e1": S["v1-epoch-1"][by_key[k]],
        "p_v1e3": S["v1-epoch-3"][by_key[k]],
        "p_v2e2": S["v2-epoch-2"][by_key[k]],
        "p_tfidf": S["tfidf"][by_key[k]],
        "pred04": ref[k]["idt_pred"],
    }
    for k in ("data.csv:762", "data.csv:888")
    if k in by_key
}
groups = defaultdict(list)
for s, recs in datasets.items():
    for r in recs:
        groups[r[TEXT_COLUMN].strip()].append((s, r[LABEL_COLUMN]))
dup = {t: v for t, v in groups.items() if len(v) > 1}
duplicates = {
    "groups": len(dup),
    "records": sum(len(v) for v in dup.values()),
    "cross_split_groups": sum(1 for v in dup.values() if len({s for s, _ in v}) > 1),
    "conflicting_label_groups": sum(1 for v in dup.values() if len({lab for _, lab in v}) > 1),
}

result = {
    "thresholds": thr,
    "resamples": len(resamples),
    "configs": config_rows,
    "eval04": eval04,
    "points": points,
    "comparisons": comparisons,
    "overconfidence": overconf,
    "ceiling": ceiling,
    "sensitivity_v2e2": sensitivity,
    "overlap": overlap,
    "all3_cases": all3_cases,
    "spearman": corr,
    "ensemble": ens,
    "pair_762_888": pair_762_888,
    "gain": gains,
    "gain_curve": gain_curve,
    "review_needed": review_needed,
    "system_rate_test": sum(T) / len(T),
    "tie_tolerance_logit": evals["v2"]["tie_tolerance_logit"],
    "distinct_scores_test": {key: len(set(S[key])) for key, _, _ in CONFIGS if key != "tfidf"},
    "duplicates": duplicates,
    "diagnostics": {run: evals[run]["diagnostics"] for run in RUNS},
    "selected": {run: evals[run]["selected"] for run in RUNS},
    "created": {run: evals[run]["created"] for run in RUNS},
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "idt_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def f(d):
    return f"{d['diff']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]" if d else "—"


print("== 操作點 ==")
for p, m in points.items():
    au = "  —   " if m["auroc"] is None else f"{m['auroc']:.4f}"
    print(
        f"  {p:18s} thr {m['threshold']:.4f} acc {m['accuracy']:.4f} F1 {m['macro_f1']:.4f} "
        f"[{m['ci']['macro_f1'][0]:.4f}, {m['ci']['macro_f1'][1]:.4f}] P {m['system_precision']:.3f} "
        f"R {m['system_recall']:.3f} [{m['ci']['system_recall'][0]:.3f}, {m['ci']['system_recall'][1]:.3f}] "
        f"TP {m['tp']} FP {m['fp']} AUROC {au}"
    )
print("== 配對 ==")
for k, v in comparisons.items():
    print(f"  {k}")
    for m in ("macro_f1", "system_recall", "system_precision", "auroc", "ap"):
        print(f"      {m:16s} {f(v[m])}")
print("== 過度自信（test）==")
for k, v in overconf.items():
    if k.endswith("/test"):
        print(f"  {k:22s} {v}")
print("== 門檻上限 ==", json.dumps(ceiling, ensure_ascii=False))
print("== 重疊 ==", json.dumps(overlap, ensure_ascii=False))
print("== 相關 ==", {k: round(v, 3) for k, v in corr.items()})
print("== 集成 ==", json.dumps(ens, ensure_ascii=False))
print("== 762/888 ==", json.dumps(pair_762_888, ensure_ascii=False))
print("== 重複 ==", duplicates)
print("== 篩檢效益（前 k% 涵蓋的系統比例）==")
for key in GAIN_KEYS:
    print(f"  {key:13s} " + "  ".join(f"{k}: {v['recall']:.3f}" + (f" [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" if v["ci"] else "") for k, v in gains[key].items()))
print("  要涵蓋 80%／90% 的系統需複核的比例：", review_needed)
print("== 三者都漏 ==")
for c in all3_cases:
    print(f"  {c['key']:14s} {c['p_v1e1']:.4f} {c['p_v2e2']:.4f} {c['p_tfidf']:.4f} {c['chars']:4d}字 {c['description'][:60]!r}")
