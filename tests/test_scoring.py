"""05 IDT 機率評分的測試：AUROC／AP／門檻、P(系統)、進度續跑、adapter 檢查與基準；不載入模型。"""

import hashlib
import importlib
import json
import math
from types import SimpleNamespace

import pytest

from utils.baseline import tfidf_scores
from utils.evaluation import (
    average_precision,
    best_threshold,
    binary_log_loss,
    bootstrap_intervals,
    merge_near_ties,
    roc_auc,
    summarize_scores,
)
from utils.progress import Progress
from utils.prompts import IDT_PROMPT_ID, build_idt_messages
from utils.scoring import (
    adapter_metadata,
    check_adapter_compatible,
    discover_adapters,
    label_token_ids,
    score_record,
    score_records,
    two_way_probability,
)
from utils.training import build_example


class CharTokenizer:
    """每個字元一個 token：1 是回合開頭、2 是回合結尾；add_generation_prompt 只補開頭。"""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        ids = []
        for message in messages:
            ids += [1, *(ord(char) for char in message["content"]), 2]
        if add_generation_prompt:
            ids.append(1)
        return ids


class BrokenPrefixTokenizer(CharTokenizer):
    """產生提示時多塞一個 token，模擬提示不是完整序列前綴的 chat template。"""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        ids = super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
        return ids + [7] if add_generation_prompt else ids


class SameFirstTokenTokenizer(CharTokenizer):
    """所有中文字都變成同一個 token，「個人」與「系統」的第一個 token 就分不開。"""

    def apply_chat_template(self, messages, **kwargs):
        return [
            9 if token > 1000 else token
            for token in super().apply_chat_template(messages, **kwargs)
        ]


class FakeLogitsModel:
    """記下每次收到的輸入；最後一個位置的 logits 依 values 設定，其餘一律 -20。"""

    def __init__(self, values: dict[int, float], vocab: int = 40_000):
        self.values, self.vocab, self.calls = values, vocab, []

    def __call__(self, *, input_ids, attention_mask, **kwargs):
        import torch

        self.calls.append({"input_ids": input_ids.tolist(), **kwargs})
        logits = torch.full((1, input_ids.shape[1], self.vocab), -20.0)
        for token, value in self.values.items():
            logits[0, -1, token] = value
        return SimpleNamespace(logits=logits)


def record(key="a:1", label="個人", description="病人跌倒", **extra):
    return {"source_id": key, "idt_target": label, "description": description, **extra}


def make_adapter(directory, weights=b"w", base="base"):
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base}), encoding="utf-8"
    )
    (directory / "adapter_model.safetensors").write_bytes(weights)
    return directory


# ---- 不依賴門檻的指標 ----


def test_roc_auc_counts_ties_as_half_and_needs_both_classes():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert roc_auc([1, 1], [0.2, 0.9]) is None


def test_average_precision_steps_through_thresholds_and_groups_ties():
    assert average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(5 / 6)
    assert average_precision([1, 0], [0.5, 0.5]) == 0.5
    assert average_precision([0, 0], [0.1, 0.2]) is None


def test_metrics_accept_weights_as_bootstrap_multiplicities():
    targets, scores = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    expanded = ([0, 0, 0, 1, 1], [0.1, 0.4, 0.4, 0.35, 0.8])
    weights = [1, 2, 1, 1]
    assert roc_auc(targets, scores, weights) == roc_auc(*expanded)
    assert average_precision(targets, scores, weights) == pytest.approx(
        average_precision(*expanded)
    )


def test_binary_log_loss_averages_and_clips_extremes():
    expected = -(math.log(0.8) + math.log(0.9)) / 2
    assert binary_log_loss([1, 0], [0.8, 0.1]) == pytest.approx(expected)
    assert math.isfinite(binary_log_loss([1], [0.0]))


# ---- 門檻 ----


def test_best_threshold_maximizes_macro_f1():
    result = best_threshold([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.25, 0.9])
    assert result["threshold"] == pytest.approx(0.225)
    assert result["macro_f1"] == pytest.approx(0.8)


def test_best_threshold_prefers_the_one_nearest_half_on_ties():
    # 0.2 與 0.775 的 macro-F1 相同，取離 0.5 較近的 0.775
    assert best_threshold([0, 1, 0, 1], [0.1, 0.3, 0.6, 0.95])["threshold"] == pytest.approx(0.775)


def test_merge_near_ties_joins_scores_whose_logits_differ_less_than_the_tolerance():
    """同一個 logit 差算出的 P(系統) 只差浮點誤差（bf16 的 logit 是離散的），要視為同分。"""
    merged = merge_near_ties([0.2450851, 0.9, 0.2450850, 0.26, 0.0, 1.0])
    assert merged[0] == merged[2] == 0.2450850  # 同一群換成其中最小的分數
    assert merged[1] == 0.9 and merged[3] == 0.26  # 差得夠遠的分數不動
    assert merged[4] == 0.0 and merged[5] == 1.0  # 0 與 1 不會讓 logit 溢位


def test_merge_near_ties_keeps_order_and_length():
    scores = [0.7, 0.1, 0.1000000001, 0.4]
    merged = merge_near_ties(scores)
    assert len(merged) == 4 and merged[0] == 0.7 and merged[3] == 0.4
    assert merged[1] == merged[2] == 0.1


def test_best_threshold_keeps_half_when_validation_has_one_class():
    result = best_threshold([0, 0, 0], [0.1, 0.2, 0.3])
    assert result["threshold"] == 0.5 and result["macro_f1"] is None
    assert "單一類別" in result["note"]


# ---- bootstrap ----


def test_bootstrap_intervals_are_reproducible_and_paired_with_the_baseline():
    targets = [0] * 30 + [1] * 10
    scores = [0.1] * 25 + [0.7] * 5 + [0.8] * 8 + [0.2] * 2
    systems = {"lora": (scores, 0.5), "tfidf": (scores, 0.5)}
    first = bootstrap_intervals(targets, systems, n=200, seed=1, baseline="tfidf")
    assert first == bootstrap_intervals(targets, systems, n=200, seed=1, baseline="tfidf")
    low, high = first["lora"]["macro_f1"]
    assert 0 <= low <= high <= 1
    assert first["lora"]["auroc"][0] <= first["lora"]["auroc"][1]
    # 同一組重抽樣、同一組分數，與基準的差距恆為 0
    assert first["lora"]["vs_baseline"] == {"macro_f1": [0.0, 0.0], "auroc": [0.0, 0.0]}
    assert "vs_baseline" not in first["tfidf"]


# ---- 標籤 token 與 P(系統) ----


def test_label_token_ids_are_the_first_answer_tokens_used_in_training():
    tokenizer = CharTokenizer()
    ids = label_token_ids(tokenizer, "病人跌倒")
    assert ids == {"個人": ord("個"), "系統": ord("系")}
    example = build_example(tokenizer, record(label="系統"), max_length=4096)
    assert example.input_ids[-example.n_label_tokens] == ids["系統"]


def test_label_token_ids_reject_labels_sharing_the_first_token():
    with pytest.raises(ValueError, match="第一個 token"):
        label_token_ids(SameFirstTokenTokenizer(), "病人跌倒")


def test_label_token_ids_reject_template_whose_prompt_is_not_a_prefix():
    with pytest.raises(ValueError, match="前綴"):
        label_token_ids(BrokenPrefixTokenizer(), "病人跌倒")


def test_two_way_probability_normalizes_between_the_two_labels():
    assert two_way_probability(math.log(0.2), math.log(0.6)) == pytest.approx(0.75)
    assert two_way_probability(-3.0, -3.0) == 0.5
    assert two_way_probability(-1000.0, 0.0) == pytest.approx(1.0)
    assert two_way_probability(0.0, -1000.0) == pytest.approx(0.0)


def test_score_record_reads_last_position_logits_of_both_labels():
    pytest.importorskip("torch")
    tokenizer = CharTokenizer()
    ids = label_token_ids(tokenizer, "病人跌倒")
    model = FakeLogitsModel({ids["個人"]: 2.0, ids["系統"]: 3.0})
    result = score_record(model, tokenizer, "病人跌倒", ids, "cpu", logits_to_keep=True)
    assert result["p_system"] == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert result["top_is_label"] is True and 0.99 < result["label_mass"] <= 1
    assert result["logprob"]["系統"] - result["logprob"]["個人"] == pytest.approx(1.0)
    prompt = tokenizer.apply_chat_template(
        build_idt_messages("病人跌倒"), tokenize=True, add_generation_prompt=True
    )
    assert model.calls[0]["input_ids"] == [prompt] and model.calls[0]["logits_to_keep"] == 1


def test_score_record_flags_when_greedy_decoding_would_not_output_a_label():
    pytest.importorskip("torch")
    tokenizer = CharTokenizer()
    ids = label_token_ids(tokenizer, "病人跌倒")
    model = FakeLogitsModel({ids["個人"]: 2.0, ids["系統"]: 3.0, 5: 9.0})
    result = score_record(model, tokenizer, "病人跌倒", ids, "cpu", logits_to_keep=False)
    assert result["top_is_label"] is False and result["label_mass"] < 0.01
    assert "logits_to_keep" not in model.calls[0]


# ---- 逐筆評分與進度檔 ----


def test_score_records_reuses_progress_and_rescores_changed_descriptions(tmp_path):
    calls = []

    def fake_score(description):
        calls.append(description)
        return {"p_system": 0.9 if "系統" in description else 0.1}

    progress = Progress(tmp_path / "score.jsonl", {"stage": "idt_score"})
    records = [record("a:1", description="資訊系統當機"), record("a:2")]
    values, report = score_records(records, score=fake_score, progress=progress)
    assert [value["p_system"] for value in values] == [0.9, 0.1]
    assert report.scored == 2 and report.reused == 0

    values, report = score_records(records, score=fake_score, progress=progress)
    assert report.reused == 2 and report.scored == 0 and len(calls) == 2

    # 描述改了，同一個識別碼也要重評
    values, report = score_records(
        [record("a:1", description="病人自行拔管")], score=fake_score, progress=progress
    )
    assert report.scored == 1 and values[0]["p_system"] == 0.1


# ---- adapter ----


def test_discover_adapters_orders_epochs_and_skips_final_copy_of_a_checkpoint(tmp_path):
    root = make_adapter(tmp_path / "idt_adapter", b"e3")
    for epoch, weights in [(1, b"e1"), (2, b"e2"), (10, b"e10"), (3, b"e3")]:
        make_adapter(root / "checkpoints" / f"epoch-{epoch}", weights)
    labels = [label for label, _ in discover_adapters(root)]
    assert labels == ["epoch-1", "epoch-2", "epoch-3", "epoch-10"]
    (root / "adapter_model.safetensors").write_bytes(b"different")
    assert [label for label, _ in discover_adapters(root)][-1] == "final"


def test_discover_adapters_uses_the_adapter_itself_without_checkpoints(tmp_path):
    root = make_adapter(tmp_path / "idt_adapter")
    assert discover_adapters(root) == [("final", root)]
    assert discover_adapters(tmp_path / "missing") == []


def test_adapter_metadata_reads_training_metadata_two_levels_up_for_checkpoints(tmp_path):
    root = make_adapter(tmp_path / "idt_adapter")
    metadata = {"base_model": "base", "idt_prompt": IDT_PROMPT_ID}
    (root / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    info = adapter_metadata(make_adapter(root / "checkpoints" / "epoch-1", b"e1"))
    assert info["base_model"] == "base" and info["idt_prompt"] == IDT_PROMPT_ID
    assert info["sha256"] == hashlib.sha256(b"e1").hexdigest()
    with pytest.raises(ValueError, match="找不到 adapter"):
        adapter_metadata(tmp_path / "missing")


def test_check_adapter_compatible_rejects_other_base_model_or_prompt():
    info = {"path": "x", "base_model": "base", "idt_prompt": IDT_PROMPT_ID}
    check_adapter_compatible(info, "base")
    check_adapter_compatible({**info, "idt_prompt": None}, "base")  # 找不到 metadata 時只比基礎模型
    with pytest.raises(ValueError, match="基礎模型必須一致"):
        check_adapter_compatible(info, "other")
    with pytest.raises(ValueError, match="IDT 提示"):
        check_adapter_compatible({**info, "idt_prompt": "0000000000000000"}, "base")


# ---- 基準與彙整 ----


def test_tfidf_scores_follow_the_training_labels():
    pytest.importorskip("sklearn")
    train = [record(f"s:{i}", "系統", "資訊系統當機無法開立醫囑") for i in range(3)]
    train += [record(f"p:{i}", "個人", "病人自行下床跌倒") for i in range(6)]
    eval_sets = {"test": [record("x:1", description="電腦系統當機"), record("x:2")]}
    scores = tfidf_scores(train, eval_sets)
    assert scores["test"][0] > 0.5 > scores["test"][1]


def test_tfidf_scores_need_both_classes():
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="單一類別"):
        tfidf_scores([record("p:1"), record("p:2")], {"test": [record()]})


def test_summarize_scores_picks_lowest_val_log_loss_among_candidates():
    val_y, test_y = [0, 0, 0, 1], [0, 0, 1, 1]
    val = {
        "zero-shot": [0.01, 0.02, 0.03, 0.99],
        "epoch-1": [0.1, 0.2, 0.1, 0.8],
        "epoch-2": [0.4, 0.6, 0.3, 0.5],
    }
    test = {
        "zero-shot": [0.2, 0.3, 0.4, 0.6],
        "epoch-1": [0.1, 0.6, 0.7, 0.9],
        "epoch-2": [0.5, 0.5, 0.5, 0.5],
    }
    summary = summarize_scores(
        val_y,
        val,
        test_y,
        test,
        candidates=["epoch-1", "epoch-2"],
        bootstrap=0,
        seed=0,
        reference=["個人", "系統", "系統", None],
    )
    # zero-shot 的驗證集 log-loss 最低，但它是基準、不在候選內
    assert summary["selected"]["config"] == "epoch-1" and summary["bootstrap"] is None
    epoch1 = summary["configs"]["epoch-1"]
    assert epoch1["threshold"]["value"] == pytest.approx(0.5)
    assert epoch1["test_at_0.5"]["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]
    assert epoch1["test"]["auroc"] == 1.0 and epoch1["val"]["log_loss"] < 0.2
    assert epoch1["agreement_with_reference"] == {"matched": 3, "compared": 3}


def test_summarize_scores_bootstraps_every_config_against_the_baseline():
    val_y, test_y = [0] * 6 + [1] * 4, [0] * 12 + [1] * 8
    val = {"epoch-1": [0.1] * 5 + [0.6] + [0.7] * 4, "tfidf": [0.2] * 6 + [0.4] * 4}
    test = {"epoch-1": [0.1] * 10 + [0.6] * 2 + [0.7] * 8, "tfidf": [0.3] * 12 + [0.5] * 8}
    summary = summarize_scores(
        val_y, val, test_y, test, candidates=["epoch-1"], bootstrap=30, seed=3, baseline="tfidf"
    )
    intervals = summary["bootstrap"]["intervals"]
    assert set(intervals) == {"epoch-1", "tfidf"}
    assert set(intervals["epoch-1"]["vs_baseline"]) == {"macro_f1", "auroc"}
    assert summary["bootstrap"]["resamples"] == 30 and summary["bootstrap"]["baseline"] == "tfidf"


def test_summarize_scores_treats_scores_differing_by_float_noise_as_ties():
    """0.3 附近四筆只差浮點誤差（同一個 logit 差）。把它們當成不同分數，門檻會切在這群中間，
    驗證集看起來完全分得開；視為同分後，整群只能判在同一邊。"""
    val_y = [0, 0, 0, 0, 0, 1, 1, 1]
    val = {"epoch-1": [0.05, 0.05, 0.1, 0.3000000, 0.3000001, 0.3000002, 0.3000003, 0.9]}
    test_y = [0, 1, 0, 1]
    test = {"epoch-1": [0.3000000, 0.3000001, 0.05, 0.9]}
    summary = summarize_scores(val_y, val, test_y, test, candidates=["epoch-1"], bootstrap=0, seed=0)
    epoch1 = summary["configs"]["epoch-1"]
    assert epoch1["threshold"]["value"] == pytest.approx(0.2)  # 0.1 與 0.3 的中點，不在同分群裡
    assert epoch1["threshold"]["val_macro_f1"] == pytest.approx(0.75)
    assert epoch1["val"]["auroc"] == pytest.approx(13 / 15, abs=1e-4)  # 同分的配對各算一半
    assert epoch1["test_at_threshold"]["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]
    assert summary["tie_tolerance_logit"] == pytest.approx(1e-4)


# ---- 05 命令列 ----


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def split_files(tmp_path):
    val = [
        record("v:1"),
        record("v:2", "系統", "資訊系統當機"),
        record("v:3", description="病人自拔"),
    ]
    test = [
        record("t:1", description="病人下床跌倒"),
        record("t:2", "系統", "電腦系統當機"),
        record("t:3", "系統", "檢驗系統異常"),
        record("t:4", description="病人自行拔管"),
    ]
    train = [record(f"s:{i}", "系統", "資訊系統當機無法開立醫囑") for i in range(3)]
    train += [record(f"p:{i}", "個人", "病人自行下床跌倒") for i in range(6)]
    return tuple(
        write_json(tmp_path / f"{name}.json", data)
        for name, data in (("val", val), ("test", test), ("train", train))
    )


@pytest.fixture
def score_script():
    return importlib.import_module("scripts.05_score_idt")


def test_score_script_dry_run_lists_zero_shot_and_checkpoints(tmp_path, capsys, score_script):
    root = make_adapter(tmp_path / "idt_adapter", b"e2")
    write_json(root / "training_metadata.json", {"base_model": "base", "idt_prompt": IDT_PROMPT_ID})
    make_adapter(root / "checkpoints" / "epoch-1", b"e1")
    make_adapter(root / "checkpoints" / "epoch-2", b"e2")
    val, test, _ = split_files(tmp_path)
    args = ["--dry-run", "--model", "base", "--adapter-dir", str(root)]
    args += ["--val-data", str(val), "--test-data", str(test), "--out-dir", str(tmp_path / "out")]
    score_script.main(args)
    output = capsys.readouterr().out
    settings = json.loads(output[output.index("\n{") + 1 :])
    assert [config["name"] for config in settings["configs"]] == ["zero-shot", "epoch-1", "epoch-2"]
    assert settings["configs"][1]["adapter"]["idt_prompt"] == IDT_PROMPT_ID
    assert not (tmp_path / "out").exists()


def test_score_script_refuses_adapter_trained_with_another_prompt(tmp_path, score_script):
    root = make_adapter(tmp_path / "idt_adapter")
    write_json(root / "training_metadata.json", {"base_model": "base", "idt_prompt": "0" * 16})
    val, test, _ = split_files(tmp_path)
    args = ["--dry-run", "--model", "base", "--adapter-dir", str(root)]
    with pytest.raises(SystemExit, match="IDT 提示"):
        score_script.main(args + ["--val-data", str(val), "--test-data", str(test)])


def test_score_script_evaluate_stage_reads_score_files_and_writes_summary(tmp_path, score_script):
    pytest.importorskip("sklearn")
    val, test, train = split_files(tmp_path)
    out = tmp_path / "out"
    configs = [{"name": "zero-shot", "adapter": None}, {"name": "epoch-1", "adapter": None}]
    fake = {
        "val": {"zero-shot": [0.3, 0.6, 0.4], "epoch-1": [0.1, 0.8, 0.2]},
        "test": {"zero-shot": [0.4, 0.5, 0.7, 0.3], "epoch-1": [0.2, 0.9, 0.6, 0.1]},
    }
    for split, data in (("val", val), ("test", test)):
        values = {
            name: [{"p_system": p, "label_mass": 0.9, "top_is_label": True} for p in scores]
            for name, scores in fake[split].items()
        }
        records = json.loads(data.read_text(encoding="utf-8"))
        score_script.save_scores(out, split, records, values, configs, {"個人": 1, "系統": 2})
    args = ["--stage", "evaluate", "--val-data", str(val), "--test-data", str(test)]
    args += ["--train-data", str(train), "--out-dir", str(out), "--bootstrap", "20"]
    score_script.main(args + ["--reference", str(tmp_path / "missing.json")])
    evaluation = json.loads((out / "idt_score_evaluation.json").read_text(encoding="utf-8"))
    assert set(evaluation["configs"]) == {"zero-shot", "epoch-1", "tfidf"}
    assert evaluation["selected"]["config"] == "epoch-1"
    assert evaluation["bootstrap"]["baseline"] == "tfidf"
    assert evaluation["diagnostics"]["epoch-1"]["top_is_label_rate"] == 1.0


def test_score_script_evaluate_stage_needs_score_files(tmp_path, score_script):
    val, test, _ = split_files(tmp_path)
    args = [
        "--stage",
        "evaluate",
        "--no-baseline",
        "--val-data",
        str(val),
        "--test-data",
        str(test),
    ]
    with pytest.raises(SystemExit, match="找不到分數檔"):
        score_script.main(args + ["--out-dir", str(tmp_path / "out")])
