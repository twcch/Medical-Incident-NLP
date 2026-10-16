"""推論各階段的邏輯測試：重試、記 None 不猜、進度續跑、評審解析與評估指標。"""

import json

import pytest

from utils.evaluation import idt_metrics, judge_summary, label_distribution
from utils.inference import RetryConfig, judge_emotion, predict_emotion, predict_idt
from utils.progress import Progress
from utils.prompts import build_judge_messages, parse_judge_response


class FakeGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, messages, *, temperature, seed):
        self.prompts.append(messages[-1]["content"])
        assert self.responses, "沒有預備的回應了"
        return self.responses.pop(0)


def record(key="t:1", label="個人", description="病人跌倒", **extra):
    return {"source_id": key, "idt_target": label, "description": description, **extra}


def judge(p=4, e=3, w=5, reason="語氣帶有檢討"):
    return json.dumps(
        {"plausibility": p, "evidence": e, "writer_perspective": w, "reason": reason},
        ensure_ascii=False,
    )


# ---- ⑥ IDT 預測 ----


def test_predict_idt_retries_with_reason_then_accepts_label(tmp_path):
    config = RetryConfig(max_attempts=3)
    generator = FakeGenerator(["我覺得是個人", "系統。"])
    progress = Progress(tmp_path / "idt.jsonl", {"stage": "idt"})
    output, report = predict_idt([record()], generate=generator, config=config, progress=progress)
    assert output[0]["idt_pred"] == "系統" and report.calls == 2 and not report.failed
    assert (
        "上一次的輸出無效" in generator.prompts[1]
        and "請只輸出「個人」或「系統」" in generator.prompts[1]
    )
    assert report.counts == {"系統": 1}


def test_predict_idt_records_none_after_max_attempts_and_retries_on_rerun(tmp_path):
    config = RetryConfig(max_attempts=2)
    progress = Progress(tmp_path / "idt.jsonl", {"stage": "idt"})
    output, report = predict_idt(
        [record()], generate=FakeGenerator(["?", "??"]), config=config, progress=progress
    )
    assert output[0]["idt_pred"] is None
    assert report.failed == [{"key": "t:1", "reason": "「??」不是「個人」或「系統」"}]

    # 重跑：None 的那筆會再試，這次成功
    output, report = predict_idt(
        [record()], generate=FakeGenerator(["個人"]), config=config, progress=progress
    )
    assert output[0]["idt_pred"] == "個人" and report.reused == 0 and report.calls == 1

    # 再跑一次：成功的直接沿用，不呼叫模型
    output, report = predict_idt(
        [record()], generate=FakeGenerator([]), config=config, progress=progress
    )
    assert output[0]["idt_pred"] == "個人" and report.reused == 1 and report.calls == 0


# ---- ⑦ 情緒預測 ----


def test_predict_emotion_uses_training_prompt_and_label_set(tmp_path):
    progress = Progress(tmp_path / "emotion.jsonl", {"stage": "emotion"})
    generator = FakeGenerator(['{"emotion": "開心"}', '{"emotion": "自責"}'])
    output, report = predict_emotion(
        [record()], generate=generator, config=RetryConfig(), progress=progress
    )
    assert output[0]["emotion_pred"] == "自責" and report.calls == 2
    assert "撰寫這段事件描述的人" in generator.prompts[0]


# ---- ⑨ 評審 ----


def test_parse_judge_response_computes_overall_and_validates_range():
    result = parse_judge_response(judge(4, 3, 5))
    assert result["scores"] == {"plausibility": 4, "evidence": 3, "writer_perspective": 5}
    assert result["overall"] == 4.0 and result["reason"] == "語氣帶有檢討"
    assert (
        parse_judge_response('{"plausibility": "4", "evidence": 4, "writer_perspective": 4}')[
            "reason"
        ]
        == ""
    )
    with pytest.raises(ValueError, match="evidence"):
        parse_judge_response(judge(4, 6, 5))
    with pytest.raises(ValueError, match="plausibility"):
        parse_judge_response('{"evidence": 3, "writer_perspective": 3}')


def test_judge_prompt_sends_description_and_label_only():
    content = build_judge_messages("病人跌倒", "自責")[0]["content"]
    assert "病人跌倒" in content and "被評的情緒標籤：自責" in content
    assert "個人" not in content and "批示" not in content


def test_judge_emotion_skips_empty_labels_and_fingerprints_the_label(tmp_path):
    config = RetryConfig(max_attempts=1)
    progress = Progress(tmp_path / "judge.jsonl", {"stage": "judge"})
    records = [record("t:1", emotion_pred="自責"), record("t:2", emotion_pred=None)]
    output, report = judge_emotion(
        records, generate=FakeGenerator([judge()]), config=config, progress=progress
    )
    assert output[0]["emotion_judge"]["overall"] == 4.0 and output[1]["emotion_judge"] is None
    assert report.failed == [{"key": "t:2", "reason": "emotion_pred 為空，無法評審"}]
    assert report.counts == {"總分≈4": 1}

    # 同一筆換了被評的標籤，進度不能沿用，要重新評
    changed = [record("t:1", emotion_pred="焦慮")]
    output, report = judge_emotion(
        changed, generate=FakeGenerator([judge(2, 2, 2)]), config=config, progress=progress
    )
    assert output[0]["emotion_judge"]["overall"] == 2.0 and report.reused == 0


def test_judge_emotion_can_score_training_labels_via_emotion_field(tmp_path):
    progress = Progress(tmp_path / "judge.jsonl", {"stage": "judge"})
    records = [{"record_id": "a:1#aug1", **record("a:1", emotion_target="無奈")}]
    output, _ = judge_emotion(
        records,
        generate=FakeGenerator([judge(5, 5, 5)]),
        config=RetryConfig(),
        progress=progress,
        emotion_field="emotion_target",
        model_name="Qwen/Qwen3.5-9B",
    )
    assert output[0]["emotion_judge"]["overall"] == 5.0
    assert output[0]["emotion_judge"]["model"] == "Qwen/Qwen3.5-9B"
    assert judge_summary(output)["models"] == ["Qwen/Qwen3.5-9B"]


# ---- ⑧ 評估指標 ----


def test_idt_metrics_counts_invalid_as_wrong_and_reports_recall():
    records = [
        record("1", "個人", idt_pred="個人"),
        record("2", "個人", idt_pred="個人"),
        record("3", "個人", idt_pred="系統"),
        record("4", "系統", idt_pred="系統"),
        record("5", "系統", idt_pred=None),
    ]
    metrics = idt_metrics(records)
    assert metrics["records"] == 5 and metrics["correct"] == 3 and metrics["accuracy"] == 0.6
    assert metrics["invalid_predictions"] == 1
    assert metrics["confusion_matrix"]["matrix"] == [[2, 1], [0, 1]]
    assert metrics["confusion_matrix"]["invalid_by_label"] == {"個人": 0, "系統": 1}
    assert metrics["per_label"]["系統"] == {
        "support": 2,
        "predicted": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert metrics["per_label"]["個人"]["recall"] == round(2 / 3, 4)


def test_judge_summary_and_label_distribution():
    records = [
        record(
            "1",
            emotion_pred="自責",
            emotion_judge={
                "scores": {"plausibility": 4, "evidence": 2, "writer_perspective": 5},
                "overall": 3.667,
                "reason": "",
            },
        ),
        record(
            "2",
            emotion_pred="自責",
            emotion_judge={
                "scores": {"plausibility": 2, "evidence": 2, "writer_perspective": 1},
                "overall": 1.667,
                "reason": "",
            },
        ),
        record("3", emotion_pred=None, emotion_judge=None),
    ]
    summary = judge_summary(records)
    assert summary["scored"] == 2 and summary["unscored"] == 1
    assert summary["models"] == []
    assert summary["mean_scores"] == {
        "plausibility": 3.0,
        "evidence": 2.0,
        "writer_perspective": 3.0,
    }
    assert summary["mean_overall_score"] == round((3.667 + 1.667) / 2, 4)
    assert summary["overall_distribution"] == {2: 1, 4: 1}
    assert label_distribution(records, "emotion_pred") == {"counts": {"自責": 2}, "invalid": 1}
