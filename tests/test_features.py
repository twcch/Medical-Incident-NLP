"""特徵工程的邏輯測試：用假的生成函式取代模型，驗證重試、驗證規則、進度續跑與中止。"""

import json

import pytest

from utils.augmentation import AugmentConfig, augment_records, derive_seed
from utils.emotion import EmotionConfig, EmotionLabelError, label_records
from utils.progress import Progress
from utils.prompts import EMOTION_LABELS, parse_emotion_response, parse_json_object


class FakeGenerator:
    """依序回傳預備好的輸出，並記錄每次收到的提示。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.seeds: list[int] = []

    def __call__(self, messages, *, temperature, seed):
        self.prompts.append(messages[-1]["content"])
        self.seeds.append(seed)
        assert self.responses, "沒有預備的回應了"
        return self.responses.pop(0)


def record(source_id="data.csv:2", description="病人跌倒", label="個人"):
    return {
        "source_id": source_id,
        "idt_target": label,
        "emotion_target": None,
        "description": description,
        "directive": "批示",
    }


def rewrite(text):
    return json.dumps({"description": text}, ensure_ascii=False)


# ---- 回應解析 ----


def test_parse_json_object_ignores_fences_and_trailing_text():
    text = '```json\n{"description": "改寫"}\n```\n以上是改寫結果。'
    assert parse_json_object(text) == {"description": "改寫"}


def test_parse_json_object_rejects_missing_or_broken_json():
    with pytest.raises(ValueError):
        parse_json_object("沒有物件")
    with pytest.raises(ValueError):
        parse_json_object('{"description": ')


def test_parse_emotion_response_normalizes_and_validates():
    assert parse_emotion_response('{"emotion": " 焦慮 "}') == "焦慮"
    with pytest.raises(ValueError):
        parse_emotion_response('{"emotion": "開心"}')
    assert len(EMOTION_LABELS) == 10


# ---- ③ 增生 ----


def test_augment_retries_and_tells_model_why(tmp_path):
    config = AugmentConfig(n=2, max_attempts=4)
    generator = FakeGenerator(
        ["不是 JSON", rewrite("病人跌倒"), rewrite("改寫一"), rewrite("改寫二")]
    )
    progress = Progress(tmp_path / "augment.jsonl", config.to_meta())

    output, report = augment_records(
        [record()], generate=generator, config=config, progress=progress, seed=42
    )

    assert [r["description"] for r in output] == ["病人跌倒", "改寫一", "改寫二"]
    assert [r["record_id"] for r in output] == ["data.csv:2", "data.csv:2#aug1", "data.csv:2#aug2"]
    assert [r["augmented"] for r in output] == [False, True, True]
    assert all(r["idt_target"] == "個人" and r["emotion_target"] is None for r in output)
    assert report.calls == 4 and report.rewrites_obtained == 2 and not report.shortfalls
    assert "上一次的輸出不可用（輸出中沒有 JSON 物件）" in generator.prompts[1]
    assert "與原文相同" in generator.prompts[2]
    assert len(set(generator.seeds)) == 4


def test_augment_rejects_duplicates_of_other_records(tmp_path):
    config = AugmentConfig(n=1, max_attempts=2)
    progress = Progress(tmp_path / "augment.jsonl", config.to_meta())
    records = [record("a:1", "第一筆"), record("a:2", "另一筆的原文")]
    # 第二筆的原文也在 taken 裡，所以第一筆的改寫不能和它一樣。
    generator = FakeGenerator(
        [rewrite("另一筆的原文"), rewrite("真正的改寫"), rewrite("第二筆改寫")]
    )

    output, report = augment_records(
        records, generate=generator, config=config, progress=progress, seed=1
    )

    assert "與既有的描述重複" in generator.prompts[1]
    assert [r["description"] for r in output if r["augmented"]] == ["真正的改寫", "第二筆改寫"]
    assert report.calls == 3 and not report.shortfalls


def test_augment_keep_partial_records_shortfall(tmp_path):
    config = AugmentConfig(n=1, max_attempts=2, on_failure="keep-partial")
    generator = FakeGenerator(["x", "y"])
    progress = Progress(tmp_path / "augment.jsonl", config.to_meta())

    output, report = augment_records(
        [record()], generate=generator, config=config, progress=progress, seed=0
    )
    assert len(output) == 1 and not output[0]["augmented"]
    assert report.shortfalls == {"data.csv:2": 1} and report.calls == 2


def test_augment_abort_raises_but_keeps_progress(tmp_path):
    config = AugmentConfig(n=1, max_attempts=1, on_failure="abort")
    generator = FakeGenerator(["x"])
    progress = Progress(tmp_path / "augment.jsonl", config.to_meta())
    with pytest.raises(RuntimeError, match="只取得 0/1"):
        augment_records([record()], generate=generator, config=config, progress=progress, seed=0)
    assert len(progress) == 1


def test_augment_resume_reuses_progress_without_calling_model(tmp_path):
    config = AugmentConfig(n=1, max_attempts=1)
    first = FakeGenerator([rewrite("改寫")])
    progress = Progress(tmp_path / "augment.jsonl", config.to_meta())
    output1, _ = augment_records(
        [record()], generate=first, config=config, progress=progress, seed=0
    )

    resumed = Progress(tmp_path / "augment.jsonl", config.to_meta())
    second = FakeGenerator([])
    output2, report = augment_records(
        [record()], generate=second, config=config, progress=resumed, seed=0
    )
    assert output2 == output1 and report.reused == 1 and report.calls == 0


def test_augment_regenerates_when_description_changed(tmp_path):
    config = AugmentConfig(n=1, max_attempts=1)
    progress = Progress(tmp_path / "augment.jsonl", config.to_meta())
    augment_records(
        [record()],
        generate=FakeGenerator([rewrite("改寫")]),
        config=config,
        progress=progress,
        seed=0,
    )
    changed = record(description="病人跌倒並撞到頭")
    generator = FakeGenerator([rewrite("新的改寫")])
    output, report = augment_records(
        [changed], generate=generator, config=config, progress=progress, seed=0
    )
    assert report.reused == 0 and output[1]["description"] == "新的改寫"


def test_augment_abort_mode_tops_up_short_entries(tmp_path):
    keep = AugmentConfig(n=2, max_attempts=1, on_failure="keep-partial")
    progress = Progress(tmp_path / "augment.jsonl", keep.to_meta())
    augment_records(
        [record()],
        generate=FakeGenerator([rewrite("改寫一"), "壞"]),
        config=keep,
        progress=progress,
        seed=0,
    )
    strict = AugmentConfig(n=2, max_attempts=1, on_failure="abort")
    generator = FakeGenerator([rewrite("改寫二")])
    output, report = augment_records(
        [record()], generate=generator, config=strict, progress=progress, seed=0
    )
    assert [r["description"] for r in output if r["augmented"]] == ["改寫一", "改寫二"]
    assert report.calls == 1


def test_derive_seed_is_stable_and_distinct():
    assert derive_seed(42, "a", 0, 0) == derive_seed(42, "a", 0, 0)
    assert derive_seed(42, "a", 0, 0) != derive_seed(42, "a", 0, 1)


# ---- 進度檔 ----


def test_progress_rejects_different_meta_unless_restart(tmp_path):
    path = tmp_path / "p.jsonl"
    Progress(path, {"stage": "augment", "n": 2}).add("k", "fp", rewrites=[])
    with pytest.raises(ValueError, match="--restart"):
        Progress(path, {"stage": "augment", "n": 3})
    fresh = Progress(path, {"stage": "augment", "n": 3}, restart=True)
    assert len(fresh) == 0


def test_progress_skips_half_written_last_line(tmp_path):
    path = tmp_path / "p.jsonl"
    progress = Progress(path, {"stage": "x"})
    progress.add("k1", "fp", value=1)
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"key": "k2", "fing')
    reloaded = Progress(path, {"stage": "x"})
    assert reloaded.get("k1", "fp") == {"key": "k1", "fingerprint": "fp", "value": 1}
    assert reloaded.get("k2", "fp") is None


# ---- ④ 情緒標註 ----


def test_emotion_retries_then_labels(tmp_path):
    config = EmotionConfig(max_attempts=3)
    generator = FakeGenerator(["哈", '{"emotion": "開心"}', '{"emotion": "焦慮"}'])
    progress = Progress(tmp_path / "emotion.jsonl", config.to_meta())
    output, report = label_records([record()], generate=generator, config=config, progress=progress)
    assert output[0]["emotion_target"] == "焦慮" and report.calls == 3
    assert "上一次的輸出無效（輸出中沒有 JSON 物件）" in generator.prompts[1]
    assert "不在十種情緒標籤內" in generator.prompts[2]
    assert report.label_counts == {"焦慮": 1}


def test_emotion_aborts_without_filling_neutral(tmp_path):
    config = EmotionConfig(max_attempts=2)
    generator = FakeGenerator(['{"emotion": "自責"}', "壞", "壞"])
    progress = Progress(tmp_path / "emotion.jsonl", config.to_meta())
    records = [record("a:1", "第一筆"), record("a:2", "第二筆")]
    with pytest.raises(EmotionLabelError, match="a:2：連續 2 次輸出無效"):
        label_records(records, generate=generator, config=config, progress=progress)
    assert len(progress) == 1  # 第一筆已保存，第二筆沒有被填成中性


def test_emotion_resume_uses_record_id_from_augmented_records(tmp_path):
    config = EmotionConfig()
    progress = Progress(tmp_path / "emotion.jsonl", config.to_meta())
    augmented = [
        {"record_id": "a:1", "augmented": False, **record("a:1", "原文")},
        {"record_id": "a:1#aug1", "augmented": True, **record("a:1", "改寫")},
    ]
    label_records(
        augmented,
        generate=FakeGenerator(['{"emotion": "中性"}', '{"emotion": "無奈"}']),
        config=config,
        progress=progress,
    )
    output, report = label_records(
        augmented, generate=FakeGenerator([]), config=config, progress=progress
    )
    assert [r["emotion_target"] for r in output] == ["中性", "無奈"] and report.reused == 2
