"""訓練資料轉換、設定驗證與 IDT 回應解析的測試；用假的 tokenizer，不需要模型。"""

import pytest

from utils.prompts import parse_idt_response
from utils.training import (
    IGNORE_INDEX,
    TrainingConfig,
    _token_ids,
    build_example,
    build_examples,
    length_stats,
)


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


def record(label="個人", description="病人跌倒", key="a:1", **extra):
    return {"source_id": key, "idt_target": label, "description": description, **extra}


def test_build_example_masks_prompt_and_keeps_label_and_end_token():
    example = build_example(CharTokenizer(), record(), max_length=4096)
    assert example.n_label_tokens == 3  # 個、人、回合結尾
    assert example.input_ids[-3:] == [ord("個"), ord("人"), 2]
    assert example.labels[:-3] == [IGNORE_INDEX] * (len(example.input_ids) - 3)
    assert example.labels[-3:] == example.input_ids[-3:]
    assert example.key == "a:1" and example.label == "個人"


def test_build_example_prefers_record_id_over_source_id():
    example = build_example(CharTokenizer(), record(record_id="a:1#aug1"), max_length=4096)
    assert example.key == "a:1#aug1"


def test_build_example_rejects_invalid_label():
    with pytest.raises(ValueError, match="不是「個人」或「系統」"):
        build_example(CharTokenizer(), record(label="系统"), max_length=4096)


def test_build_example_rejects_template_whose_prompt_is_not_a_prefix():
    with pytest.raises(ValueError, match="前綴"):
        build_example(BrokenPrefixTokenizer(), record(), max_length=4096)


def test_build_examples_lists_every_too_long_record_without_truncating():
    tokenizer = CharTokenizer()
    short_length = len(build_example(tokenizer, record(), max_length=10_000).input_ids)
    records = [record(key="a:1"), record(key="a:2", description="很長" * 50), record(key="a:3")]
    with pytest.raises(ValueError, match="a:2") as info:
        build_examples(tokenizer, records, max_length=short_length + 10, name="訓練資料")
    assert "1 筆超過" in str(info.value) and "不截斷" in str(info.value)
    assert len(build_examples(tokenizer, records, max_length=10_000, name="訓練資料")) == 3


def test_token_ids_accepts_dict_tensor_like_and_nested_lists():
    class TensorLike:
        def tolist(self):
            return [[5, 6]]

    assert _token_ids({"input_ids": [[1, 2]]}) == [1, 2]
    assert _token_ids(TensorLike()) == [5, 6]
    assert _token_ids([3, 4]) == [3, 4]


def test_length_stats_reports_label_token_counts():
    examples = build_examples(
        CharTokenizer(), [record(), record(label="系統", key="a:2")], 4096, "x"
    )
    stats = length_stats(examples)
    assert stats["count"] == 2 and stats["label_tokens"] == [3]
    assert stats["min"] <= stats["mean"] <= stats["max"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"lora_dropout": 1.0},
        {"warmup_ratio": 1.0},
        {"learning_rate": 0.0},
        {"target_modules": "("},
        {"gradient_accumulation_steps": True},
    ],
)
def test_training_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_training_config_round_trips_to_dict():
    config = TrainingConfig(epochs=2, lora_rank=16)
    assert TrainingConfig(**config.to_dict()) == config


@pytest.mark.parametrize(
    "text,expected", [("個人", "個人"), (" 系統。", "系統"), ("個 人", "個人")]
)
def test_parse_idt_response_accepts_labels(text, expected):
    assert parse_idt_response(text) == expected


@pytest.mark.parametrize("text", ["個人，因為護理師疏忽", "", "系统", "不確定"])
def test_parse_idt_response_rejects_everything_else(text):
    with pytest.raises(ValueError):
        parse_idt_response(text)
