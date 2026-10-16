"""模型層的測試：LangChain chat model 的包裝、訊息轉換、設定驗證與雲端 provider 的建立。"""

import argparse
import importlib.util

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from utils.cli import (
    LOCAL_OPTIONS,
    PER_MODEL_LOCAL_OPTIONS,
    add_model_arguments,
    model_config_from_args,
)
from utils.llm import (
    DEFAULT_AUGMENT_MODEL,
    DEFAULT_EMOTION_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    PROVIDER_LOCAL,
    LocalChatModel,
    ModelConfig,
    build_chat_model,
    chat_generate,
    close_chat_model,
    message_text,
    to_dict_messages,
    to_langchain_messages,
)


def test_chat_generate_wraps_any_langchain_chat_model():
    chat = FakeListChatModel(responses=['  {"emotion": "焦慮"}  ', "第二次"])
    generate = chat_generate(chat)
    assert (
        generate([{"role": "user", "content": "hi"}], temperature=0.0, seed=1)
        == '{"emotion": "焦慮"}'
    )
    assert generate([{"role": "user", "content": "hi"}], temperature=0.8, seed=2) == "第二次"
    close_chat_model(chat)  # 非本機模型沒有東西要釋放，不應出錯


def test_message_conversion_round_trips_roles():
    dicts = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    messages = to_langchain_messages(dicts)
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage, AIMessage]
    assert to_dict_messages(messages) == dicts
    with pytest.raises(ValueError, match="角色"):
        to_langchain_messages([{"role": "tool", "content": "x"}])


def test_message_text_joins_content_blocks():
    message = AIMessage(content=[{"type": "text", "text": "個"}, {"type": "text", "text": "人"}])
    assert message_text(message) == "個人"


def test_model_config_defaults_to_local_and_meta_tracks_provider():
    local = ModelConfig(model="google/gemma-4-E2B-it")
    assert local.is_local and local.provider == PROVIDER_LOCAL
    assert local.to_meta() == {
        "provider": "huggingface",
        "model": "google/gemma-4-E2B-it",
        "max_new_tokens": 1024,
    }
    remote = ModelConfig(
        model="qwen3.5:9b",
        provider="openai",
        base_url="http://localhost:11434/v1",
        max_new_tokens=64,
    )
    assert not remote.is_local
    assert remote.to_meta() == {
        "provider": "openai",
        "model": "qwen3.5:9b",
        "max_new_tokens": 64,
        "base_url": "http://localhost:11434/v1",
    }
    for bad in ({"provider": ""}, {"model": " "}, {"base_url": ""}, {"load_in_4bit": 1}):
        with pytest.raises(ValueError):
            ModelConfig(**bad)


def test_build_chat_model_openai_uses_init_chat_model(monkeypatch):
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ModelConfig(
        model="gpt-4o-mini",
        provider="openai",
        max_new_tokens=64,
        base_url="http://localhost:11434/v1",
    )
    chat = build_chat_model(config, temperature=0.7)
    assert isinstance(chat, ChatOpenAI)
    assert chat.temperature == 0.7 and chat.max_tokens == 64
    assert chat.openai_api_base == "http://localhost:11434/v1"


def test_build_chat_model_reads_api_key_from_named_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MY_KEY", "abc")
    config = ModelConfig(model="gpt-4o-mini", provider="openai", api_key_env="MY_KEY")
    chat = build_chat_model(config, temperature=0.0)
    assert chat.openai_api_key.get_secret_value() == "abc"
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        build_chat_model(
            ModelConfig(model="x", provider="openai", api_key_env="MISSING_KEY"), temperature=0
        )


def test_build_chat_model_explains_missing_provider_package():
    if importlib.util.find_spec("langchain_anthropic"):
        pytest.skip("langchain-anthropic 已安裝，無法測缺套件的訊息")
    with pytest.raises(RuntimeError, match="pip install langchain-anthropic"):
        build_chat_model(
            ModelConfig(model="claude-sonnet-4-5", provider="anthropic"), temperature=0
        )


def test_cli_model_arguments_with_and_without_prefix():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser, title="模型", default_model="base", default_max_new_tokens=100)
    add_model_arguments(
        parser,
        title="評審",
        prefix="judge",
        default_model="judge-model",
        default_max_new_tokens=50,
        local_options=("dtype", "load_in_4bit"),
    )
    args = parser.parse_args(
        [
            "--provider",
            "openai",
            "--base-url",
            "http://x",
            "--judge-dtype",
            "float16",
            "--judge-load-in-4bit",
        ]
    )
    base = model_config_from_args(args)
    assert base.provider == "openai" and base.model == "base" and base.base_url == "http://x"
    assert base.max_new_tokens == 100 and base.device == "auto"
    judge = model_config_from_args(args, "judge", device="cuda")
    assert judge.is_local and judge.model == "judge-model" and judge.dtype == "float16"
    assert judge.load_in_4bit and judge.device == "cuda" and judge.max_new_tokens == 50


def test_cli_inherit_groups_fall_back_to_shared_config():
    """02 的用法：不帶前綴的共用選項是預設，--augment-*／--emotion-* 只覆寫有指定的選項。"""
    parser = argparse.ArgumentParser()
    add_model_arguments(
        parser,
        title="共用",
        default_model="shared",
        default_max_new_tokens=100,
        local_options=LOCAL_OPTIONS,
    )
    for prefix in ("augment", "emotion"):
        add_model_arguments(
            parser, title=prefix, prefix=prefix, inherit=True, local_options=PER_MODEL_LOCAL_OPTIONS
        )
    args = parser.parse_args(
        [
            "--dtype",
            "float16",
            "--load-in-4bit",
            "--emotion-provider",
            "openai",
            "--emotion-model",
            "gpt-4o-mini",
            "--emotion-max-new-tokens",
            "64",
        ]
    )
    shared = model_config_from_args(args)
    augment = model_config_from_args(args, "augment", base=shared)
    emotion = model_config_from_args(args, "emotion", base=shared)
    assert augment == shared
    assert augment.model == "shared" and augment.dtype == "float16" and augment.load_in_4bit
    assert emotion.provider == "openai" and emotion.model == "gpt-4o-mini"
    assert emotion.max_new_tokens == 64 and emotion.dtype == "float16"  # 沒指定的沿用共用
    assert shared.same_weights(augment) and not shared.same_weights(emotion)
    # 沒有共用設定時，inherit 群組沒指定的選項用 ModelConfig 的預設。
    assert model_config_from_args(parser.parse_args([]), "augment") == ModelConfig()


def test_same_weights_ignores_generation_settings_but_not_dtype_or_provider():
    base = ModelConfig(model="m", dtype="bfloat16", max_new_tokens=1024)
    assert base.same_weights(ModelConfig(model="m", dtype="bfloat16", max_new_tokens=64))
    assert base.same_weights(ModelConfig(model="m", dtype="bfloat16", local_files_only=True))
    assert not base.same_weights(ModelConfig(model="m", dtype="float16"))
    assert not base.same_weights(ModelConfig(model="m", dtype="bfloat16", load_in_4bit=True))
    assert not base.same_weights(ModelConfig(model="other", dtype="bfloat16"))
    assert not base.same_weights(ModelConfig(model="m", dtype="bfloat16", provider="openai"))


class _FakeHfModel:
    def eval(self):
        return self


class _FakeTokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 0  # 沒有 <|channel> 之類的 token


def test_build_chat_model_share_reuses_weights_only_for_same_model():
    first = LocalChatModel(
        settings=ModelConfig(model="base", max_new_tokens=1024),
        temperature=0.8,
        device="cpu",
        dtype="float32",
        hf_model=_FakeHfModel(),
        chat_template_source=_FakeTokenizer(),
    )
    second = build_chat_model(
        ModelConfig(model="base", max_new_tokens=64), temperature=0.0, share=first
    )
    assert isinstance(second, LocalChatModel) and second.hf_model is first.hf_model
    assert second.temperature == 0.0 and second.settings.max_new_tokens == 64
    assert second.device == "cpu" and second.dtype == "float32"
    with pytest.raises(ValueError, match="不是同一份權重"):
        build_chat_model(ModelConfig(model="other"), temperature=0.0, share=first)


def test_build_features_resolves_stage_models_from_cli_then_llm_defaults(monkeypatch):
    """02：--augment-model／--emotion-model > --model > llm.py 的 DEFAULT_AUGMENT_MODEL／DEFAULT_EMOTION_MODEL。"""
    features = importlib.import_module("scripts.02_build_features")
    parser = features.build_parser()
    augment, emotion = features.model_configs_from_args(parser.parse_args([]))
    assert (augment.model, emotion.model) == (DEFAULT_AUGMENT_MODEL, DEFAULT_EMOTION_MODEL)
    # 兩個常數各自接到對的階段：改 llm.py 的常數就能讓兩階段固定用不同模型。
    monkeypatch.setattr(features, "DEFAULT_AUGMENT_MODEL", "augment-default")
    monkeypatch.setattr(features, "DEFAULT_EMOTION_MODEL", "emotion-default")
    augment, emotion = features.model_configs_from_args(parser.parse_args([]))
    assert (augment.model, emotion.model) == ("augment-default", "emotion-default")
    augment, emotion = features.model_configs_from_args(parser.parse_args(["--model", "shared"]))
    assert augment.model == emotion.model == "shared"
    augment, emotion = features.model_configs_from_args(
        parser.parse_args(["--model", "shared", "--emotion-model", "other", "--load-in-4bit"])
    )
    assert augment.model == "shared" and emotion.model == "other"
    assert augment.load_in_4bit and emotion.load_in_4bit  # 沒指定的選項仍沿用共用設定


def test_inference_resolves_models_from_cli_then_llm_defaults(monkeypatch):
    """04：⑥ 用 --model（預設 DEFAULT_MODEL）；⑦ 依序 --emotion-model、--model、DEFAULT_EMOTION_MODEL。"""
    inference = importlib.import_module("scripts.04_inference")
    parser = inference.build_parser()
    monkeypatch.setattr(inference, "DEFAULT_EMOTION_MODEL", "emotion-default")
    idt, emotion, judge = inference.model_configs_from_args(parser.parse_args([]))
    assert idt.model == DEFAULT_MODEL and emotion.model == "emotion-default"
    assert judge.model == DEFAULT_JUDGE_MODEL and idt.is_local and emotion.is_local
    idt, emotion, _ = inference.model_configs_from_args(
        parser.parse_args(["--model", "base", "--device", "cpu"])
    )
    assert idt.model == emotion.model == "base" and emotion.device == "cpu"  # 試跑時兩邊一起換
    idt, emotion, _ = inference.model_configs_from_args(
        parser.parse_args(
            ["--model", "base", "--emotion-model", "other", "--emotion-provider", "openai"]
        )
    )
    assert idt.model == "base" and emotion.model == "other" and emotion.provider == "openai"
