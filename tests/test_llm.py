"""模型層的測試：LangChain chat model 的包裝、訊息轉換、設定驗證與雲端 provider 的建立。"""

import argparse
import importlib.util

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from utils.cli import add_model_arguments, model_config_from_args
from utils.llm import (
    PROVIDER_LOCAL,
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
