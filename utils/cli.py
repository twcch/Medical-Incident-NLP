"""命令列共用：模型參數群組與對應的 ModelConfig，02 與 04 的各個模型都用同一組選項。"""

import argparse

from utils.llm import DEVICE_CHOICES, DTYPE_CHOICES, PROVIDER_LOCAL, ModelConfig

LOCAL_OPTIONS = ("device", "dtype", "local_files_only", "load_in_4bit")


def add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    title: str,
    prefix: str = "",
    default_model: str | None,
    default_max_new_tokens: int,
    local_options: tuple[str, ...] = ("device", "dtype", "local_files_only"),
) -> argparse._ArgumentGroup:
    """加入 --[prefix-]provider／model／base-url／api-key-env／max-new-tokens 與本機選項。"""
    dash = f"--{prefix}-" if prefix else "--"
    group = parser.add_argument_group(title)
    group.add_argument(
        f"{dash}provider",
        default=PROVIDER_LOCAL,
        help="huggingface（本機 transformers，預設）、openai、ollama、anthropic、google_genai…；"
        "OpenAI 相容服務用 openai 加 base-url",
    )
    group.add_argument(
        f"{dash}model",
        default=default_model,
        help="Hugging Face ID、本機目錄，或該 provider 上的模型名"
        + ("" if default_model else "；預設沿用 --model"),
    )
    group.add_argument(
        f"{dash}base-url", help="雲端／自架服務的端點，例如 http://localhost:11434/v1"
    )
    group.add_argument(
        f"{dash}api-key-env",
        help="讀哪個環境變數當 API key；不給就用 provider 預設（如 OPENAI_API_KEY）",
    )
    group.add_argument(
        f"{dash}max-new-tokens",
        type=int,
        default=default_max_new_tokens,
        help="每次生成的 token 上限",
    )
    if "device" in local_options:
        group.add_argument(f"{dash}device", choices=DEVICE_CHOICES, default="auto", help="本機模型")
    if "dtype" in local_options:
        group.add_argument(f"{dash}dtype", choices=DTYPE_CHOICES, default="auto", help="本機模型")
    if "local_files_only" in local_options:
        group.add_argument(
            f"{dash}local-files-only", action="store_true", help="本機模型只用已下載的權重"
        )
    if "load_in_4bit" in local_options:
        group.add_argument(
            f"{dash}load-in-4bit",
            action="store_true",
            help="本機模型用 bitsandbytes nf4 載入（需 CUDA）",
        )
    return group


def model_config_from_args(args: argparse.Namespace, prefix: str = "", **overrides) -> ModelConfig:
    """把 add_model_arguments 解析出來的值組成 ModelConfig；overrides 可補上群組裡沒有的欄位。"""
    key = f"{prefix}_" if prefix else ""
    values = {}
    for name in (
        "model",
        "provider",
        "base_url",
        "api_key_env",
        "max_new_tokens",
        "device",
        "dtype",
        "local_files_only",
        "load_in_4bit",
    ):
        if hasattr(args, key + name):
            values[name] = getattr(args, key + name)
    values.update(overrides)
    return ModelConfig(**values)
