"""命令列共用：模型參數群組與對應的 ModelConfig，02 與 04 的各個模型都用同一組選項。

一支 script 可以有一組不帶前綴的共用選項（--model、--provider…），再加上帶前綴的各階段
選項（--emotion-model…）。帶前綴的群組用 inherit=True 建立時，每個選項都預設 None，表示
沒指定就沿用共用選項；由 model_config_from_args(args, prefix, base=共用設定) 補上。
"""

import argparse

from utils.llm import DEVICE_CHOICES, DTYPE_CHOICES, PROVIDER_LOCAL, ModelConfig

# 本機模型的選項：device 與 local_files_only 跟著機器走，dtype 與 load_in_4bit 跟著模型走。
LOCAL_OPTIONS = ("device", "dtype", "local_files_only", "load_in_4bit")
PER_MODEL_LOCAL_OPTIONS = ("dtype", "load_in_4bit")
MODEL_FIELDS = (
    "model",
    "provider",
    "base_url",
    "api_key_env",
    "max_new_tokens",
    "device",
    "dtype",
    "local_files_only",
    "load_in_4bit",
)


def add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    title: str,
    prefix: str = "",
    default_model: str | None = None,
    default_max_new_tokens: int | None = None,
    local_options: tuple[str, ...] = ("device", "dtype", "local_files_only"),
    inherit: bool = False,
    model_help: str | None = None,
) -> argparse._ArgumentGroup:
    """加入 --[prefix-]provider／model／base-url／api-key-env／max-new-tokens 與本機選項。

    inherit=True 時所有選項都預設 None（沒指定就沿用不帶前綴的共用選項），default_* 不使用。
    model_help 可換掉 --[prefix-]model 的說明，例如交代預設值來自哪個常數。
    """
    dash = f"--{prefix}-" if prefix else "--"

    def default(value):
        return None if inherit else value

    def fallback(option: str) -> str:
        return f"；預設沿用 --{option}" if inherit else ""

    group = parser.add_argument_group(title)
    group.add_argument(
        f"{dash}provider",
        default=default(PROVIDER_LOCAL),
        help="huggingface（本機 transformers）、openai、ollama、anthropic、google_genai…；"
        "OpenAI 相容服務用 openai 加 base-url"
        + (fallback("provider") if inherit else "；預設 huggingface"),
    )
    group.add_argument(
        f"{dash}model",
        default=default(default_model),
        help=model_help
        or "Hugging Face ID、本機目錄，或該 provider 上的模型名"
        + ("" if default(default_model) else "；預設沿用 --model"),
    )
    group.add_argument(
        f"{dash}base-url",
        help="雲端／自架服務的端點，例如 http://localhost:11434/v1" + fallback("base-url"),
    )
    group.add_argument(
        f"{dash}api-key-env",
        help="讀哪個環境變數當 API key；不給就用 provider 預設（如 OPENAI_API_KEY）"
        + fallback("api-key-env"),
    )
    group.add_argument(
        f"{dash}max-new-tokens",
        type=int,
        default=default(default_max_new_tokens),
        help="每次生成的 token 上限" + fallback("max-new-tokens"),
    )
    if "device" in local_options:
        group.add_argument(
            f"{dash}device",
            choices=DEVICE_CHOICES,
            default=default("auto"),
            help="本機模型" + fallback("device"),
        )
    if "dtype" in local_options:
        group.add_argument(
            f"{dash}dtype",
            choices=DTYPE_CHOICES,
            default=default("auto"),
            help="本機模型" + fallback("dtype"),
        )
    if "local_files_only" in local_options:
        group.add_argument(
            f"{dash}local-files-only",
            action="store_true",
            default=default(False),
            help="本機模型只用已下載的權重" + fallback("local-files-only"),
        )
    if "load_in_4bit" in local_options:
        group.add_argument(
            f"{dash}load-in-4bit",
            action="store_true",
            default=default(False),
            help="本機模型用 bitsandbytes nf4 載入（需 CUDA）" + fallback("load-in-4bit"),
        )
    return group


def first_given(*values):
    """回傳第一個不是 None 的值：命令列有指定的優先，最後才是 utils/llm.py 的預設常數。"""
    return next(value for value in values if value is not None)


def model_config_from_args(
    args: argparse.Namespace, prefix: str = "", *, base: ModelConfig | None = None, **overrides
) -> ModelConfig:
    """把 add_model_arguments 解析出來的值組成 ModelConfig。

    值是 None 的選項（inherit 群組沒指定的，或沒給的 base-url 等）沿用 base；沒有 base 就用
    ModelConfig 的預設。overrides 最後套上，可補群組裡沒有的欄位（例如共用的 device）。
    """
    key = f"{prefix}_" if prefix else ""
    values = base.to_dict() if base else {}
    for name in MODEL_FIELDS:
        value = getattr(args, key + name, None)
        if value is not None:
            values[name] = value
    values.update(overrides)
    return ModelConfig(**values)
