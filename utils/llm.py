"""模型層：所有呼叫模型的步驟都透過 LangChain 的 BaseChatModel 介面（invoke）。

- provider = "huggingface"（預設）：本機 transformers 模型，含 LoRA adapter 與 4-bit，由本模組的
  LocalChatModel 實作 BaseChatModel。不用 langchain-huggingface 的 ChatHuggingFace，因為它無法
  關閉 Gemma 4／Qwen 的思考模式、也不能掛 adapter、控制 seed 或 4-bit 載入。
- 其他 provider（openai、ollama、anthropic、google_genai…）：交給 langchain 的 init_chat_model，
  裝好對應的整合套件就能切換；OpenAI 相容的服務（Ollama、vLLM、LM Studio）用 provider=openai
  加 base_url。API key 從環境變數或 .env 讀。

③④⑦⑨ 的流程只依賴 generate(messages, *, temperature, seed) -> str 這個呼叫方式，
由 chat_generate() 把任何 BaseChatModel 包成這個函式；測試用任何函式取代即可。
torch 與 transformers 只在真正載入本機模型時才 import。
"""

import gc
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

# 基礎模型：03 訓練 LoRA 與 04 ⑥ IDT 推論一定用它（adapter 必須和基礎模型一致）。只用文字輸入輸出。
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
# 02 ③ 訓練集增生的預設模型，以及 02 ④ 訓練集情緒標註與 04 ⑦ 測試集情緒推論共用的預設模型
# （④⑦ 同一個模型、同一份提示），都不套 adapter。要固定用不同模型就改這兩行；執行時 --model
# 會同時蓋掉該 script 用到的所有預設，--augment-model／--emotion-model 只蓋自己那個階段。
DEFAULT_AUGMENT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_EMOTION_MODEL = "Qwen/Qwen3.5-2B"
# ⑨ 情緒評審模型；bf16 約 19 GB，12 GB 顯卡需 load_in_4bit（約 6 GB）或改用 Qwen/Qwen3.5-4B。
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.8-27B"
PROVIDER_LOCAL = "huggingface"
# 常見 provider 對應的整合套件；init_chat_model 支援的其他名稱也可以用。
PROVIDER_PACKAGES = {
    "openai": "langchain-openai",
    "azure_openai": "langchain-openai",
    "anthropic": "langchain-anthropic",
    "ollama": "langchain-ollama",
    "google_genai": "langchain-google-genai",
    "groq": "langchain-groq",
    "deepseek": "langchain-deepseek",
    "mistralai": "langchain-mistralai",
}
# 各 provider「生成 token 上限」的參數名；沒列的不傳，用該服務的預設。
MAX_TOKENS_PARAM = {
    "openai": "max_tokens",
    "azure_openai": "max_tokens",
    "anthropic": "max_tokens",
    "ollama": "num_predict",
    "google_genai": "max_output_tokens",
    "groq": "max_tokens",
    "deepseek": "max_tokens",
    "mistralai": "max_tokens",
    "xai": "max_tokens",
    "together": "max_tokens",
    "fireworks": "max_tokens",
    "openrouter": "max_tokens",
}
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")
DTYPE_CHOICES = ("auto", "float32", "float16", "bfloat16")

# 依序嘗試的模型類別：多模態（Gemma 4）→ 圖文 → 純文字（Qwen 等，用於小模型試跑）。
_MODEL_CLASSES = ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM")
# 推理模型的思考區塊；提示已關閉思考，保險起見輸出仍會剔除。
_THINKING_BLOCKS = re.compile(r"<\|channel>.*?<channel\|>|<think>.*?</think>", re.DOTALL)
_CHANNEL_TOKENS = ("<|channel>", "<channel|>")
_ROLE_TO_MESSAGE = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}
_TYPE_TO_ROLE = {"human": "user", "ai": "assistant", "system": "system"}

# 流程用的呼叫方式：generate(messages, *, temperature, seed) -> 模型輸出的文字
Generate = Callable[..., str]


@dataclass(frozen=True)
class ModelConfig:
    model: str = DEFAULT_MODEL
    provider: str = PROVIDER_LOCAL
    # 以下只對本機模型有意義
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 1024
    local_files_only: bool = False
    load_in_4bit: bool = False  # bitsandbytes nf4，只支援 CUDA；評審模型用
    # 以下只對雲端／API provider 有意義
    base_url: str | None = None  # OpenAI 相容服務（Ollama、vLLM、LM Studio）或自架端點
    api_key_env: str | None = None  # 讀哪個環境變數當 API key；None 用 provider 的預設變數

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model 必須是模型名稱（Hugging Face ID、本機目錄或服務上的模型名）。")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider 必須是非空字串，例如 huggingface、openai、ollama。")
        if self.device not in DEVICE_CHOICES:
            raise ValueError(f"device 必須是 {'、'.join(DEVICE_CHOICES)} 之一。")
        if self.dtype not in DTYPE_CHOICES:
            raise ValueError(f"dtype 必須是 {'、'.join(DTYPE_CHOICES)} 之一。")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens 必須是大於 0 的整數。")
        if not isinstance(self.load_in_4bit, bool):
            raise ValueError("load_in_4bit 必須是 bool。")
        for name in ("base_url", "api_key_env"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} 必須是非空字串或 None。")

    @property
    def is_local(self) -> bool:
        return self.provider == PROVIDER_LOCAL

    def to_dict(self) -> dict:
        return asdict(self)

    def to_meta(self) -> dict:
        """寫進進度檔的部分：換 provider、模型、端點或 token 上限就不能和舊進度混用。"""
        meta = {
            "provider": self.provider,
            "model": self.model,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.is_local and self.load_in_4bit:
            meta["load_in_4bit"] = True  # 量化會改變輸出；雲端 provider 沒有這回事
        if self.base_url:
            meta["base_url"] = self.base_url
        return meta

    def same_weights(self, other: "ModelConfig") -> bool:
        """兩個設定是否載入同一份本機權重；是的話可用 build_chat_model 的 share 共用，不必重複載入。

        只比對決定權重的欄位；max_new_tokens 等生成參數不同仍可共用。
        """
        if not (self.is_local and other.is_local):
            return False
        return all(
            getattr(self, name) == getattr(other, name)
            for name in ("model", "device", "dtype", "load_in_4bit")
        )


def resolve_device(choice: str) -> str:
    import torch

    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 --device cuda，但 CUDA 不可用。")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("指定 --device mps，但 MPS 不可用。")
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(choice: str, device: str):
    import torch

    if choice != "auto":
        return getattr(torch, choice)
    # Gemma 系列用 float16 容易溢位，GPU 一律優先 bfloat16；CPU 用 float32 最穩。
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def load_tokenizer(model_name: str, local_files_only: bool = False):
    """只送文字，用 tokenizer 套 chat template 就夠。

    Gemma 4 的多模態 processor 需要影像／音訊套件，缺了會 ImportError，
    所以 tokenizer 優先、processor 只是備援。
    """
    import transformers

    transformers.logging.set_verbosity_error()
    kwargs = {"local_files_only": local_files_only}
    try:
        return transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return transformers.AutoProcessor.from_pretrained(model_name, **kwargs)


def load_base_model(config: ModelConfig):
    """載入本機基礎模型並搬到裝置；回傳 (model, device, dtype 名稱)。

    load_in_4bit 時用 bitsandbytes nf4 量化（只支援 CUDA），由 device_map 直接放上 GPU，
    之後不能再 .to()。
    """
    import transformers

    transformers.logging.set_verbosity_error()
    device = resolve_device(config.device)
    torch_dtype = resolve_dtype(config.dtype, device)
    load_kwargs: dict = {"local_files_only": config.local_files_only, "dtype": torch_dtype}
    if config.load_in_4bit:
        if device != "cuda":
            raise RuntimeError(f"load_in_4bit 需要 CUDA 與 bitsandbytes，目前裝置是 {device}。")
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "load_in_4bit 需要安裝 bitsandbytes（pip install bitsandbytes）。"
            ) from exc
        load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        load_kwargs["device_map"] = {"": 0}
    errors = []
    for name in _MODEL_CLASSES:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(config.model, **load_kwargs)
        except ValueError as exc:
            # 這個 Auto 類別的對照表沒有該模型架構，換下一個類別再試。
            errors.append(f"{name}: {str(exc).splitlines()[0]}")
            continue
        if not config.load_in_4bit:
            model = model.to(device)
        dtype_name = "nf4" if config.load_in_4bit else str(torch_dtype).removeprefix("torch.")
        return model, device, dtype_name
    raise RuntimeError(f"無法載入模型 {config.model}：\n" + "\n".join(errors))


def release(model=None) -> None:
    """釋放模型與快取，讓之後的步驟可以載入另一個模型。"""
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def message_text(message: BaseMessage) -> str:
    """AIMessage.text 在 1.x 是屬性、0.3 是方法；內容是區塊清單時也只取文字。"""
    text = getattr(message, "text", None)
    if isinstance(
        text, str
    ):  # 1.x 的屬性回傳的是可呼叫的 str 子類別，先判斷 str 才不會走到棄用的方法
        return str(text)
    if callable(text):
        text = text()
    if isinstance(text, str):
        return text
    return message.content if isinstance(message.content, str) else str(message.content)


def to_langchain_messages(messages: list[dict]) -> list[BaseMessage]:
    """流程用的 {"role", "content"} 清單 → LangChain 訊息。"""
    result = []
    for message in messages:
        cls = _ROLE_TO_MESSAGE.get(message.get("role"))
        if cls is None:
            raise ValueError(f"不支援的訊息角色：{message.get('role')!r}")
        result.append(cls(content=message["content"]))
    return result


def to_dict_messages(messages: list[BaseMessage]) -> list[dict]:
    """LangChain 訊息 → chat template 用的 {"role", "content"} 清單。"""
    return [
        {"role": _TYPE_TO_ROLE.get(message.type, message.type), "content": message_text(message)}
        for message in messages
    ]


class LocalChatModel(BaseChatModel):
    """把本機 transformers 模型包成 LangChain 的 BaseChatModel。

    invoke 時可用 kwargs 覆寫 temperature 與 seed（seed 固定讓同一次嘗試可重現）；
    溫度 0 走 greedy，否則用模型 generation_config 的抽樣設定。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: ModelConfig
    temperature: float = 0.0
    device: str
    dtype: str
    hf_model: Any = Field(exclude=True, repr=False)
    chat_template_source: Any = Field(exclude=True, repr=False)  # tokenizer 或 processor
    _tokenizer: Any = PrivateAttr(default=None)
    _channel_ids: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        # 多模態 processor 內含 tokenizer；純文字模型時來源本身就是 tokenizer。
        self._tokenizer = getattr(self.chat_template_source, "tokenizer", self.chat_template_source)
        self.hf_model.eval()
        self._channel_ids = self._token_ids(*_CHANNEL_TOKENS)

    @property
    def _llm_type(self) -> str:
        return "local-transformers"

    @property
    def _identifying_params(self) -> dict:
        return {**self.settings.to_meta(), "temperature": self.temperature}

    def _token_ids(self, *tokens: str) -> tuple[int, ...] | None:
        """回傳 tokens 的 id；任何一個不在詞彙表就回傳 None。"""
        unk = getattr(self._tokenizer, "unk_token_id", None)
        ids = []
        for token in tokens:
            token_id = self._tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == unk:
                return None
            ids.append(token_id)
        return tuple(ids)

    def _strip_channel(self, tokens: list[int]) -> list[int]:
        """剔除 Gemma 4 的 <|channel>…<channel|> 思考區塊；沒有結尾就整段丟掉。"""
        if not self._channel_ids:
            return tokens
        open_id, close_id = self._channel_ids
        while open_id in tokens:
            start = tokens.index(open_id)
            end = tokens.index(close_id, start) if close_id in tokens[start:] else len(tokens) - 1
            tokens = tokens[:start] + tokens[end + 1 :]
        return tokens

    def generate_text(self, messages: list[dict], *, temperature: float, seed: int) -> str:
        import torch

        inputs = self.chat_template_source.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        generate_kwargs: dict = {"max_new_tokens": self.settings.max_new_tokens}
        if temperature > 0:
            generate_kwargs.update(do_sample=True, temperature=float(temperature))
            torch.manual_seed(seed)
            if self.device == "mps":
                torch.mps.manual_seed(seed)
        else:
            generate_kwargs.update(do_sample=False)
        with torch.inference_mode():
            output = self.hf_model.generate(**inputs, **generate_kwargs)
        new_tokens = self._strip_channel(output[0, input_len:].tolist())
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _THINKING_BLOCKS.sub("", text).strip()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = self.generate_text(
            to_dict_messages(messages),
            temperature=kwargs.get("temperature", self.temperature),
            seed=kwargs.get("seed", 0),
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def close(self) -> None:
        release(self.hf_model)
        self.hf_model = None


def _init_remote_chat_model(config: ModelConfig, temperature: float) -> BaseChatModel:
    """雲端／API provider 交給 langchain 的 init_chat_model；缺整合套件時提示要裝哪個。"""
    from langchain.chat_models import init_chat_model

    try:
        from dotenv import load_dotenv

        load_dotenv()  # API key 可放在專案根目錄的 .env（已在 .gitignore）
    except ImportError:
        pass
    kwargs: dict = {"temperature": temperature}
    param = MAX_TOKENS_PARAM.get(config.provider)
    if param:
        kwargs[param] = config.max_new_tokens
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"環境變數 {config.api_key_env} 沒有設定 API key。")
        kwargs["api_key"] = api_key
    try:
        return init_chat_model(config.model, model_provider=config.provider, **kwargs)
    except ImportError as exc:
        package = PROVIDER_PACKAGES.get(
            config.provider, f"langchain-{config.provider.replace('_', '-')}"
        )
        raise RuntimeError(
            f"provider「{config.provider}」需要安裝 {package}：pip install {package}"
        ) from exc


def build_chat_model(
    config: ModelConfig,
    *,
    temperature: float,
    share: BaseChatModel | None = None,
    hf_model=None,
    tokenizer=None,
    device: str | None = None,
    dtype: str | None = None,
) -> BaseChatModel:
    """依 provider 建立 LangChain chat model。

    本機模型可傳入已載入的 hf_model（例如套了 adapter 的 PEFT 模型），或用 share 共用另一個
    LocalChatModel 已載入的權重（同一模型、不同溫度），避免重複載入。
    """
    if not config.is_local:
        return _init_remote_chat_model(config, temperature)
    if isinstance(share, LocalChatModel):
        if not share.settings.same_weights(config):
            raise ValueError(
                f"share 的模型是 {share.settings.provider}:{share.settings.model}"
                f"（dtype {share.settings.dtype}、4-bit {share.settings.load_in_4bit}），"
                f"與要建立的 {config.provider}:{config.model}（dtype {config.dtype}、"
                f"4-bit {config.load_in_4bit}）不是同一份權重，不能共用。"
            )
        hf_model, tokenizer = share.hf_model, share.chat_template_source
        device, dtype = share.device, share.dtype
    if hf_model is None:
        hf_model, device, dtype = load_base_model(config)
    if tokenizer is None:
        tokenizer = load_tokenizer(config.model, config.local_files_only)
    return LocalChatModel(
        settings=config,
        temperature=temperature,
        device=device or resolve_device(config.device),
        dtype=dtype or config.dtype,
        hf_model=hf_model,
        chat_template_source=tokenizer,
    )


def chat_generate(chat_model: BaseChatModel) -> Generate:
    """把任何 BaseChatModel 包成流程用的 generate(messages, *, temperature, seed) -> str。

    本機模型每次呼叫都帶 temperature 與 seed；雲端 provider 用建立時的 temperature，
    seed 多數服務不支援，輸出不保證可重現。
    """
    local = isinstance(chat_model, LocalChatModel)

    def generate(messages: list[dict], *, temperature: float, seed: int) -> str:
        kwargs = {"temperature": temperature, "seed": seed} if local else {}
        response = chat_model.invoke(to_langchain_messages(messages), **kwargs)
        return message_text(response).strip()

    return generate


def close_chat_model(chat_model: BaseChatModel | None) -> None:
    """本機模型釋放權重與快取；雲端模型沒有東西要釋放。"""
    if isinstance(chat_model, LocalChatModel):
        chat_model.close()


def describe_chat_model(chat_model: BaseChatModel) -> str:
    if isinstance(chat_model, LocalChatModel):
        return f"{chat_model.settings.model}（本機，裝置 {chat_model.device}、精度 {chat_model.dtype}）"
    params = getattr(chat_model, "_identifying_params", {}) or {}
    name = params.get("model_name") or params.get("model") or type(chat_model).__name__
    return f"{name}（{type(chat_model).__name__}）"
