"""純文字使用多模態模型，並以 context manager 控制模型記憶體生命週期。"""

import gc
import math
import re
from pathlib import Path

from src.config import RuntimeConfig
from src.json_io import parse_json


def parse_json_object(raw: str) -> dict:
    """接受 JSON 或單一 Markdown JSON 區塊；拒絕任意前後文與非物件。"""
    if not isinstance(raw, str):
        raise ValueError("模型輸出的 JSON 必須為字串。")
    value = raw.strip()
    if value.startswith("```"):
        # 必須完整匹配單一區塊，避免把額外說明文字默默忽略。
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", value, re.DOTALL)
        if not match:
            raise ValueError("模型輸出的 JSON 程式碼區塊不完整。")
        value = match.group(1).strip()
    result = parse_json(value)
    if not isinstance(result, dict):
        raise ValueError("模型輸出必須為 JSON 物件。")
    return result


def resolve_device(config: RuntimeConfig) -> str:
    """auto 依序選 CUDA、MPS、CPU；明確指定的 GPU 無法使用時直接回報。"""
    # 延後載入重型依賴，讓純資料處理或測試不必先初始化 Torch。
    import torch

    device = config.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求使用 CUDA，但此 Python 環境沒有可用的 CUDA GPU。")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("要求使用 MPS，但此 Python 環境沒有可用的 MPS 裝置。")
    return device


def resolve_dtype(config: RuntimeConfig, device: str):
    """保留指定精度；auto 依硬體支援選 bfloat16、float16 或 CPU 的 float32。"""
    import torch

    if config.dtype != "auto":
        return getattr(torch, config.dtype)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if device == "mps" else torch.float32


def load_model(model_id: str, config: RuntimeConfig, *, training: bool = False):
    """實際需要模型時才載入依賴與權重，回傳模型、文字處理器、裝置及精度。

    訓練置於單一裝置；推論允許 Accelerate 自動分配 CUDA 記憶體。
    """
    try:
        from transformers import AutoModelForMultimodalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "請安裝 requirements.txt；Gemma 4 需要 Transformers >= 5.10.1。"
        ) from exc

    device = resolve_device(config)
    dtype = resolve_dtype(config, device)
    set_seed(config.seed)
    options = {"local_files_only": config.local_files_only}
    # 本專案僅有文字；不初始化需要 torchvision／音訊依賴的多模態 processor。
    processor = TextProcessor(AutoTokenizer.from_pretrained(model_id, **options))
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map={"": device} if training or device != "cuda" else "auto",
        **options,
    )
    return model, processor, device, dtype


def release_memory() -> None:
    """先移除 Python 未引用物件，再清理可用 GPU 裝置的快取。"""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class TextProcessor:
    """保持訓練／推論的相同介面，所有文字都使用模型原生 chat template。"""

    def __init__(self, tokenizer):
        """包裝既有 tokenizer，提供本專案需要的文字處理介面。"""
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args, **kwargs):
        """沿用 tokenizer 原生模板與選項，維持訓練及推論的角色格式。"""
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """交由同一 tokenizer 將生成 token 還原為文字。"""
        return self.tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        """隨 adapter 保存 tokenizer，供後續重載使用相同文字格式。"""
        return self.tokenizer.save_pretrained(*args, **kwargs)


class ModelSession:
    """以 with 區塊載入模型，離開或載入失敗時釋放模型參照。"""

    def __init__(
        self,
        model_id: str,
        config: RuntimeConfig | None = None,
        adapter_path: str | Path | None = None,
    ):
        """僅保存載入設定；進入 with 區塊時才配置模型記憶體。"""
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id 必須為非空的模型 ID 或本機模型路徑。")
        self.model_id = model_id
        self.config = config or RuntimeConfig()
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.model = None
        self.processor = None

    def __enter__(self):
        """先驗證 adapter 所屬基礎模型，再載入權重並切換為推論模式。"""
        if self.model is not None:
            raise RuntimeError("同一個 ModelSession 不可重複進入 with 區塊。")
        try:
            if self.adapter_path is not None:
                # 在配置大型模型前檢查 adapter，避免錯誤路徑或模型不符浪費記憶體。
                adapter_config_path = self.adapter_path / "adapter_config.json"
                if not adapter_config_path.is_file():
                    raise FileNotFoundError(
                        f"找不到 IDT LoRA adapter：{adapter_config_path}；請先執行 train。"
                    )
                adapter_config = parse_json_object(adapter_config_path.read_text(encoding="utf-8"))
                expected = adapter_config.get("base_model_name_or_path")
                if expected and expected != self.model_id:
                    raise ValueError(
                        f"adapter 基礎模型為 {expected}，與指定的 {self.model_id} 不一致。"
                    )
                from peft import PeftModel
            self.model, self.processor, _, _ = load_model(self.model_id, self.config)
            if self.adapter_path is not None:
                self.model = PeftModel.from_pretrained(
                    self.model, str(self.adapter_path), is_trainable=False
                )
            self.model.eval()
            return self
        except BaseException:
            # KeyboardInterrupt 也要清理已載入模型，避免中斷後仍占用 GPU。
            self.close()
            raise

    def generate(
        self,
        system_prompt: str,
        text: str,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
    ) -> str:
        """只回傳本次生成內容；temperature=0 使用確定性的 greedy decoding。"""
        if self.model is None:
            raise RuntimeError("請在 with ModelSession(...) 區塊內呼叫 generate。")
        if not isinstance(system_prompt, str) or not isinstance(text, str):
            raise ValueError("system_prompt 與 text 必須為字串。")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature 必須為大於等於 0 的有限數值。")
        token_limit = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        # 只有 None 才採用預設；0 仍是無效輸入，不能藉由 truthiness 被偷偷替換。
        if isinstance(token_limit, bool) or not isinstance(token_limit, int) or token_limit < 1:
            raise ValueError("max_new_tokens 必須為大於 0 的整數。")

        import torch

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        generation_options = {
            "max_new_tokens": token_limit,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            # greedy decoding 不使用 temperature，避免向生成器傳遞無效的 0。
            generation_options["temperature"] = temperature
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_options)
        # 只解碼新生成的 token，不將輸入 prompt 交給標籤／JSON 解析器。
        response = self.processor.decode(output[0][input_length:], skip_special_tokens=True).strip()
        if not response:
            raise ValueError(f"{self.model_id} 未產生可用的文字。")
        return response

    def close(self) -> None:
        """移除所有模型參照後再清理快取，可重複呼叫。"""
        self.model = None
        self.processor = None
        release_memory()

    def __exit__(self, exc_type, exc_value, traceback):
        """不攔截區塊內的例外；先清理模型再讓例外向外傳遞。"""
        self.close()
