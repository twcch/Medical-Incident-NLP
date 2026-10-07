"""純文字使用多模態模型，並以 context manager 控制模型記憶體生命週期。"""

import gc
import json
import re
from pathlib import Path

from src.config import RuntimeConfig


def parse_json_object(raw: str) -> dict:
    """接受 JSON 或單一 Markdown JSON 區塊；拒絕任意前後文與非物件。"""
    value = raw.strip()
    if value.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", value, re.DOTALL)
        if not match:
            raise ValueError("模型輸出的 JSON 程式碼區塊不完整。")
        value = match.group(1).strip()
    result = json.loads(value, parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError(f"JSON 不允許 {x}")))
    if not isinstance(result, dict):
        raise ValueError("模型輸出必須為 JSON 物件。")
    return result


def resolve_device(config: RuntimeConfig) -> str:
    import torch

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求使用 CUDA，但此 Python 環境沒有可用的 CUDA GPU。")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("要求使用 MPS，但此 Python 環境沒有可用的 MPS 裝置。")
    return device


def resolve_dtype(config: RuntimeConfig, device: str):
    import torch

    if config.dtype != "auto":
        return getattr(torch, config.dtype)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if device == "mps" else torch.float32


def load_model(model_id: str, config: RuntimeConfig, *, training: bool = False):
    """訓練置於單一裝置；推論允許 Accelerate 自動分配 CUDA 記憶體。"""
    try:
        from transformers import AutoModelForMultimodalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise RuntimeError("請安裝 requirements.txt；Gemma 4 需要 Transformers >= 5.10.1。") from exc

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


def release_memory():
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
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.tokenizer.save_pretrained(*args, **kwargs)


class ModelSession:
    def __init__(self, model_id: str, config: RuntimeConfig | None = None,
                 adapter_path: str | Path | None = None):
        self.model_id = model_id
        self.config = config or RuntimeConfig()
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.model = None
        self.processor = None

    def __enter__(self):
        try:
            if self.adapter_path is not None:
                adapter_config_path = self.adapter_path / "adapter_config.json"
                if not adapter_config_path.is_file():
                    raise FileNotFoundError(f"找不到 IDT LoRA adapter：{adapter_config_path}；請先執行 train。")
                adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                expected = adapter_config.get("base_model_name_or_path")
                if expected and expected != self.model_id:
                    raise ValueError(f"adapter 基礎模型為 {expected}，與指定的 {self.model_id} 不一致。")
                from peft import PeftModel
            self.model, self.processor, _, _ = load_model(self.model_id, self.config)
            if self.adapter_path is not None:
                self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path), is_trainable=False)
            self.model.eval()
            return self
        except Exception:
            self.close()
            raise

    def generate(self, system_prompt: str, text: str, temperature: float = 0.0,
                 max_new_tokens: int | None = None) -> str:
        import torch

        if self.model is None:
            raise RuntimeError("請在 with ModelSession(...) 區塊內呼叫 generate。")
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        generation_options = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_options["temperature"] = temperature
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_options)
        # 不把 prompt 或模型內部思考回傳給標籤／JSON 解析器。
        response = self.processor.decode(output[0][input_length:], skip_special_tokens=True).strip()
        if not response:
            raise ValueError(f"{self.model_id} 未產生可用的文字。")
        return response

    def close(self):
        self.model = None
        self.processor = None
        release_memory()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
