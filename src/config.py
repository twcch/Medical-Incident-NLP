"""Hugging Face 模型與 LoRA 訓練設定，不在 import 時下載模型。"""

import math
import re
from dataclasses import dataclass


def _require_positive_int(name: str, value: int) -> None:
    """bool 在 Python 也是 int，設定檢查需明確排除。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必須為大於 0 的整數。")


def _require_finite_number(name: str, value: float) -> None:
    """排除非數值與 NaN／Infinity，避免一般大小比較漏掉無效參數。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必須為有限數值。")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} 必須為有限數值。")


@dataclass(frozen=True)
class RuntimeConfig:
    """模型載入與生成共用設定；建構時即拒絕無法執行的參數。"""

    gemma_model: str = "google/gemma-4-E2B-it"
    judge_model: str = "Qwen/Qwen3.5-9B"
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 1024
    local_files_only: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        """先驗證使用者設定；裝置是否可用則留待模型載入時檢查。"""
        if not isinstance(self.device, str) or self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device 必須為 auto、cuda、mps 或 cpu。")
        if not isinstance(self.dtype, str) or self.dtype not in {
            "auto",
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ValueError("dtype 必須為 auto、float32、float16 或 bfloat16。")
        _require_positive_int("max_new_tokens", self.max_new_tokens)
        for name in ("gemma_model", "judge_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} 的模型 ID 或本機模型路徑必須為非空字串。")
        # 同一 seed 會交給 Python、NumPy 與 Torch，採用 NumPy 可接受的整數範圍。
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed 必須為介於 0 與 2**32 - 1 的整數。")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only 必須為 bool。")


@dataclass(frozen=True)
class TrainingConfig:
    """LoRA 訓練參數；整數計數與有限浮點數分別驗證。"""

    epochs: float = 3.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_length: int = 4096
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # 僅對文字模型加 adapter，避免同名的影像／音訊層被訓練。
    target_modules: str = r".*language_model\..*\.(q_proj|v_proj)"
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        """依參數用途驗證計數、連續數值及 target_modules 格式。"""
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_length",
            "lora_rank",
            "lora_alpha",
        ):
            _require_positive_int(name, getattr(self, name))
        for name in ("epochs", "learning_rate"):
            value = getattr(self, name)
            _require_finite_number(name, value)
            if value <= 0:
                raise ValueError(f"{name} 必須大於 0。")
        _require_finite_number("lora_dropout", self.lora_dropout)
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout 必須介於 0（含）與 1（不含）之間。")
        if not isinstance(self.target_modules, str) or not self.target_modules.strip():
            raise ValueError("target_modules 必須為非空的正規表示式字串。")
        # 只確認 regex 語法；是否匹配實際模型層仍由 PEFT 在建立 adapter 時確認。
        try:
            re.compile(self.target_modules)
        except re.error as exc:
            raise ValueError(f"target_modules 的正規表示式無效：{exc}") from exc
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing 必須為 bool。")
