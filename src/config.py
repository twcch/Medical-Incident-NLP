"""Hugging Face 模型與 LoRA 訓練設定，不在 import 時下載模型。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    gemma_model: str = "google/gemma-4-E2B-it"
    judge_model: str = "Qwen/Qwen3.5-9B"
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 1024
    local_files_only: bool = False
    seed: int = 42

    def __post_init__(self):
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device 必須為 auto、cuda、mps 或 cpu。")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype 必須為 auto、float32、float16 或 bfloat16。")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必須大於 0。")
        if not self.gemma_model.strip() or not self.judge_model.strip():
            raise ValueError("模型 ID 或本機模型路徑不可為空。")


@dataclass(frozen=True)
class TrainingConfig:
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

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.gradient_accumulation_steps,
               self.learning_rate, self.max_length, self.lora_rank, self.lora_alpha) <= 0:
            raise ValueError("訓練步數、學習率、序列長度與 LoRA 參數必須大於 0。")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout 必須介於 0（含）與 1（不含）之間。")
