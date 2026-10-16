"""步驟 ⑤：IDT LoRA 訓練。

基礎模型加 LoRA（只掛在文字模型的 q_proj／v_proj），輸入只有事件描述，輸出「個人」或
「系統」；loss 只算 assistant 標籤與結束 token，情緒標註不參與訓練。每個 epoch 結束以
驗證集計算損失，只供觀察過擬合，不用來挑 checkpoint。超過 max_length 的紀錄直接報錯，
不截斷；標籤不是「個人」或「系統」也報錯，不用預設值填補。

每個 epoch 結束會存 adapter 與 optimizer 狀態到 checkpoints/epoch-N/，可從最後一個續跑。
每個例子單獨前向（micro-batch 1），以梯度累積湊成有效 batch；Gemma 的詞彙表有 26 萬，
只計算標籤位置的 logits 才不會讓顯存被整段序列的 logits 吃光。
"""

import inspect
import math
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.preprocessing import LABEL_COLUMN, TEXT_COLUMN, VALID_IDT_LABELS
from utils.prompts import build_idt_messages

# 多模態模型的文字層在 language_model 底下；純文字模型（試跑用）可改成 r".*\.(q_proj|v_proj)"。
DEFAULT_TARGET_MODULES = r".*language_model\..*\.(q_proj|v_proj)"
IGNORE_INDEX = -100
CHECKPOINT_DIR = "checkpoints"
STATE_FILE = "trainer_state.pt"


def _positive_int(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必須是大於 0 的整數。")


def _finite(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} 必須是有限數值。")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_length: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: str = DEFAULT_TARGET_MODULES
    gradient_checkpointing: bool = True
    log_every: int = 10

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "gradient_accumulation_steps",
            "max_length",
            "lora_rank",
            "lora_alpha",
            "log_every",
        ):
            _positive_int(name, getattr(self, name))
        for name in (
            "learning_rate",
            "warmup_ratio",
            "weight_decay",
            "max_grad_norm",
            "lora_dropout",
        ):
            _finite(name, getattr(self, name))
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate 與 max_grad_norm 必須大於 0。")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("warmup_ratio 與 lora_dropout 必須介於 0（含）與 1（不含）之間。")
        if self.weight_decay < 0:
            raise ValueError("weight_decay 不可為負數。")
        if not isinstance(self.target_modules, str) or not self.target_modules.strip():
            raise ValueError("target_modules 必須是非空的正規表示式。")
        try:
            re.compile(self.target_modules)
        except re.error as exc:
            raise ValueError(f"target_modules 的正規表示式無效：{exc}") from exc
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing 必須是 bool。")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Example:
    """一筆訓練／驗證例子：完整 token 序列，以及只在標籤位置有值的 labels。"""

    key: str
    label: str
    input_ids: list[int]
    labels: list[int]
    n_label_tokens: int


class ExampleTooLong(ValueError):
    def __init__(self, key: str, length: int, max_length: int):
        super().__init__(f"{key} 有 {length} 個 token，超過 max_length={max_length}")
        self.key, self.length = key, length


def _token_ids(result) -> list[int]:
    """apply_chat_template 依版本可能回傳 list、BatchEncoding 或 tensor，統一成 list[int]。"""
    if hasattr(result, "keys") and "input_ids" in result:
        result = result["input_ids"]
    if hasattr(result, "tolist"):
        result = result.tolist()
    if result and isinstance(result[0], list):
        result = result[0]
    return [int(token) for token in result]


def build_example(tokenizer, record: dict, max_length: int) -> Example:
    """把一筆紀錄變成「提示 + 標籤」序列；提示位置的 labels 是 IGNORE_INDEX，不算 loss。"""
    key = record.get("record_id") or record["source_id"]
    label = record[LABEL_COLUMN]
    if label not in VALID_IDT_LABELS:
        raise ValueError(f"{key} 的 idt_target「{label}」不是「個人」或「系統」。")
    messages = build_idt_messages(record[TEXT_COLUMN])
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": label}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if len(full_ids) > max_length:
        raise ExampleTooLong(key, len(full_ids), max_length)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{key}：chat template 的提示不是完整序列的前綴，無法標記 loss 範圍。")
    n_label_tokens = len(full_ids) - len(prompt_ids)
    if n_label_tokens < 1:
        raise ValueError(f"{key}：標籤沒有產生任何 token。")
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return Example(key, label, full_ids, labels, n_label_tokens)


def build_examples(tokenizer, records: list[dict], max_length: int, name: str) -> list[Example]:
    """整批轉換；超長的紀錄一次列出後報錯，不截斷。"""
    examples, too_long = [], []
    for record in records:
        try:
            examples.append(build_example(tokenizer, record, max_length))
        except ExampleTooLong as exc:
            too_long.append(f"{exc.key}（{exc.length}）")
    if too_long:
        shown = "、".join(too_long[:10]) + ("…" if len(too_long) > 10 else "")
        raise ValueError(
            f"{name} 有 {len(too_long)} 筆超過 max_length={max_length}，不截斷：{shown}。"
            "請調高 --max-length 或先處理這些紀錄。"
        )
    return examples


def length_stats(examples: list[Example]) -> dict:
    lengths = sorted(len(example.input_ids) for example in examples)
    if not lengths:
        return {"count": 0}
    return {
        "count": len(lengths),
        "min": lengths[0],
        "mean": round(sum(lengths) / len(lengths), 1),
        "p95": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))],
        "max": lengths[-1],
        "label_tokens": sorted({example.n_label_tokens for example in examples}),
    }


def matched_modules(model, pattern: str) -> list[str]:
    regex = re.compile(pattern)
    return [name for name, _ in model.named_modules() if name and regex.fullmatch(name)]


def prepare_for_training(model, gradient_checkpointing: bool):
    """關掉 KV cache、視需要開梯度檢查點；回傳同一個模型。"""
    config = getattr(model, "config", None)
    for item in (config, getattr(config, "text_config", None)):
        if item is not None and hasattr(item, "use_cache"):
            item.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def attach_lora(model, config: TrainingConfig):
    """依正規表示式挑模組掛 LoRA；一個都沒匹配就報錯並提示模型裡實際的模組名稱。"""
    from peft import LoraConfig, get_peft_model

    names = matched_modules(model, config.target_modules)
    if not names:
        hints = [name for name, _ in model.named_modules() if name.endswith("q_proj")][:3]
        raise ValueError(
            f"target_modules「{config.target_modules}」沒有匹配任何模組；"
            f"模型裡的 q_proj 例如：{hints}。純文字模型可用 --target-modules '.*\\.(q_proj|v_proj)'。"
        )
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora), names


def load_checkpoint_adapter(model, checkpoint: Path):
    """把 checkpoint 的 adapter 掛回基礎模型並設為可訓練，供續跑。"""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(checkpoint), is_trainable=True)


def supports_logits_to_keep(model) -> bool:
    """模型 forward 是否接受 logits_to_keep；接受時只算標籤位置的 logits，省下大量顯存。"""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return "logits_to_keep" in inspect.signature(base.forward).parameters


def label_loss(model, example: Example, device: str, *, logits_to_keep: bool):
    """回傳 (標籤 token 的 loss 總和, 標籤 token 數, 所有標籤 token 是否都是 argmax)。

    只取最後 n+1 個位置的 logits：位置 i 預測 token i+1，所以倒數第 n+1 到倒數第 2 個位置
    正好對應 n 個標籤 token。
    """
    import torch
    import torch.nn.functional as F

    input_ids = torch.tensor([example.input_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    keep = example.n_label_tokens + 1
    kwargs = {"logits_to_keep": keep} if logits_to_keep else {}
    logits = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits[0]
    predictions = logits[-keep:-1].float()
    targets = input_ids[0, -keep + 1 :]
    loss = F.cross_entropy(predictions, targets, reduction="sum")
    matched = bool((predictions.detach().argmax(-1) == targets).all())
    return loss, example.n_label_tokens, matched


def evaluate(model, examples: list[Example], device: str, *, logits_to_keep: bool) -> dict:
    """驗證集的平均標籤 token loss，以及整個標籤都答對（teacher forcing 下 argmax）的比例。"""
    import torch

    model.eval()
    loss_sum, tokens, matched = 0.0, 0, 0
    with torch.no_grad():
        for example in examples:
            loss, n_tokens, ok = label_loss(model, example, device, logits_to_keep=logits_to_keep)
            loss_sum += loss.item()
            tokens += n_tokens
            matched += ok
    model.train()
    return {
        "loss": loss_sum / max(tokens, 1),
        "label_match": matched / max(len(examples), 1),
        "records": len(examples),
    }


def rng_state() -> dict:
    """存下各裝置的亂數狀態，續跑時 LoRA dropout 的取樣才會和不中斷的訓練一致。"""
    import torch

    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict) -> None:
    import torch

    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(model, directory: Path, state: dict) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory))
    torch.save({**state, "rng": rng_state()}, directory / STATE_FILE)


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = [path for path in checkpoint_dir.glob("epoch-*") if (path / STATE_FILE).is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: int(path.name.split("-")[1]))


def load_state(checkpoint: Path) -> dict:
    import torch

    return torch.load(checkpoint / STATE_FILE, map_location="cpu", weights_only=False)


def train_adapter(
    *,
    model,
    train_examples: list[Example],
    val_examples: list[Example],
    config: TrainingConfig,
    device: str,
    seed: int,
    checkpoint_dir: Path,
    state: dict | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """跑完剩餘的 epoch，回傳每個 epoch 的訓練／驗證紀錄。

    每個 epoch 的資料順序由 seed + epoch 決定，續跑時順序不變。
    """
    import torch
    from torch.nn.utils import clip_grad_norm_
    from transformers import get_linear_schedule_with_warmup

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    n_train = len(train_examples)
    accumulation = config.gradient_accumulation_steps
    total_steps = math.ceil(n_train / accumulation) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )
    history: list[dict] = []
    start_epoch = global_step = 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, start_epoch, global_step = state["history"], state["epoch"], state["global_step"]
        if "rng" in state:
            set_rng_state(state["rng"])
    logits_to_keep = supports_logits_to_keep(model)
    log(
        "只計算標籤位置的 logits："
        + ("是" if logits_to_keep else "否（模型不支援 logits_to_keep，會算整段序列的 logits）")
    )
    started, steps_at_start = time.perf_counter(), global_step

    for epoch in range(start_epoch, config.epochs):
        order = list(range(n_train))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, tokens, epoch_started = 0.0, 0, time.perf_counter()
        for i, index in enumerate(order, start=1):
            loss, n_tokens, _ = label_loss(
                model, train_examples[index], device, logits_to_keep=logits_to_keep
            )
            # 先對這筆的標籤 token 取平均，再除以累積步數，等於對有效 batch 取平均。
            (loss / n_tokens / accumulation).backward()
            loss_sum += loss.item()
            tokens += n_tokens
            if i % accumulation == 0 or i == n_train:
                clip_grad_norm_(params, config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % config.log_every == 0 or i == n_train:
                    elapsed = time.perf_counter() - started
                    done = global_step - steps_at_start
                    remaining = elapsed / done * (total_steps - global_step) if done else 0
                    log(
                        f"epoch {epoch + 1}/{config.epochs} step {global_step}/{total_steps} "
                        f"loss {loss_sum / tokens:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                        f"已 {elapsed / 60:.1f} 分，約剩 {remaining / 60:.1f} 分"
                    )
        validation = evaluate(model, val_examples, device, logits_to_keep=logits_to_keep)
        record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(tokens, 1),
            "val_loss": validation["loss"],
            "val_label_match": validation["label_match"],
            "global_step": global_step,
            "seconds": round(time.perf_counter() - epoch_started, 1),
        }
        history.append(record)
        log(
            f"=== epoch {epoch + 1} 結束：train_loss {record['train_loss']:.4f}、"
            f"val_loss {record['val_loss']:.4f}、val_label_match {record['val_label_match']:.3f} ==="
        )
        save_checkpoint(
            model,
            checkpoint_dir / f"epoch-{epoch + 1}",
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "history": history,
                "config": config.to_dict(),
            },
        )
    return history
