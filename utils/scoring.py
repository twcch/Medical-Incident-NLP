"""步驟 05 的模型端：不生成文字，直接讀回答第一個 token 的 logit，得到每筆的 P(系統)。

04 的 ⑥ 用貪婪解碼，等於固定在 P(系統) > 0.5 才判「系統」，也看不出模型有多確定。這裡用與
訓練相同的提示與 chat template，取回答位置上「個人」「系統」兩個標籤第一個 token 的 log 機率，
在兩者之間正規化成 P(系統)，之後才能算 AUROC／AP、挑 checkpoint、在驗證集調門檻。
標籤第一個 token 的取法與 03 的 build_example 相同（Qwen3.5 的「個人」「系統」各是一個 token）；
兩個標籤的第一個 token 必須不同，否則無法只看一個位置。torch 只在真正評分時才 import。
"""

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from utils import io
from utils.emotion import record_key
from utils.preprocessing import TEXT_COLUMN, VALID_IDT_LABELS
from utils.progress import Progress, fingerprint
from utils.prompts import IDT_PROMPT_ID, build_idt_messages
from utils.training import CHECKPOINT_DIR, _token_ids

ZERO_SHOT = "zero-shot"  # 不掛 adapter 的基礎模型，同一份 IDT 提示
FINAL_ADAPTER = "final"  # adapter 目錄最上層、與所有 checkpoint 權重都不同的 adapter
SCORE_METHOD = "first-token-logit"  # 寫進進度檔；評分方式改了，舊進度就不能沿用
WEIGHTS_FILE = "adapter_model.safetensors"


def prompt_token_ids(tokenizer, description: str) -> list[int]:
    """⑥ 推論時送進模型的序列（含 assistant 開頭與關閉思考的區塊），與訓練時標籤前的部分相同。"""
    return _token_ids(
        tokenizer.apply_chat_template(
            build_idt_messages(description),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def label_token_ids(tokenizer, description: str) -> dict[str, int]:
    """回傳 {"個人": id, "系統": id}：模型回答時第一個 token 的 id，序列建法與 03 的 build_example 相同。"""
    messages = build_idt_messages(description)
    prompt = prompt_token_ids(tokenizer, description)
    ids: dict[str, int] = {}
    for label in VALID_IDT_LABELS:
        full = _token_ids(
            tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": label}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        if full[: len(prompt)] != prompt:
            raise ValueError("chat template 的提示不是完整序列的前綴，無法定位回答的第一個 token。")
        if len(full) <= len(prompt):
            raise ValueError(f"標籤「{label}」沒有產生任何 token。")
        ids[label] = full[len(prompt)]
    if len(set(ids.values())) < len(ids):
        raise ValueError(
            f"「個人」與「系統」的第一個 token 相同（{ids}），無法只看一個位置的 logit。"
        )
    return ids


def two_way_probability(logprob_person: float, logprob_system: float) -> float:
    """只在兩個標籤之間正規化的 P(系統) = 1 / (1 + exp(lp個人 − lp系統))，避免 exp 溢位。"""
    difference = logprob_person - logprob_system
    if difference >= 0:
        ratio = math.exp(-difference)
        return ratio / (1 + ratio)
    return 1 / (1 + math.exp(difference))


def score_record(
    model, tokenizer, description: str, label_ids: dict[str, int], device: str, *, logits_to_keep
) -> dict:
    """一次前向，取最後一個位置的 log 機率。回傳 P(系統)、兩個標籤的 log 機率、兩者合計的機率
    質量，以及整個詞彙表裡機率最高的 token 是不是其中一個標籤（貪婪解碼會不會直接輸出標籤）。"""
    import torch

    input_ids = torch.tensor([prompt_token_ids(tokenizer, description)], device=device)
    kwargs = {"logits_to_keep": 1} if logits_to_keep else {}
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **kwargs)
    log_probs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
    logprob = {label: float(log_probs[token]) for label, token in label_ids.items()}
    person, system = VALID_IDT_LABELS
    return {
        "p_system": two_way_probability(logprob[person], logprob[system]),
        "logprob": logprob,
        "label_mass": sum(math.exp(value) for value in logprob.values()),
        "top_is_label": int(log_probs.argmax()) in label_ids.values(),
    }


@dataclass
class ScoreReport:
    records: int = 0
    reused: int = 0
    scored: int = 0

    def to_dict(self) -> dict:
        return {"records": self.records, "reused_from_progress": self.reused, "scored": self.scored}


def score_records(
    records: list[dict],
    *,
    score: Callable[[str], dict],
    progress: Progress,
    on_record: Callable[[int, dict, dict, bool], None] | None = None,
) -> tuple[list[dict], ScoreReport]:
    """逐筆評分並寫進度檔；識別碼與描述都相同的直接沿用，描述改了就重評。回傳與 records 同序的結果。"""
    report = ScoreReport(records=len(records))
    values: list[dict] = []
    for position, record in enumerate(records, start=1):
        key = record_key(record)
        fp = fingerprint(key, record[TEXT_COLUMN])
        entry = progress.get(key, fp)
        reused = entry is not None
        if reused:
            value = entry["value"]
            report.reused += 1
        else:
            value = score(record[TEXT_COLUMN])
            progress.add(key, fp, value=value)
            report.scored += 1
        values.append(value)
        if on_record:
            on_record(position, record, value, reused)
    return values, report


def weights_sha256(adapter: Path) -> str | None:
    """adapter 權重檔的 SHA-256；用來辨認最終 adapter 與某個 checkpoint 是不是同一份。"""
    path = Path(adapter) / WEIGHTS_FILE
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_metadata(adapter: Path) -> dict:
    """adapter 的基礎模型、訓練時的 IDT 提示雜湊與權重雜湊。

    checkpoints/epoch-N 沒有自己的 training_metadata.json，改讀上兩層 adapter 目錄的那份
    （03 訓練完才寫入；中斷的訓練沒有，這時只能從 adapter_config.json 比對基礎模型）。
    """
    adapter = Path(adapter)
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"找不到 adapter：{config_path}")
    metadata_path = adapter / "training_metadata.json"
    if not metadata_path.is_file() and adapter.parent.name == CHECKPOINT_DIR:
        metadata_path = adapter.parent.parent / "training_metadata.json"
    metadata = io.load_json(metadata_path) if metadata_path.is_file() else {}
    return {
        "path": str(adapter),
        "base_model": metadata.get("base_model")
        or io.load_json(config_path).get("base_model_name_or_path"),
        "idt_prompt": metadata.get("idt_prompt"),
        "training_metadata": str(metadata_path) if metadata else None,
        "sha256": weights_sha256(adapter),
    }


def check_adapter_compatible(info: dict, model_name: str) -> None:
    """基礎模型與 IDT 提示都必須與訓練時相同；找不到訓練 metadata 時只能比對基礎模型。"""
    if info.get("base_model") and info["base_model"] != model_name:
        raise ValueError(
            f"{info['path']} 是用 {info['base_model']} 訓練的，本次 --model 是 {model_name}；"
            "基礎模型必須一致。"
        )
    if info.get("idt_prompt") and info["idt_prompt"] != IDT_PROMPT_ID:
        raise ValueError(
            f"{info['path']} 是用 IDT 提示 {info['idt_prompt']} 訓練的，目前 utils/prompts.py 的是 "
            f"{IDT_PROMPT_ID}；提示詞改了要用 03 重新訓練，或換回訓練時的提示。"
        )


def discover_adapters(adapter_dir: Path) -> list[tuple[str, Path]]:
    """列出 adapter_dir/checkpoints/epoch-N（依 N 排序）。最上層的 adapter 若與某個 checkpoint
    權重相同（03 的最終 adapter 就是最後一個 epoch）就不重複列，不同才另列為 final。"""
    adapter_dir = Path(adapter_dir)
    epochs = []
    checkpoints = adapter_dir / CHECKPOINT_DIR
    if checkpoints.is_dir():
        for path in checkpoints.iterdir():
            match = re.fullmatch(r"epoch-(\d+)", path.name)
            if match and (path / "adapter_config.json").is_file():
                epochs.append((int(match.group(1)), path))
    adapters = [(f"epoch-{epoch}", path) for epoch, path in sorted(epochs)]
    if (adapter_dir / "adapter_config.json").is_file():
        final = weights_sha256(adapter_dir)
        if final is None or final not in {weights_sha256(path) for _, path in adapters}:
            adapters.append((FINAL_ADAPTER, adapter_dir))
    return adapters
