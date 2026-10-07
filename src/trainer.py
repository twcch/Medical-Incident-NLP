"""以人工 IDT 標籤對 Gemma 執行本機 LoRA supervised fine-tuning。"""

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.config import RuntimeConfig, TrainingConfig
from src.llm import load_model, release_memory
from src.prompts import IDT_SYSTEM_PROMPT, VALID_IDT_LABELS

TEXT_FIELD = "description"
IDT_TARGET = "idt_target"
LOGGER = logging.getLogger(__name__)


def load_examples(json_path: str) -> list:
    """僅使用 description 與人工 IDT 標籤；情緒偽標籤不作為訓練目標。"""
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("訓練資料必須是 JSON 紀錄陣列。")
    examples = []
    targets_by_text = {}
    for index, record in enumerate(records, start=1):
        text = str(record.get("content", {}).get(TEXT_FIELD) or "").strip()
        label = str(record.get(IDT_TARGET) or "").strip()
        if not text or label not in VALID_IDT_LABELS:
            raise ValueError(f"第 {index} 筆訓練資料缺少描述或有效人工 IDT 標籤，請先執行清洗。")
        targets_by_text.setdefault(text, set()).add(label)
        examples.append({"messages": [
            {"role": "system", "content": IDT_SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": label},
        ]})
    if not examples:
        raise ValueError("沒有可用的 IDT 訓練樣本。")
    conflicts = sum(len(labels) > 1 for labels in targets_by_text.values())
    if conflicts:
        LOGGER.warning("有 %d 組相同事件描述具有互相衝突的人工 IDT 標籤；保留原標籤，請人工核對來源。", conflicts)
    return examples


def write_jsonl(examples: list, jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def encode_example(example: dict, processor, max_length: int) -> dict:
    """使用相同 chat template 建立 prompt/completion 邊界，避免把 prompt 算進 loss。"""
    messages = example["messages"]
    prompt = processor.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    full = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    if not full.startswith(prompt):
        raise ValueError("chat template 的 assistant 起始格式不一致，無法可靠建立 loss mask。")
    tokenizer = processor.tokenizer
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt/completion 分詞邊界不一致，無法可靠建立 loss mask。")
    if len(input_ids) > max_length:
        raise ValueError(
            f"訓練樣本需要 {len(input_ids)} tokens，超過 max_length={max_length}；"
            "請提高 --max-length，不會截掉醫療事件內容或目標標籤。"
        )
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("assistant 訓練目標沒有有效 tokens。")
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class IDTDataset:
    def __init__(self, examples: list, processor, max_length: int):
        self.items = []
        for index, example in enumerate(examples, start=1):
            try:
                self.items.append(encode_example(example, processor, max_length))
            except ValueError as exc:
                raise ValueError(f"第 {index} 筆：{exc}") from exc

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, items: list) -> dict:
        import torch

        length = max(len(item["input_ids"]) for item in items)
        return {
            key: torch.tensor([
                item[key] + [padding] * (length - len(item[key])) for item in items
            ], dtype=torch.long)
            for key, padding in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))
        }


def run(json_path: str = "data/interim/train_features.json", *,
        config: RuntimeConfig | None = None, training_config: TrainingConfig | None = None,
        models_dir: str = "models", interim_dir: str = "data/interim") -> str:
    config = config or RuntimeConfig()
    training_config = training_config or TrainingConfig()
    examples = load_examples(json_path)
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("LoRA 訓練需要 requirements.txt 的 transformers、peft、torch 與 accelerate。") from exc

    set_seed(config.seed)
    output_dir = Path(models_dir) / "idt_adapter"
    model = processor = dataset = hf_trainer = None
    try:
        model, processor, device, dtype = load_model(config.gemma_model, config, training=True)
        dataset = IDTDataset(examples, processor, training_config.max_length)
        write_jsonl(examples, Path(interim_dir) / "idt_train.jsonl")
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("模型 tokenizer 未定義 pad_token_id。")
        model.config.use_cache = False
        model.config.get_text_config().use_cache = False
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=training_config.lora_rank,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            target_modules=training_config.target_modules,
            bias="none",
        ))
        model.print_trainable_parameters()
        if training_config.gradient_checkpointing:
            model.enable_input_require_grads()
        arguments = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config.epochs,
            per_device_train_batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            gradient_checkpointing=training_config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=device == "cuda" and dtype == torch.bfloat16,
            fp16=device == "cuda" and dtype == torch.float16,
            use_cpu=device == "cpu",
            optim="adamw_torch",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            seed=config.seed,
        )
        hf_trainer = Trainer(
            model=model, args=arguments, train_dataset=dataset,
            data_collator=CompletionCollator(pad_token_id),
        )
        print(f"[idt] {config.gemma_model} LoRA，{len(dataset)} 筆，device={device}，dtype={dtype}")
        train_result = hf_trainer.train()
        hf_trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))
        metadata = {
            "task": "idt", "method": "lora", "base_model": config.gemma_model,
            "n_train": len(dataset), "target_field": IDT_TARGET,
            "input_field": f"content.{TEXT_FIELD}", "assistant_only_loss": True,
            "seed": config.seed, "runtime": asdict(config), "training": asdict(training_config),
            "data_sha256": hashlib.sha256(Path(json_path).read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(IDT_SYSTEM_PROMPT.encode()).hexdigest(),
            "metrics": train_result.metrics,
        }
        (output_dir / "training_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"[idt] 已儲存 LoRA adapter：{output_dir}")
        return str(output_dir)
    finally:
        model = processor = dataset = hf_trainer = None
        release_memory()


if __name__ == "__main__":
    run()
