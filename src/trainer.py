"""以人工 IDT 標籤對 Gemma 執行本機 LoRA supervised fine-tuning。"""

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.config import RuntimeConfig, TrainingConfig
from src.json_io import load_records, save_json
from src.llm import load_model, release_memory
from src.prompts import IDT_SYSTEM_PROMPT, VALID_IDT_LABELS

TEXT_FIELD = "description"
IDT_TARGET = "idt_target"
LOGGER = logging.getLogger(__name__)


def load_examples(json_path: str) -> list:
    """僅使用 description 與人工 IDT 標籤；情緒偽標籤不作為訓練目標。"""
    records = load_records(json_path)
    examples = []
    targets_by_text = {}
    for index, record in enumerate(records, start=1):
        content = record.get("content")
        if not isinstance(content, dict):
            raise ValueError(f"第 {index} 筆訓練資料的 content 必須為 JSON 物件，請先執行清洗。")
        description = content.get(TEXT_FIELD)
        target = record.get(IDT_TARGET)
        if not isinstance(description, str) or not isinstance(target, str):
            raise ValueError(
                f"第 {index} 筆訓練資料的描述與人工 IDT 標籤必須為字串，請先執行清洗。"
            )
        text = description.strip()
        label = target.strip()
        if not text or label not in VALID_IDT_LABELS:
            raise ValueError(f"第 {index} 筆訓練資料缺少描述或有效人工 IDT 標籤，請先執行清洗。")
        targets_by_text.setdefault(text, set()).add(label)
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": IDT_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": label},
                ]
            }
        )
    if not examples:
        raise ValueError("沒有可用的 IDT 訓練樣本。")
    # 衝突只警告、不自動選標籤，保留人工資料的原始判斷供後續核對。
    conflicts = sum(len(labels) > 1 for labels in targets_by_text.values())
    if conflicts:
        LOGGER.warning(
            "有 %d 組相同事件描述具有互相衝突的人工 IDT 標籤；保留原標籤，請人工核對來源。",
            conflicts,
        )
    return examples


def write_jsonl(examples: list, jsonl_path: Path) -> None:
    """保留每筆 messages 為一行，供人工追查實際訓練輸入。"""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, allow_nan=False) + "\n")


def encode_example(example: dict, processor, max_length: int) -> dict:
    """使用相同 chat template 建立 prompt/completion 邊界，避免把 prompt 算進 loss。"""
    messages = example["messages"]
    prompt = processor.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    # 字串前綴與 token 前綴都必須一致，才能把相同位置視為 assistant 回覆起點。
    if not full.startswith(prompt):
        raise ValueError("chat template 的 assistant 起始格式不一致，無法可靠建立 loss mask。")
    tokenizer = processor.tokenizer
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt/completion 分詞邊界不一致，無法可靠建立 loss mask。")
    if len(input_ids) > max_length:
        # 完整事件與人工標籤比截斷後勉強訓練更重要，超長時請使用者調整設定。
        raise ValueError(
            f"訓練樣本需要 {len(input_ids)} tokens，超過 max_length={max_length}；"
            "請提高 --max-length，不會截掉醫療事件內容或目標標籤。"
        )
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("assistant 訓練目標沒有有效 tokens。")
    # -100 是交叉熵忽略索引；只監督 assistant 回覆及其結束 token。
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class IDTDataset:
    """訓練前先驗證所有樣本，避免訓練半途才發現超長序列。"""

    def __init__(self, examples: list, processor, max_length: int):
        """預先編碼並把驗證錯誤補上筆數，方便定位來源資料。"""
        self.items = []
        for index, example in enumerate(examples, start=1):
            try:
                self.items.append(encode_example(example, processor, max_length))
            except ValueError as exc:
                raise ValueError(f"第 {index} 筆：{exc}") from exc

    def __len__(self):
        """提供 Trainer 用於計算批次與訓練步數的樣本數。"""
        return len(self.items)

    def __getitem__(self, index):
        """提供已驗證的編碼結果，避免每個 epoch 重複分詞。"""
        return self.items[index]


class CompletionCollator:
    """批次採右側 padding；補齊位置的 attention 與 loss 均忽略。"""

    def __init__(self, pad_token_id: int):
        """使用模型 tokenizer 定義的 padding token，避免自行猜測 token ID。"""
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("pad_token_id 必須為大於等於 0 的整數。")
        self.pad_token_id = pad_token_id

    def __call__(self, items: list) -> dict:
        """確認三個欄位對齊後，補到批次最長序列並轉成整數 tensor。"""
        if not items:
            raise ValueError("訓練批次不可為空。")
        for index, item in enumerate(items, start=1):
            keys = ("input_ids", "attention_mask", "labels")
            if not isinstance(item, dict) or any(
                not isinstance(item.get(key), list) for key in keys
            ):
                raise ValueError(
                    f"批次第 {index} 筆必須包含 input_ids、attention_mask 與 labels 陣列。"
                )
            lengths = {len(item[key]) for key in keys}
            if len(lengths) != 1 or not item["input_ids"]:
                raise ValueError(
                    f"批次第 {index} 筆的 input_ids、attention_mask 與 labels 必須等長且非空。"
                )

        import torch

        length = max(len(item["input_ids"]) for item in items)
        return {
            key: torch.tensor(
                [item[key] + [padding] * (length - len(item[key])) for item in items],
                dtype=torch.long,
            )
            for key, padding in (
                ("input_ids", self.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            )
        }


def run(
    json_path: str = "data/interim/train_features.json",
    *,
    config: RuntimeConfig | None = None,
    training_config: TrainingConfig | None = None,
    models_dir: str = "models",
    interim_dir: str = "data/interim",
) -> str:
    """訓練及儲存 IDT adapter；無論成功或失敗都清理模型記憶體。"""
    config = config or RuntimeConfig()
    training_config = training_config or TrainingConfig()
    examples = load_examples(json_path)
    # 先驗證資料，再載入訓練套件與模型；格式錯誤不用等待重型依賴初始化。
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "LoRA 訓練需要 requirements.txt 的 transformers、peft、torch 與 accelerate。"
        ) from exc

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
        # 訓練不保留推論用的 KV cache，並與 gradient checkpointing 的重算流程配合。
        model.config.use_cache = False
        model.config.get_text_config().use_cache = False
        # PEFT 凍結基礎模型；僅在指定文字層加入低秩 adapter 參數。
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=training_config.lora_rank,
                lora_alpha=training_config.lora_alpha,
                lora_dropout=training_config.lora_dropout,
                target_modules=training_config.target_modules,
                bias="none",
            ),
        )
        model.print_trainable_parameters()
        if training_config.gradient_checkpointing:
            # 基礎權重凍結時仍需保留輸入的梯度路徑，供 checkpoint 重算與 LoRA 更新。
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
            model=model,
            args=arguments,
            train_dataset=dataset,
            data_collator=CompletionCollator(pad_token_id),
        )
        print(f"[idt] {config.gemma_model} LoRA，{len(dataset)} 筆，device={device}，dtype={dtype}")
        train_result = hf_trainer.train()
        hf_trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))
        # 保存參數與來源雜湊，方便比較實驗使用的資料檔、提示詞及訓練設定。
        metadata = {
            "task": "idt",
            "method": "lora",
            "base_model": config.gemma_model,
            "n_train": len(dataset),
            "target_field": IDT_TARGET,
            "input_field": f"content.{TEXT_FIELD}",
            "assistant_only_loss": True,
            "seed": config.seed,
            "runtime": asdict(config),
            "training": asdict(training_config),
            "data_sha256": hashlib.sha256(Path(json_path).read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(IDT_SYSTEM_PROMPT.encode()).hexdigest(),
            "metrics": train_result.metrics,
        }
        save_json(metadata, output_dir / "training_metadata.json")
        print(f"[idt] 已儲存 LoRA adapter：{output_dir}")
        return str(output_dir)
    finally:
        # Trainer 也持有模型參照，需一起移除才有機會回收實際模型記憶體。
        model = processor = dataset = hf_trainer = None
        release_memory()


if __name__ == "__main__":
    run()
