"""驗證 assistant loss mask，以及小型 Gemma LoRA 的訓練、儲存和重載。"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import trainer
from src.config import RuntimeConfig, TrainingConfig
from src.llm import parse_json_object


class FixtureTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) % 59 + 3 for char in text]}


class FixtureProcessor:
    tokenizer = FixtureTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        # 短序列便於在 CPU 上驗證架構；真正的模板另以官方 tokenizer 查核。
        prompt = "[S][U]" + messages[1]["content"] + "[A]"
        return prompt if len(messages) == 2 else prompt + messages[2]["content"] + "[E]"

    def save_pretrained(self, path):
        (Path(path) / "fixture_processor.json").write_text("{}")


def example(text="合成測試事件", label="個人"):
    return {"messages": [{"role": "system", "content": "規則"},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": label}]}


class TrainingDataTests(unittest.TestCase):
    def test_loss_only_includes_assistant_response_and_ending(self):
        processor = FixtureProcessor()
        encoded = trainer.encode_example(example(), processor, 128)
        length = len(processor.tokenizer(processor.apply_chat_template(
            example()["messages"][:-1]))["input_ids"])
        self.assertEqual(encoded["labels"][:length], [-100] * length)
        self.assertEqual(encoded["labels"][length:], encoded["input_ids"][length:])
        self.assertGreater(len(encoded["labels"][length:]), 0)

    def test_context_overflow_fails_instead_of_truncating_the_target(self):
        with self.assertRaisesRegex(ValueError, "超過 max_length"):
            trainer.encode_example(example(), FixtureProcessor(), 4)

    def test_invalid_template_boundary_fails(self):
        processor = FixtureProcessor()
        with patch.object(processor, "apply_chat_template", side_effect=["[A]", "[B]個人"]):
            with self.assertRaisesRegex(ValueError, "起始格式不一致"):
                trainer.encode_example(example(), processor, 128)

    def test_training_uses_idt_target_and_warns_about_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(json.dumps([
                {"idt_target": label, "emotion_target": "憤怒", "content": {"description": "合成事件"}}
                for label in ("個人", "系統")
            ]))
            with self.assertLogs(trainer.LOGGER, level="WARNING") as messages:
                examples = trainer.load_examples(str(path))
            self.assertIn("1 組", messages.output[0])
            self.assertEqual([item["messages"][-1]["content"] for item in examples], ["個人", "系統"])

    def test_json_parser_rejects_non_objects_extra_text_and_nonfinite(self):
        for raw in ('[]', '說明 {"emotion": "中性"}', '{"score": NaN}'):
            with self.assertRaises(ValueError):
                parse_json_object(raw)
        self.assertEqual(parse_json_object('```json\n{"emotion": "中性"}\n```'), {"emotion": "中性"})


HAS_LORA = all(importlib.util.find_spec(name) is not None
               for name in ("torch", "transformers", "peft", "accelerate"))


@unittest.skipUnless(HAS_LORA, "小型 LoRA 整合測試需要安裝 requirements.txt")
class TinyLoRAIntegrationTests(unittest.TestCase):
    def test_gemma_lora_training_mask_padding_save_and_reload(self):
        import torch
        from peft import PeftModel
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

        def tiny_model():
            text = Gemma4TextConfig(
                vocab_size=64, vocab_size_per_layer_input=64, hidden_size=16,
                hidden_size_per_layer_input=4, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=8,
                num_global_key_value_heads=1, layer_types=["sliding_attention", "full_attention"],
                max_position_embeddings=128, use_cache=False, sliding_window=16,
            )
            config = Gemma4Config(text_config=text, vision_config=None, audio_config=None,
                                 image_token_id=61, video_token_id=62, audio_token_id=63)
            config._name_or_path = "tiny-gemma-fixture"
            return Gemma4ForConditionalGeneration(config)

        model = tiny_model()
        processor = FixtureProcessor()
        collated = trainer.CompletionCollator(0)([
            trainer.encode_example(example("短"), processor, 128),
            trainer.encode_example(example("較長的合成事件"), processor, 128),
        ])
        self.assertTrue((collated["labels"][collated["attention_mask"] == 0] == -100).all())
        initial = [(value, value.detach().clone()) for value in model.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            base_config_dir = Path(directory) / "tiny-base"
            model.config.save_pretrained(base_config_dir)
            model.config._name_or_path = str(base_config_dir)
            model.name_or_path = str(base_config_dir)
            path = Path(directory) / "train.json"
            path.write_text(json.dumps([
                {"idt_target": "個人", "content": {"description": "短"}},
                {"idt_target": "系統", "content": {"description": "較長的合成事件"}},
            ]))
            with patch.object(trainer, "load_model", return_value=(model, processor, "cpu", torch.float32)):
                adapter = Path(trainer.run(
                    str(path), config=RuntimeConfig(gemma_model=str(base_config_dir), device="cpu"),
                    training_config=TrainingConfig(epochs=1, batch_size=2,
                                                  gradient_accumulation_steps=1, max_length=128),
                    models_dir=str(Path(directory) / "models"),
                    interim_dir=str(Path(directory) / "interim"),
                ))
            self.assertTrue((adapter / "adapter_config.json").is_file())
            self.assertTrue((adapter / "adapter_model.safetensors").is_file())
            metadata = json.loads((adapter / "training_metadata.json").read_text())
            self.assertEqual(metadata["n_train"], 2)
            self.assertTrue(metadata["assistant_only_loss"])
            lora_weights = [value for name, value in model.named_parameters() if "lora_B" in name]
            self.assertTrue(any(torch.count_nonzero(weight).item() > 0 for weight in lora_weights))
            # LoRA 應保持原始 base 權重不變，只更新 adapter。
            for parameter, value in initial:
                self.assertTrue(torch.equal(value, parameter))
            reloaded = PeftModel.from_pretrained(tiny_model(), str(adapter), is_trainable=False)
            with torch.inference_mode():
                result = reloaded(**collated)
            self.assertTrue(torch.isfinite(result.loss))
            self.assertEqual(tuple(result.logits.shape[:2]), tuple(collated["input_ids"].shape))


if __name__ == "__main__":
    unittest.main()
