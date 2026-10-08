"""驗證 assistant loss mask，以及小型 Gemma LoRA 的訓練、儲存和重載。"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src import trainer
from src.config import RuntimeConfig, TrainingConfig
from src.llm import parse_json_object


class FixtureTokenizer:
    """用固定規則映射字元到小詞彙表，方便在 CPU 檢查 token 邊界。"""

    pad_token_id = 0

    def __call__(self, text, **kwargs):
        """產生可重現的 token ID；這個替身只驗證結構，不模擬真實分詞品質。"""
        return {"input_ids": [ord(char) % 59 + 3 for char in text]}


class FixtureProcessor:
    """提供簡短角色模板與保存介面，讓測試專注於 mask 與訓練流程。"""

    tokenizer = FixtureTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        """prompt 與完整對話共用相同前綴，保留可檢查的 assistant 結束記號。"""
        # 短序列便於在 CPU 上驗證架構；真正的模板另以官方 tokenizer 查核。
        prompt = "[S][U]" + messages[1]["content"] + "[A]"
        return prompt if len(messages) == 2 else prompt + messages[2]["content"] + "[E]"

    def save_pretrained(self, path):
        """建立保存標記，確認訓練流程有一起保存文字處理器。"""
        (Path(path) / "fixture_processor.json").write_text("{}")


def example(text="合成測試事件", label="個人"):
    """建立不含真實醫療資料的三角色對話，供 loss mask 與微型訓練共用。"""
    return {
        "messages": [
            {"role": "system", "content": "規則"},
            {"role": "user", "content": text},
            {"role": "assistant", "content": label},
        ]
    }


class TrainingDataTests(unittest.TestCase):
    """驗證人工標籤、模板邊界與 padding，不需載入預訓練模型。"""

    def test_loss_only_includes_assistant_response_and_ending(self):
        """prompt 全部忽略，只讓回覆與結束 token 參與訓練 loss。"""
        processor = FixtureProcessor()
        encoded = trainer.encode_example(example(), processor, 128)
        length = len(
            processor.tokenizer(processor.apply_chat_template(example()["messages"][:-1]))[
                "input_ids"
            ]
        )
        self.assertEqual(encoded["labels"][:length], [-100] * length)
        self.assertEqual(encoded["labels"][length:], encoded["input_ids"][length:])
        self.assertGreater(len(encoded["labels"][length:]), 0)

    def test_context_overflow_fails_instead_of_truncating_the_target(self):
        """超長樣本必須明確失敗，避免截斷事件內容或正確標籤。"""
        with self.assertRaisesRegex(ValueError, "超過 max_length"):
            trainer.encode_example(example(), FixtureProcessor(), 4)

    def test_invalid_template_boundary_fails(self):
        """模板字串前綴不同時，不能憑長度猜測回覆起點。"""
        processor = FixtureProcessor()
        with patch.object(processor, "apply_chat_template", side_effect=["[A]", "[B]個人"]):
            with self.assertRaisesRegex(ValueError, "起始格式不一致"):
                trainer.encode_example(example(), processor, 128)

    def test_inconsistent_token_boundary_fails(self):
        """即使模板字串前綴一致，分詞結果也需要再次確認。"""
        processor = FixtureProcessor()
        processor.tokenizer = Mock(side_effect=[{"input_ids": [1, 2]}, {"input_ids": [1, 3, 4]}])
        with self.assertRaisesRegex(ValueError, "分詞邊界不一致"):
            trainer.encode_example(example(), processor, 128)

    def test_training_uses_idt_target_and_warns_about_conflicting_labels(self):
        """情緒欄位不作為 IDT 目標；相同事件的人工標籤衝突只警告並保留。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "idt_target": label,
                            "emotion_target": "憤怒",
                            "content": {"description": "合成事件"},
                        }
                        for label in ("個人", "系統")
                    ]
                )
            )
            with self.assertLogs(trainer.LOGGER, level="WARNING") as messages:
                examples = trainer.load_examples(str(path))
            self.assertIn("1 組", messages.output[0])
            self.assertEqual(
                [item["messages"][-1]["content"] for item in examples], ["個人", "系統"]
            )

    def test_training_records_reject_malformed_content_and_nonstring_fields(self):
        """錯誤描述結構不得轉成字串繼續訓練，訊息需保留來源筆數。"""
        records = [
            {"idt_target": "個人", "content": None},
            {"idt_target": "個人", "content": []},
            {"idt_target": "個人", "content": {"description": 123}},
            {"idt_target": ["個人"], "content": {"description": "事件"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            for record in records:
                path.write_text(json.dumps([record]), encoding="utf-8")
                with self.subTest(record=record), self.assertRaisesRegex(ValueError, "第 1 筆"):
                    trainer.load_examples(str(path))

    def test_collator_rejects_invalid_padding_and_malformed_batches(self):
        """在 tensor 建構前攔下空批次、缺欄位與三欄長度不一致的資料。"""
        for value in (None, -1, True, 1.5):
            with (
                self.subTest(pad_token_id=value),
                self.assertRaisesRegex(ValueError, "pad_token_id"),
            ):
                trainer.CompletionCollator(value)
        collator = trainer.CompletionCollator(0)
        batches = [
            [],
            [{}],
            [{"input_ids": [1, 2], "attention_mask": [1], "labels": [-100, 2]}],
            [{"input_ids": [], "attention_mask": [], "labels": []}],
        ]
        for batch in batches:
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                collator(batch)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "padding 驗證需要 torch")
    def test_collator_padding_preserves_prompt_mask_and_ignores_padding_loss(self):
        """使用真實 Torch tensor 驗證右側補齊，確認 prompt 與 padding 均無 loss。"""
        batch = trainer.CompletionCollator(0)(
            [
                {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            ]
        )
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 0], [1, 2, 3]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, -100], [-100, 2, 3]])

    def test_json_parser_rejects_non_objects_extra_text_and_nonfinite(self):
        """模型輸出只接受純物件或完整 JSON 區塊，避免寬鬆解析誤收內容。"""
        for raw in ("[]", '說明 {"emotion": "中性"}', '{"score": NaN}'):
            with self.assertRaises(ValueError):
                parse_json_object(raw)
        self.assertEqual(
            parse_json_object('```json\n{"emotion": "中性"}\n```'), {"emotion": "中性"}
        )


# 一般資料測試不依賴整套訓練環境，只有下方真正訓練的測試需要四項依賴。
HAS_LORA = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "peft", "accelerate")
)


@unittest.skipUnless(HAS_LORA, "小型 LoRA 整合測試需要安裝 requirements.txt")
class TinyLoRAIntegrationTests(unittest.TestCase):
    """以隨機初始化的微型 Gemma 真正訓練 LoRA，驗證保存後可重載。"""

    def test_gemma_lora_training_mask_padding_save_and_reload(self):
        """同時檢查 mask、padding、adapter 更新、基礎權重凍結與重載後有限 loss。"""
        import torch
        from peft import PeftModel
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

        def tiny_model():
            """僅建立小型文字子模型，避免下載權重或初始化影像／音訊模組。"""
            text = Gemma4TextConfig(
                vocab_size=64,
                vocab_size_per_layer_input=64,
                hidden_size=16,
                hidden_size_per_layer_input=4,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                global_head_dim=8,
                num_global_key_value_heads=1,
                layer_types=["sliding_attention", "full_attention"],
                max_position_embeddings=128,
                use_cache=False,
                sliding_window=16,
            )
            config = Gemma4Config(
                text_config=text,
                vision_config=None,
                audio_config=None,
                image_token_id=61,
                video_token_id=62,
                audio_token_id=63,
            )
            # adapter 需要記錄可核對的基礎模型名稱，即使測試模型並非下載取得。
            config._name_or_path = "tiny-gemma-fixture"
            return Gemma4ForConditionalGeneration(config)

        model = tiny_model()
        processor = FixtureProcessor()
        collated = trainer.CompletionCollator(0)(
            [
                trainer.encode_example(example("短"), processor, 128),
                trainer.encode_example(example("較長的合成事件"), processor, 128),
            ]
        )
        self.assertTrue((collated["labels"][collated["attention_mask"] == 0] == -100).all())
        # 在 PEFT 包裝前保存基礎權重快照，訓練後逐一確認沒有被更新。
        initial = [(value, value.detach().clone()) for value in model.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            base_config_dir = Path(directory) / "tiny-base"
            model.config.save_pretrained(base_config_dir)
            model.config._name_or_path = str(base_config_dir)
            model.name_or_path = str(base_config_dir)
            path = Path(directory) / "train.json"
            path.write_text(
                json.dumps(
                    [
                        {"idt_target": "個人", "content": {"description": "短"}},
                        {"idt_target": "系統", "content": {"description": "較長的合成事件"}},
                    ]
                )
            )
            # 只替代模型載入；LoRA 包裝、Trainer、保存與重載均走真實套件流程。
            with patch.object(
                trainer, "load_model", return_value=(model, processor, "cpu", torch.float32)
            ):
                adapter = Path(
                    trainer.run(
                        str(path),
                        config=RuntimeConfig(gemma_model=str(base_config_dir), device="cpu"),
                        training_config=TrainingConfig(
                            epochs=1, batch_size=2, gradient_accumulation_steps=1, max_length=128
                        ),
                        models_dir=str(Path(directory) / "models"),
                        interim_dir=str(Path(directory) / "interim"),
                    )
                )
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
