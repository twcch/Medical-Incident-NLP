"""設定在模型載入前拒絕不合法型別、非有限數值與無效 LoRA pattern。"""

import unittest

from src.config import RuntimeConfig, TrainingConfig


class RuntimeConfigTests(unittest.TestCase):
    """確認執行設定在載入大型模型前便回報不合法輸入。"""

    def test_generation_limit_requires_positive_integer(self):
        """生成上限只接受正整數，避免 bool 或小數混入 token 計數。"""
        for value in (0, -1, True, 1.5, "10", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_new_tokens"):
                RuntimeConfig(max_new_tokens=value)

    def test_model_identifiers_require_nonempty_strings(self):
        """兩個模型的 ID 均須有效，避免稍後於不同流程才發現缺漏。"""
        for name in ("gemma_model", "judge_model"):
            for value in ("", "  ", None, 123):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    RuntimeConfig(**{name: value})

    def test_invalid_device_dtype_seed_and_boolean_option_are_rejected(self):
        """設定型別及 seed 範圍錯誤都應提供欄位名稱供使用者定位。"""
        for name, value in (
            ("device", []),
            ("dtype", None),
            ("seed", -1),
            ("seed", 2**32),
            ("seed", True),
            ("seed", 1.5),
            ("local_files_only", "false"),
        ):
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                RuntimeConfig(**{name: value})


class TrainingConfigTests(unittest.TestCase):
    """驗證 LoRA 訓練參數的數值邊界與合法設定相容性。"""

    def test_integer_training_parameters_reject_fractions_and_booleans(self):
        """批次、累積步數、長度與 LoRA 大小都需要可實際計數的整數。"""
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_length",
            "lora_rank",
            "lora_alpha",
        ):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    TrainingConfig(**{name: value})

    def test_training_numbers_reject_nonfinite_values(self):
        """NaN／Infinity 不得穿過大小比較，進入最佳化器或 dropout。"""
        for name in ("epochs", "learning_rate", "lora_dropout"):
            for value in (float("nan"), float("inf"), float("-inf"), True, "0.1"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    TrainingConfig(**{name: value})

    def test_dropout_limits_and_lora_pattern_are_checked(self):
        """先檢查 dropout 範圍與 regex 語法，不必載入 PEFT 才回報設定錯誤。"""
        for name, value in (
            ("lora_dropout", -0.01),
            ("lora_dropout", 1),
            ("target_modules", ""),
            ("target_modules", "["),
            ("target_modules", ["q_proj"]),
            ("gradient_checkpointing", "false"),
        ):
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                TrainingConfig(**{name: value})

    def test_fractional_epochs_and_zero_dropout_remain_supported(self):
        """數值驗證仍允許短輪數實驗與停用 dropout 的合法情況。"""
        config = TrainingConfig(epochs=0.5, lora_dropout=0)
        self.assertEqual(config.epochs, 0.5)
        self.assertEqual(config.lora_dropout, 0)


if __name__ == "__main__":
    unittest.main()
