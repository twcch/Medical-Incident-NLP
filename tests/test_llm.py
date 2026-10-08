"""以替身驗證模型 session 生命週期與生成參數，測試不下載模型。"""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import llm
from src.config import RuntimeConfig


class FakeInputs(dict):
    """模擬 tokenizer 的批次字典與 .to() 介面，測試不用配置真實 tensor。"""

    def to(self, device):
        """保留呼叫形狀即可；假輸入無須移到 GPU。"""
        return self


class JSONResponseTests(unittest.TestCase):
    """模型文字必須能嚴格解析為一個 JSON 物件。"""

    def test_non_string_and_nonfinite_json_responses_are_rejected(self):
        """兼顧非字串、非標準常數及合法指數語法造成的浮點數溢位。"""
        for raw in (
            None,
            {},
            b"{}",
            '{"score": Infinity}',
            '{"score": -Infinity}',
            '{"score": 1e400}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                llm.parse_json_object(raw)

    def test_incomplete_or_extra_markdown_is_rejected(self):
        """不從不完整區塊或帶說明的輸出中猜測 JSON 範圍。"""
        for raw in ("```json\n{}", "```json\n{}\n```\n說明", "```python\n{}\n```"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                llm.parse_json_object(raw)


class ModelSessionTests(unittest.TestCase):
    """以模型替身檢查 session 邊界、生成選項及錯誤時的資源清理。"""

    def test_generate_requires_an_active_session(self):
        """尚未進入 with 時先回報生命週期錯誤，避免誤用未載入模型。"""
        with self.assertRaisesRegex(RuntimeError, "with ModelSession"):
            llm.ModelSession("fixture").generate("規則", "事件")

    def test_invalid_generation_parameters_fail_before_generation(self):
        """無效參數不能進入模板處理或模型生成，省下不必要的運算。"""
        session = llm.ModelSession("fixture")
        session.model = Mock()
        session.processor = Mock()
        cases = [{"max_new_tokens": value} for value in (0, -1, True, 1.5)]
        cases += [{"temperature": value} for value in (-1, float("nan"), float("inf"), True)]
        cases += [{"text": None}, {"system_prompt": None}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                arguments = {"system_prompt": "規則", "text": "事件", **kwargs}
                session.generate(**arguments)
        session.model.generate.assert_not_called()
        session.processor.apply_chat_template.assert_not_called()

    def test_generation_uses_default_or_explicit_limit_and_only_decodes_completion(self):
        """以兩個 prompt token 模擬輸出切片，同時驗證 greedy 與 sampling 選項。"""
        session = llm.ModelSession("fixture", RuntimeConfig(max_new_tokens=7))
        session.model = Mock(device="cpu")
        session.model.generate.return_value = [[1, 2, 3, 4]]
        session.processor = Mock()
        session.processor.apply_chat_template.return_value = FakeInputs(
            input_ids=SimpleNamespace(shape=(1, 2)),
        )
        session.processor.decode.return_value = " 個人 "
        # 只替代 inference_mode 的 context manager，不必載入 Torch 或下載模型。
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            self.assertEqual(session.generate("規則", "事件"), "個人")
            options = session.model.generate.call_args.kwargs
            self.assertEqual(options["max_new_tokens"], 7)
            self.assertFalse(options["do_sample"])
            self.assertNotIn("temperature", options)
            session.processor.decode.assert_called_with([3, 4], skip_special_tokens=True)
            session.generate("規則", "事件", max_new_tokens=3, temperature=0.5)
            options = session.model.generate.call_args.kwargs
            self.assertEqual(options["max_new_tokens"], 3)
            self.assertTrue(options["do_sample"])
            self.assertEqual(options["temperature"], 0.5)

    def test_loading_interruption_clears_references_and_releases_memory(self):
        """模擬模型已配置後收到中斷，確認模型與 processor 參照仍會移除。"""
        session = llm.ModelSession("fixture")
        model = Mock()
        model.eval.side_effect = KeyboardInterrupt
        with (
            patch.object(llm, "load_model", return_value=(model, Mock(), "cpu", None)),
            patch.object(llm, "release_memory") as release,
        ):
            with self.assertRaises(KeyboardInterrupt):
                session.__enter__()
        self.assertIsNone(session.model)
        self.assertIsNone(session.processor)
        release.assert_called_once_with()

    def test_reentering_session_keeps_outer_model_until_context_exit(self):
        """重入被拒絕時不能順便清掉外層 with 正在使用的模型。"""
        session = llm.ModelSession("fixture")
        model = Mock()
        with (
            patch.object(llm, "load_model", return_value=(model, Mock(), "cpu", None)) as load,
            patch.object(llm, "release_memory") as release,
        ):
            with session:
                with self.assertRaisesRegex(RuntimeError, "重複進入"):
                    session.__enter__()
                self.assertIs(session.model, model)
                release.assert_not_called()
            load.assert_called_once()
            release.assert_called_once_with()
        self.assertIsNone(session.model)

    def test_invalid_adapter_config_fails_before_loading_the_model(self):
        """adapter 設定格式錯誤應在載入昂貴的基礎模型之前停止。"""
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "adapter_config.json").write_text("[]", encoding="utf-8")
            session = llm.ModelSession("fixture", adapter_path=directory)
            with patch.object(llm, "load_model") as load, patch.object(llm, "release_memory"):
                with self.assertRaisesRegex(ValueError, "JSON 物件"):
                    session.__enter__()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
