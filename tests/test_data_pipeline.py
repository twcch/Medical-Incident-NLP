"""使用人工 fixture 與假模型驗證資料流程，不讀取研究資料或下載模型。"""

import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from src import data_augmentation as augmentation
from src import data_preprocessing as preprocessing


class DataPreprocessingTests(unittest.TestCase):
    def test_excel_cleaning_and_provenance(self):
        frame = pd.DataFrame({
            " IDT分析(個人,系統) ": [" 個　人 ", "系統", "OR", "個人", " 系統 ", "系統", "個人"],
            "事件描述": [" 測試事件甲 ", "測試事件乙", "無效", "  ", "測試事件乙", "測試事件丙", "測試事件甲"],
            "批示": [None, " 處理\r\n步驟 ", None, None, "處理\n步驟", None, None],
        })
        with tempfile.TemporaryDirectory() as temporary:
            workbook = Path(temporary) / "fixture.xlsx"
            output = Path(temporary) / "clean.json"
            frame.to_excel(workbook, sheet_name="112.01", index=False)
            with redirect_stdout(io.StringIO()):
                cleaned = preprocessing.run(["112.01"], str(output), raw_file=workbook)
            records = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(cleaned.attrs["cleaning_stats"], {
            "input_records": 7, "invalid_targets": 1, "empty_descriptions": 1,
            "duplicates": 2, "output_records": 3,
        })
        self.assertEqual([item["idt_target"] for item in records], ["個人", "系統", "系統"])
        self.assertEqual([item["source_id"] for item in records], ["112.01:2", "112.01:3", "112.01:7"])
        self.assertEqual(records[1]["content"]["directive"], "處理\n步驟")
        self.assertEqual(records[0]["content"]["directive"], "")
        self.assertTrue(all(not item["is_augmented"] and item["emotion_target"] == "" for item in records))
        self.assertEqual(set(records[0]["content"]), {"description", "directive"})

    def test_keep_valid_target_writes_normalized_label_without_mutating_input(self):
        frame = pd.DataFrame({"idt_target": [" 個\t人 ", "系統", None, "1905-02"]})
        cleaned = preprocessing.keep_valid_target(frame)
        self.assertEqual(cleaned["idt_target"].tolist(), ["個人", "系統"])
        self.assertEqual(frame.iloc[0]["idt_target"], " 個\t人 ")

    def test_missing_workbook_column_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            workbook = Path(temporary) / "fixture.xlsx"
            pd.DataFrame({"事件描述": ["測試資料"]}).to_excel(workbook, sheet_name="112.01", index=False)
            with self.assertRaisesRegex(ValueError, "缺少欄位"):
                preprocessing.load_and_merge_data(workbook, ["112.01"], preprocessing.COLUMNS)


class DataAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"idt_target": "個人", "emotion_target": "焦慮", "content": {"description": "測試事件甲", "directive": "批示甲"}, "source_id": "112.01:2", "emotion_annotation_model": "old-model", "emotion_annotation_source": "llm_generated", "emotion_annotation_is_gold": False, "emotion_pred": "舊預測", "emotion_prediction_model": "old-model", "emotion_prediction_source": "llm_generated", "emotion_judge": {"overall_score": 5}, "judge_score": 90},
            {"idt_target": "系統", "emotion_target": "", "content": {"description": "測試事件乙", "directive": ""}, "source_id": "112.01:3"},
        ]

    @staticmethod
    def valid_generator(system_prompt, text, **kwargs):
        payload = json.loads(text)
        description = payload["description"]
        return json.dumps({"augmented": [f"{description}改寫一", f"{description}改寫二"]}, ensure_ascii=False)

    def test_fixed_augmentation_preserves_labels_and_clears_generated_emotion(self):
        before = copy.deepcopy(self.records)
        with redirect_stdout(io.StringIO()):
            output = augmentation.augment_records(self.records, n=2, generator=self.valid_generator)
        self.assertEqual(self.records, before)
        self.assertEqual(len(output), 6)
        self.assertEqual([item["idt_target"] for item in output], ["個人", "系統", "個人", "個人", "系統", "系統"])
        self.assertEqual([item["record_id"] for item in output], ["112.01:2", "112.01:3", "112.01:2:aug:1", "112.01:2:aug:2", "112.01:3:aug:1", "112.01:3:aug:2"])
        self.assertEqual(output[0]["emotion_target"], "焦慮")
        self.assertTrue(all(item["emotion_target"] == "" for item in output[2:]))
        self.assertTrue(all(item["is_augmented"] for item in output[2:]))
        self.assertTrue(all("judge_score" not in item for item in output[2:]))
        self.assertTrue(all(not any(field.startswith("emotion_") and field != "emotion_target" for field in item) for item in output[2:]))
        self.assertEqual(output[2]["content"]["directive"], "批示甲")
        self.assertEqual(output[2]["source_id"], output[0]["source_id"])

    def test_zero_augmentation_never_calls_model(self):
        generator = Mock(side_effect=AssertionError("模型不應被呼叫"))
        output = augmentation.augment_records(self.records, n=0, generator=generator)
        self.assertEqual(len(output), 2)
        generator.assert_not_called()

    def test_retries_repeated_or_existing_descriptions(self):
        generator = Mock(side_effect=[
            json.dumps({"augmented": ["測試事件甲", "測試事件乙", "唯一改寫一", "唯一改寫一"]}),
            json.dumps({"augmented": ["唯一改寫一", "唯一改寫二"]}),
        ])
        result = augmentation.paraphrase_text("測試事件甲", n=2, generator=generator, label="個人", avoid_texts=["測試事件乙"])
        self.assertEqual(result, ["唯一改寫一", "唯一改寫二"])
        self.assertEqual(generator.call_count, 2)

    def test_duplicate_and_invalid_outputs_fail_explicitly(self):
        generator = Mock(return_value=json.dumps({"augmented": ["測試事件甲", ""]}))
        with self.assertRaisesRegex(augmentation.AugmentationError, "0/2"):
            augmentation.paraphrase_text("測試事件甲", n=2, max_retries=1, generator=generator)
        self.assertEqual(generator.call_count, 2)
        with self.assertLogs(augmentation.LOGGER, level="WARNING"):
            with self.assertRaisesRegex(augmentation.AugmentationError, "無效 JSON"):
                augmentation.paraphrase_text("測試事件甲", n=1, max_retries=0, generator=lambda *a, **k: '{"augmented": [123]}')

    def test_invalid_or_already_augmented_records_do_not_call_model(self):
        generator = Mock()
        for changes in ({"idt_target": "OR"}, {"is_augmented": True}):
            invalid = {**self.records[0], **changes}
            with self.assertRaises(ValueError):
                augmentation.augment_records([invalid], generator=generator)
        generator.assert_not_called()

    def test_model_failure_does_not_expose_text(self):
        generator = Mock(side_effect=RuntimeError("測試事件甲含個資"))
        with self.assertRaises(augmentation.AugmentationError) as raised:
            augmentation.paraphrase_text("測試事件甲", n=1, generator=generator)
        self.assertNotIn("測試事件甲", str(raised.exception))
        self.assertIn("RuntimeError", str(raised.exception))

    def test_failed_batch_does_not_replace_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "train.json"
            output = Path(temporary) / "augmented.json"
            source.write_text(json.dumps(self.records, ensure_ascii=False), encoding="utf-8")
            output.write_text("previous result", encoding="utf-8")
            with patch.object(augmentation, "augment_records", side_effect=augmentation.AugmentationError("測試失敗")):
                with self.assertRaises(augmentation.AugmentationError):
                    augmentation.run(str(source), str(output))
            self.assertEqual(output.read_text(encoding="utf-8"), "previous result")
            with self.assertRaisesRegex(ValueError, "不可覆寫"):
                augmentation.run(str(source), str(source))


if __name__ == "__main__":
    unittest.main()
