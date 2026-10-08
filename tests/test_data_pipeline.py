"""使用人工 fixture 與假模型驗證資料流程，不讀取研究資料或下載模型。"""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src import data_augmentation as augmentation
from src import data_preprocessing as preprocessing
from src import feature_engineering as features


class DataPreprocessingTests(unittest.TestCase):
    """以臨時 Excel 與 DataFrame 驗證清洗順序、來源追蹤及輸入限制。"""

    def test_excel_cleaning_and_provenance(self):
        """混合各種待清洗狀況，核對剩餘筆數、統計與原始 Excel 列號。"""
        # 同一份合成資料包含全形空白、非法標籤、空描述及換行格式不同的重複。
        frame = pd.DataFrame(
            {
                " IDT分析(個人,系統) ": [
                    " 個　人 ",
                    "系統",
                    "OR",
                    "個人",
                    " 系統 ",
                    "系統",
                    "個人",
                ],
                "事件描述": [
                    " 測試事件甲 ",
                    "測試事件乙",
                    "無效",
                    "  ",
                    "測試事件乙",
                    "測試事件丙",
                    "測試事件甲",
                ],
                "批示": [None, " 處理\r\n步驟 ", None, None, "處理\n步驟", None, None],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            workbook = Path(temporary) / "fixture.xlsx"
            output = Path(temporary) / "clean.json"
            frame.to_excel(workbook, sheet_name="112.01", index=False)
            with redirect_stdout(io.StringIO()):
                cleaned = preprocessing.run(["112.01"], str(output), raw_file=workbook)
            records = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            cleaned.attrs["cleaning_stats"],
            {
                "input_records": 7,
                "invalid_targets": 1,
                "empty_descriptions": 1,
                "duplicates": 2,
                "output_records": 3,
            },
        )
        self.assertEqual([item["idt_target"] for item in records], ["個人", "系統", "系統"])
        self.assertEqual(
            [item["source_id"] for item in records], ["112.01:2", "112.01:3", "112.01:7"]
        )
        self.assertEqual(records[1]["content"]["directive"], "處理\n步驟")
        self.assertEqual(records[0]["content"]["directive"], "")
        self.assertTrue(
            all(not item["is_augmented"] and item["emotion_target"] == "" for item in records)
        )
        self.assertEqual(set(records[0]["content"]), {"description", "directive"})

    def test_keep_valid_target_writes_normalized_label_without_mutating_input(self):
        """合法標籤在副本中正規化，呼叫端保留原字串與原始資料。"""
        frame = pd.DataFrame({"idt_target": [" 個\t人 ", "系統", None, "1905-02"]})
        cleaned = preprocessing.keep_valid_target(frame)
        self.assertEqual(cleaned["idt_target"].tolist(), ["個人", "系統"])
        self.assertEqual(frame.iloc[0]["idt_target"], " 個\t人 ")

    def test_missing_workbook_column_is_reported(self):
        """不完整的工作表要明確報告缺欄，不能產生缺少研究內容的輸出。"""
        with tempfile.TemporaryDirectory() as temporary:
            workbook = Path(temporary) / "fixture.xlsx"
            pd.DataFrame({"事件描述": ["測試資料"]}).to_excel(
                workbook, sheet_name="112.01", index=False
            )
            with self.assertRaisesRegex(ValueError, "缺少欄位"):
                preprocessing.load_and_merge_data(workbook, ["112.01"], preprocessing.COLUMNS)

    def test_invalid_selection_is_rejected_before_opening_workbook(self):
        """錯誤名稱序列在任何 Excel I/O 前被拒絕。"""
        selections = (
            ([], preprocessing.COLUMNS),
            ("112.01", preprocessing.COLUMNS),
            (["112.01", "112.01"], preprocessing.COLUMNS),
            (["112.01"], ["事件描述", "事件描述"]),
            (["112.01"], ["_source_sheet"]),
        )
        with patch.object(preprocessing.pd, "ExcelFile") as open_workbook:
            for sheets, columns in selections:
                with self.subTest(sheets=sheets, columns=columns):
                    with self.assertRaises(ValueError):
                        preprocessing.load_and_merge_data("unused.xlsx", sheets, columns)
        open_workbook.assert_not_called()

    def test_rename_rejects_duplicate_or_reserved_names(self):
        """新欄名不得重複或佔用來源欄位，拒絕時原 DataFrame 維持不變。"""
        frame = pd.DataFrame({"標籤": ["個人"], "描述": ["合成事件"]})
        for names in (["idt_target", "idt_target"], ["idt_target", "_source_sheet"]):
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    preprocessing.rename_columns(frame, names)
        self.assertEqual(frame.columns.tolist(), ["標籤", "描述"])

    def test_json_fallback_ids_follow_output_order_for_non_numeric_index(self):
        """沒有 Excel 來源時依輸出順序編號，不依賴 DataFrame 索引型別。"""
        frame = pd.DataFrame(
            {"idt_target": ["個人", "系統"], "description": ["合成事件甲", "合成事件乙"]},
            index=["custom-a", "custom-b"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "clean.json"
            preprocessing.save_data_to_json(frame, output)
            records = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual([record["source_id"] for record in records], ["row:1", "row:2"])

    def test_partial_provenance_does_not_replace_existing_output(self):
        """來源欄位不完整時保留舊輸出，不以 fallback ID 掩蓋來源缺漏。"""
        frame = pd.DataFrame(
            {
                "idt_target": ["個人"],
                "description": ["合成事件"],
                "_source_sheet": ["112.01"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "clean.json"
            output.write_text("previous result", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "必須同時存在"):
                preprocessing.save_data_to_json(frame, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "previous result")


class DataAugmentationTests(unittest.TestCase):
    """以固定回應的假模型驗證增生份數、去重、標籤保留與失敗保護。"""

    def setUp(self):
        """建立兩筆合成原文，其中一筆刻意附帶需在改寫後清除的舊情緒資訊。"""
        self.records = [
            {
                "idt_target": "個人",
                "emotion_target": "焦慮",
                "content": {"description": "測試事件甲", "directive": "批示甲"},
                "source_id": "112.01:2",
                "emotion_annotation_model": "old-model",
                "emotion_annotation_source": "llm_generated",
                "emotion_annotation_is_gold": False,
                "emotion_pred": "舊預測",
                "emotion_prediction_model": "old-model",
                "emotion_prediction_source": "llm_generated",
                "emotion_judge": {"overall_score": 5},
                "judge_score": 90,
            },
            {
                "idt_target": "系統",
                "emotion_target": "",
                "content": {"description": "測試事件乙", "directive": ""},
                "source_id": "112.01:3",
            },
        ]

    @staticmethod
    def valid_generator(system_prompt, text, **kwargs):
        """模擬生成介面，依輸入描述回傳可預期的兩個版本，不載入真實模型。"""
        payload = json.loads(text)
        description = payload["description"]
        return json.dumps(
            {"augmented": [f"{description}改寫一", f"{description}改寫二"]}, ensure_ascii=False
        )

    def test_fixed_augmentation_preserves_labels_and_clears_generated_emotion(self):
        """核對原文與改寫的順序、來源關係，以及改寫後舊標註不被沿用。"""
        before = copy.deepcopy(self.records)
        with redirect_stdout(io.StringIO()):
            output = augmentation.augment_records(self.records, n=2, generator=self.valid_generator)
        self.assertEqual(self.records, before)
        self.assertEqual(len(output), 6)
        self.assertEqual(
            [item["idt_target"] for item in output],
            ["個人", "系統", "個人", "個人", "系統", "系統"],
        )
        self.assertEqual(
            [item["record_id"] for item in output],
            [
                "112.01:2",
                "112.01:3",
                "112.01:2:aug:1",
                "112.01:2:aug:2",
                "112.01:3:aug:1",
                "112.01:3:aug:2",
            ],
        )
        self.assertEqual(output[0]["emotion_target"], "焦慮")
        self.assertTrue(all(item["emotion_target"] == "" for item in output[2:]))
        self.assertTrue(all(item["is_augmented"] for item in output[2:]))
        self.assertTrue(all("judge_score" not in item for item in output[2:]))
        self.assertTrue(
            all(
                not any(
                    field.startswith("emotion_") and field != "emotion_target" for field in item
                )
                for item in output[2:]
            )
        )
        self.assertEqual(output[2]["content"]["directive"], "批示甲")
        self.assertEqual(output[2]["source_id"], output[0]["source_id"])

    def test_zero_augmentation_never_calls_model(self):
        """增生份數為零時只返回原文副本，不消耗任何模型生成資源。"""
        generator = Mock(side_effect=AssertionError("模型不應被呼叫"))
        output = augmentation.augment_records(self.records, n=0, generator=generator)
        self.assertEqual(len(output), 2)
        generator.assert_not_called()

    def test_retries_repeated_or_existing_descriptions(self):
        """保留首次取得的唯一版本，重試時只補足並排除原文與其他既有描述。"""
        generator = Mock(
            side_effect=[
                json.dumps({"augmented": ["測試事件甲", "測試事件乙", "唯一改寫一", "唯一改寫一"]}),
                json.dumps({"augmented": ["唯一改寫一", "唯一改寫二"]}),
            ]
        )
        result = augmentation.paraphrase_text(
            "測試事件甲", n=2, generator=generator, label="個人", avoid_texts=["測試事件乙"]
        )
        self.assertEqual(result, ["唯一改寫一", "唯一改寫二"])
        self.assertEqual(generator.call_count, 2)

    def test_duplicate_and_invalid_outputs_fail_explicitly(self):
        """空值、重複內容及不合法 JSON 結構耗盡重試後都必須顯式失敗。"""
        generator = Mock(return_value=json.dumps({"augmented": ["測試事件甲", ""]}))
        with self.assertRaisesRegex(augmentation.AugmentationError, "0/2"):
            augmentation.paraphrase_text("測試事件甲", n=2, max_retries=1, generator=generator)
        self.assertEqual(generator.call_count, 2)
        with self.assertLogs(augmentation.LOGGER, level="WARNING"):
            with self.assertRaisesRegex(augmentation.AugmentationError, "無效 JSON"):
                augmentation.paraphrase_text(
                    "測試事件甲",
                    n=1,
                    max_retries=0,
                    generator=lambda *a, **k: '{"augmented": [123]}',
                )

    def test_invalid_or_already_augmented_records_do_not_call_model(self):
        """拒絕非法人工標籤與二次增生輸入，並確認驗證失敗前未呼叫模型。"""
        generator = Mock()
        for changes in ({"idt_target": "OR"}, {"is_augmented": True}):
            invalid = {**self.records[0], **changes}
            with self.assertRaises(ValueError):
                augmentation.augment_records([invalid], generator=generator)
        generator.assert_not_called()

    def test_non_text_directive_is_rejected_including_falsy_values(self):
        """0、False 及空列表都不是合法批示，不能被當成空字串吞掉。"""
        generator = Mock()
        for directive in (0, False, []):
            with self.subTest(directive=directive):
                record = copy.deepcopy(self.records[0])
                record["content"]["directive"] = directive
                with self.assertRaisesRegex(ValueError, "directive 必須是字串"):
                    augmentation.augment_records([record], generator=generator)
        generator.assert_not_called()

    def test_invalid_excluded_texts_are_rejected_before_model_call(self):
        """排除集合必須是字串序列，單一字串不可被逐字拆成排除項目。"""
        generator = Mock()
        for excluded in ("合成事件", ["合成事件", 123], None):
            with self.subTest(excluded=excluded):
                with self.assertRaisesRegex(ValueError, "avoid_texts"):
                    augmentation.paraphrase_text(
                        "合成原文", n=1, generator=generator, avoid_texts=excluded
                    )
        generator.assert_not_called()

    def test_model_failure_does_not_expose_text(self):
        """模型例外只留下錯誤型別，對外訊息不能夾帶輸入敘述。"""
        generator = Mock(side_effect=RuntimeError("測試事件甲含個資"))
        with self.assertRaises(augmentation.AugmentationError) as raised:
            augmentation.paraphrase_text("測試事件甲", n=1, generator=generator)
        self.assertNotIn("測試事件甲", str(raised.exception))
        self.assertIn("RuntimeError", str(raised.exception))

    def test_failed_batch_does_not_replace_output(self):
        """整批增生失敗保留既有輸出，且輸出路徑不得覆寫原始訓練資料。"""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "train.json"
            output = Path(temporary) / "augmented.json"
            source.write_text(json.dumps(self.records, ensure_ascii=False), encoding="utf-8")
            output.write_text("previous result", encoding="utf-8")
            with patch.object(
                augmentation,
                "augment_records",
                side_effect=augmentation.AugmentationError("測試失敗"),
            ):
                with self.assertRaises(augmentation.AugmentationError):
                    augmentation.run(str(source), str(output))
            self.assertEqual(output.read_text(encoding="utf-8"), "previous result")
            with self.assertRaisesRegex(ValueError, "不可覆寫"):
                augmentation.run(str(source), str(source))


class EmotionFeatureTests(unittest.TestCase):
    """驗證情緒標註的整批更新與模型來源資訊，不把模型標籤視為人工真值。"""

    def test_failed_annotation_batch_preserves_all_input_records(self):
        """第一筆成功、第二筆耗盡重試時，整批原始標註仍保持原樣。"""
        records = [
            {"content": {"description": "合成事件甲"}, "emotion_target": "原標註甲"},
            {"content": {"description": "合成事件乙"}, "emotion_target": "原標註乙"},
        ]
        before = copy.deepcopy(records)
        generator = Mock(side_effect=["中性", "無效標籤", "無效標籤", "無效標籤"])
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                features.add_emotion_feature(records, generator=generator)
        self.assertEqual(records, before)
        self.assertEqual(generator.call_count, 4)

    def test_successful_annotation_retains_in_place_api_and_source_metadata(self):
        """成功後仍回傳同一個列表，並同時補齊情緒與模型來源欄位。"""
        records = [{"content": {"description": "合成事件"}, "source_id": "112.01:2"}]
        with redirect_stdout(io.StringIO()):
            result = features.add_emotion_feature(
                records, generator=Mock(return_value='{"emotion": "中性"}'), model_name="fake-model"
            )
        self.assertIs(result, records)
        self.assertEqual(records[0]["emotion_target"], "中性")
        self.assertEqual(records[0]["emotion_annotation_model"], "fake-model")
        self.assertEqual(records[0]["emotion_annotation_source"], "llm_generated")
        self.assertFalse(records[0]["emotion_annotation_is_gold"])
        self.assertEqual(records[0]["source_id"], "112.01:2")


if __name__ == "__main__":
    unittest.main()
