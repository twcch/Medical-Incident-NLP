"""使用合成文字與假模型，驗證情緒評分及模型生命週期。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src import evaluation, feature_engineering, inference
from src.config import RuntimeConfig


def valid_judgment():
    """產生三面向平均為 4 的固定評分，每次回傳獨立物件以免測試互相影響。"""
    return {
        "scores": {
            "emotion_plausibility": 5,
            "evidence_support": 4,
            "writer_perspective": 3,
        },
        "rationale": "描述平鋪直敘；撰寫者情緒仍有推測限制。",
    }


def sample_record(**fields):
    """用合成描述組合所需欄位，測試不依賴正式醫療資料。"""
    return {"content": {"description": "已記錄流程並完成通報。"}, **fields}


class JudgeValidationTests(unittest.TestCase):
    """檢查模型回應的結構、分數尺度，以及錯誤重試的界線。"""

    def test_score_is_computed_from_all_three_dimensions(self):
        """模型提供的總分與算術平均不符時，仍採用程式計算的平均。"""
        raw = valid_judgment()
        raw["overall_score"] = 5
        result = evaluation.parse_judge_response(json.dumps(raw))
        self.assertEqual(result["overall_score"], 4)
        self.assertEqual(set(result["scores"]), set(evaluation.SCORE_DIMENSIONS))

    def test_rejects_invalid_scores_including_bool_and_nonfinite(self):
        """避免布林值、文字或非有限分數進入評估統計。"""
        for value in (True, "4", None, 0, 6, float("nan"), float("inf")):
            with self.subTest(value=value):
                raw = valid_judgment()
                raw["scores"]["evidence_support"] = value
                with self.assertRaises(ValueError):
                    evaluation.parse_judge_response(json.dumps(raw))

    def test_rejects_incomplete_and_extra_score_dimensions(self):
        """固定三個評分面向，不能藉缺少或新增面向改變總分權重。"""
        missing = valid_judgment()
        missing["scores"].pop("writer_perspective")
        extra = valid_judgment()
        extra["scores"]["accuracy"] = 5
        for raw in (missing, extra):
            with self.assertRaises(ValueError):
                evaluation.parse_judge_response(json.dumps(raw))

    def test_rejects_empty_rationale_extra_fields_and_invalid_overall_score(self):
        """評分理由與欄位結構都需合法，不能夾帶情緒 accuracy。"""
        empty = valid_judgment()
        empty["rationale"] = "  "
        extra = valid_judgment()
        extra["accuracy"] = 1
        overall = valid_judgment()
        overall["overall_score"] = True
        for raw in (empty, extra, overall):
            with self.assertRaises(ValueError):
                evaluation.parse_judge_response(json.dumps(raw))

    def test_retries_invalid_judge_output_then_returns_valid_score(self):
        """假模型先失敗後成功，確認流程可重試且留下評審模型來源。"""
        generator = Mock(side_effect=["不是 JSON", json.dumps(valid_judgment())])
        result = evaluation.judge_emotion(
            "已完成通報。", "中性", generator=generator, model_name="假 Qwen"
        )
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(result["overall_score"], 4)
        self.assertEqual(result["judge_model"], "假 Qwen")

    def test_exhausted_judge_retries_raise_without_default_scores(self):
        """持續無效的回應必須明確失敗，不能填入預設分數掩蓋問題。"""
        generator = Mock(return_value='{"scores": {}, "rationale": "不足"}')
        with self.assertRaisesRegex(RuntimeError, "3 次嘗試"):
            evaluation.judge_emotion(
                "已完成通報。", "中性", generator=generator, model_name="假 Qwen"
            )
        self.assertEqual(generator.call_count, 3)

    def test_invalid_token_limit_is_rejected_before_generation(self):
        """生成上限不合法時就停止，避免錯誤參數傳入模型。"""
        generator = Mock()
        for limit in (True, 0, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                evaluation.judge_emotion(
                    "已完成通報。",
                    "中性",
                    generator=generator,
                    model_name="假 Qwen",
                    max_new_tokens=limit,
                )
        generator.assert_not_called()


class AnnotationAndMetricTests(unittest.TestCase):
    """區分人工 IDT 真值與模型情緒標註，驗證統計與資料回填規則。"""

    def test_annotations_have_model_provenance_and_are_not_gold(self):
        """情緒標註保留模型來源，且明確標示不屬於人工真值。"""
        records = [sample_record(idt_target="個人")]
        generator = Mock(return_value='{"emotion": "中性"}')
        feature_engineering.add_emotion_feature(records, generator=generator, model_name="假 Gemma")
        self.assertEqual(records[0]["emotion_target"], "中性")
        self.assertEqual(records[0]["emotion_annotation_model"], "假 Gemma")
        self.assertFalse(records[0]["emotion_annotation_is_gold"])

    def test_invalid_emotion_does_not_fall_back_to_neutral(self):
        """未知情緒標籤不能被靜默替換成中性。"""
        generator = Mock(return_value='{"emotion": "未知"}')
        with self.assertRaises(RuntimeError):
            feature_engineering.annotate_emotion("已完成通報。", generator=generator)
        self.assertEqual(generator.call_count, 3)

    def test_empty_description_is_rejected_before_generation(self):
        """空白事件描述屬於資料錯誤，不能交由模型補出標註。"""
        generator = Mock()
        with self.assertRaises(ValueError):
            feature_engineering.add_emotion_feature(
                [{"content": {"description": "  "}}], generator=generator
            )
        generator.assert_not_called()

    def test_invalid_label_response_cannot_contain_explanations_or_extra_fields(self):
        """要求單一標籤的流程不能接受附帶說明或額外 JSON 欄位。"""
        for raw in ('{"emotion": "中性", "reason": "平鋪直敘"}', "中性，因為平鋪直敘"):
            with self.assertRaises(ValueError):
                feature_engineering.parse_label_response(
                    raw, "emotion", feature_engineering.EMOTION_LABELS
                )

    def test_only_valid_idt_gold_labels_enter_accuracy_and_confusion_matrix(self):
        """用有／無人工標籤的混合資料核對 accuracy 分母與矩陣方向。"""
        records = [
            sample_record(idt_target="個人", idt_pred="個人"),
            sample_record(idt_target="系統", idt_pred="個人"),
            sample_record(idt_target="其他", idt_pred="系統"),
            sample_record(idt_pred="系統"),
        ]
        result = evaluation.evaluate_idt(records)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["confusion_matrix"]["系統"]["個人"], 1)
        self.assertEqual(result["n_without_valid_target"], 2)

    def test_idt_without_comparable_labels_has_no_accuracy(self):
        """沒有有效比較資料時回傳 None，並保留可重疊的缺漏計數。"""
        records = [
            sample_record(),
            sample_record(idt_target="個人"),
            sample_record(idt_target="其他", idt_pred="系統"),
        ]
        result = evaluation.evaluate_idt(records)
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["n_missing_prediction"], 2)
        self.assertEqual(result["n_without_valid_target"], 2)
        self.assertTrue(
            all(count == 0 for row in result["confusion_matrix"].values() for count in row.values())
        )

    def test_invalid_idt_prediction_is_rejected_even_without_gold_label(self):
        """缺少真值不代表可放過錯誤的模型預測格式。"""
        with self.assertRaisesRegex(ValueError, "第 1 筆 idt_pred"):
            evaluation.evaluate_idt([sample_record(idt_pred="未知")])

    def test_emotion_target_is_judged_without_accuracy_or_idt_truth(self):
        """舊 emotion_target 僅作為待評分模型標註，不需要 IDT 真值。"""
        records = [sample_record(emotion_target="焦慮")]
        generator = Mock(return_value=json.dumps(valid_judgment()))
        result = evaluation.evaluate_emotions(
            records, generator=generator, prediction_field="emotion_target"
        )
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["mean_overall_score"], 4)
        self.assertNotIn("accuracy", result)
        self.assertEqual(records[0]["emotion_judge"]["prediction_field"], "emotion_target")
        payload = json.loads(generator.call_args.args[1])
        self.assertEqual(payload["emotion"], "焦慮")

    def test_all_emotion_inputs_are_validated_before_loading_a_model(self):
        """後段輸入無效時，前段也不應載模型或留下半批評分。"""
        records = [sample_record(emotion_pred="中性"), sample_record(emotion_pred="未知")]
        with patch.object(evaluation, "ModelSession") as session:
            with self.assertRaisesRegex(ValueError, "第 2 筆 emotion_pred"):
                evaluation.evaluate_emotions(records)
        session.assert_not_called()
        self.assertNotIn("emotion_judge", records[0])

    def test_invalid_evaluation_options_are_rejected_before_loading_a_model(self):
        """先攔截無效評分選項，避免發生不必要的模型載入。"""
        for options in (
            {"max_retries": -1},
            {"max_retries": True},
            {"prediction_field": " "},
            {"prediction_field": []},
            {"generator": "無效生成器"},
        ):
            with self.subTest(options=options), patch.object(evaluation, "ModelSession") as session:
                with self.assertRaises(ValueError):
                    evaluation.evaluate_emotions([sample_record(emotion_pred="中性")], **options)
                session.assert_not_called()

    def test_failed_batch_does_not_replace_any_existing_emotion_judgment(self):
        """第二筆評分失敗時，第一筆原有評分也必須完整保留。"""
        records = [
            sample_record(emotion_pred="中性", emotion_judge={"overall_score": 2}),
            sample_record(emotion_pred="中性"),
        ]
        original = copy.deepcopy(records)
        generator = Mock(side_effect=[json.dumps(valid_judgment()), "不是 JSON"])
        with self.assertRaisesRegex(RuntimeError, "第 2 筆情緒評分失敗"):
            evaluation.evaluate_emotions(records, generator=generator, max_retries=0)
        self.assertEqual(records, original)

    def test_model_session_is_released_after_judge_failure(self):
        """即使評分失敗，context manager 仍須執行模型釋放。"""
        with patch.object(evaluation, "ModelSession") as session:
            session.return_value.__enter__.return_value.generate.return_value = "不是 JSON"
            with self.assertRaises(RuntimeError):
                evaluation.evaluate_emotions([sample_record(emotion_pred="中性")], max_retries=0)
        session.assert_called_once()
        session.return_value.__exit__.assert_called_once()


class InferenceLifecycleTests(unittest.TestCase):
    """用假 session 與暫存檔檢查模型順序、空資料及評分失敗後的輸出狀態。"""

    def test_empty_pipeline_writes_empty_results_without_loading_models(self):
        """空資料仍產生可讀的結果檔，且不載入 Gemma 或 Qwen。"""
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text("[]", encoding="utf-8")
            with (
                patch.object(inference, "ModelSession") as gemma,
                patch.object(evaluation, "ModelSession") as qwen,
            ):
                summaries = inference.run(input_path, directory)
            predictions = json.loads(
                (Path(directory) / "inference_predictions.json").read_text(encoding="utf-8")
            )
        gemma.assert_not_called()
        qwen.assert_not_called()
        self.assertEqual(predictions, [])
        self.assertIsNone(summaries[0]["accuracy"])
        self.assertIsNone(summaries[1]["mean_overall_score"])
        self.assertEqual(summaries[1]["n"], 0)

    def test_invalid_description_is_rejected_before_loading_an_adapter(self):
        """推論前完整驗證描述，避免為錯誤資料載入 IDT adapter。"""
        records = [sample_record(), {"content": {"description": "  "}}]
        with patch.object(inference, "ModelSession") as session:
            with self.assertRaisesRegex(ValueError, "第 2 筆"):
                inference.run_inference(records)
        session.assert_not_called()

    def test_models_are_released_before_the_next_phase_and_emotion_is_regenerated(self):
        """記錄假模型進出順序，確認單一模型記憶體占用與情緒重新推論。"""
        events = []
        active = []
        config = RuntimeConfig(gemma_model="假 Gemma", judge_model="假 Qwen")

        class FakeSession:
            """以固定回應取代真模型，並偵測是否同時開啟兩個模型階段。"""

            def __init__(self, model_id, runtime, adapter_path=None):
                """依 adapter 與模型 ID 判斷目前是 IDT、情緒推論或評分。"""
                self.model_id = model_id
                self.phase = (
                    "idt"
                    if adapter_path is not None
                    else ("judge" if model_id == runtime.judge_model else "emotion")
                )

            def __enter__(self):
                """進入階段前確認上一個模型已退出，並保存生命週期事件。"""
                if active:
                    raise AssertionError("同時載入兩個模型")
                active.append(self.phase)
                events.append(("enter", self.phase))
                return self

            def __exit__(self, *args):
                """模擬釋放模型，供下一個階段的互斥檢查使用。"""
                events.append(("exit", self.phase))
                active.pop()

            def generate(self, system_prompt, text, **kwargs):
                """各階段回傳合法固定輸出，讓測試集中在流程與來源欄位。"""
                if self.phase == "idt":
                    return "個人"
                if self.phase == "emotion":
                    return '{"emotion": "中性"}'
                return json.dumps(valid_judgment())

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text(
                json.dumps([sample_record(idt_target="個人", emotion_target="憤怒")]),
                encoding="utf-8",
            )
            with (
                patch.object(inference, "ModelSession", FakeSession),
                patch.object(evaluation, "ModelSession", FakeSession),
            ):
                summaries = inference.run(str(input_path), directory, config=config)
            predictions = json.loads((Path(directory) / "inference_predictions.json").read_text())
            self.assertEqual(predictions[0]["emotion_pred"], "中性")
            self.assertEqual(predictions[0]["emotion_target"], "憤怒")
            self.assertEqual(summaries[0]["accuracy"], 1)
            self.assertNotIn("accuracy", summaries[1])
            self.assertEqual(summaries[1]["mean_overall_score"], 4)
        self.assertEqual(
            events,
            [
                ("enter", "idt"),
                ("exit", "idt"),
                ("enter", "emotion"),
                ("exit", "emotion"),
                ("enter", "judge"),
                ("exit", "judge"),
            ],
        )

    def test_judge_failure_preserves_predictions_and_replaces_stale_evaluation(self):
        """Qwen 失敗後保留 Gemma 預測，並以 not_scored 取代上一輪評估。"""
        config = RuntimeConfig(gemma_model="假 Gemma", judge_model="假 Qwen")
        predictions = [sample_record(idt_pred="個人", emotion_pred="中性")]
        unscored = [{"task": "idt"}, {"task": "emotion", "evaluation_status": "not_scored"}]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text("[]", encoding="utf-8")
            evaluation_path = Path(directory) / "inference_evaluation.json"
            evaluation_path.write_text('[{"accuracy": 1}]', encoding="utf-8")
            with (
                patch.object(inference, "run_inference", return_value=(predictions, unscored)),
                patch.object(inference, "evaluate_emotions", side_effect=RuntimeError("Qwen 失敗")),
            ):
                with self.assertRaisesRegex(RuntimeError, "Qwen 失敗"):
                    inference.run(str(input_path), directory, config=config)
            self.assertTrue((Path(directory) / "inference_predictions.json").is_file())
            status = json.loads(evaluation_path.read_text())
            self.assertEqual(status[1]["evaluation_status"], "not_scored")


if __name__ == "__main__":
    unittest.main()
