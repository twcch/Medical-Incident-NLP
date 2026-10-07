"""使用合成文字與假模型，驗證情緒評分及模型生命週期。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.config import RuntimeConfig
from src import evaluation, feature_engineering, inference


def valid_judgment():
    return {
        "scores": {
            "emotion_plausibility": 5,
            "evidence_support": 4,
            "writer_perspective": 3,
        },
        "rationale": "描述平鋪直敘；撰寫者情緒仍有推測限制。",
    }


def sample_record(**fields):
    return {"content": {"description": "已記錄流程並完成通報。"}, **fields}


class JudgeValidationTests(unittest.TestCase):
    def test_score_is_computed_from_all_three_dimensions(self):
        raw = valid_judgment()
        raw["overall_score"] = 5
        result = evaluation.parse_judge_response(json.dumps(raw))
        self.assertEqual(result["overall_score"], 4)
        self.assertEqual(set(result["scores"]), set(evaluation.SCORE_DIMENSIONS))

    def test_rejects_invalid_scores_including_bool_and_nonfinite(self):
        for value in (True, "4", None, 0, 6, float("nan"), float("inf")):
            with self.subTest(value=value):
                raw = valid_judgment()
                raw["scores"]["evidence_support"] = value
                with self.assertRaises(ValueError):
                    evaluation.parse_judge_response(json.dumps(raw))

    def test_rejects_incomplete_and_extra_score_dimensions(self):
        missing = valid_judgment()
        missing["scores"].pop("writer_perspective")
        extra = valid_judgment()
        extra["scores"]["accuracy"] = 5
        for raw in (missing, extra):
            with self.assertRaises(ValueError):
                evaluation.parse_judge_response(json.dumps(raw))

    def test_rejects_empty_rationale_extra_fields_and_invalid_overall_score(self):
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
        generator = Mock(side_effect=["不是 JSON", json.dumps(valid_judgment())])
        result = evaluation.judge_emotion(
            "已完成通報。", "中性", generator=generator, model_name="假 Qwen"
        )
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(result["overall_score"], 4)
        self.assertEqual(result["judge_model"], "假 Qwen")

    def test_exhausted_judge_retries_raise_without_default_scores(self):
        generator = Mock(return_value='{"scores": {}, "rationale": "不足"}')
        with self.assertRaisesRegex(RuntimeError, "3 次嘗試"):
            evaluation.judge_emotion(
                "已完成通報。", "中性", generator=generator, model_name="假 Qwen"
            )
        self.assertEqual(generator.call_count, 3)


class AnnotationAndMetricTests(unittest.TestCase):
    def test_annotations_have_model_provenance_and_are_not_gold(self):
        records = [sample_record(idt_target="個人")]
        generator = Mock(return_value='{"emotion": "中性"}')
        feature_engineering.add_emotion_feature(
            records, generator=generator, model_name="假 Gemma"
        )
        self.assertEqual(records[0]["emotion_target"], "中性")
        self.assertEqual(records[0]["emotion_annotation_model"], "假 Gemma")
        self.assertFalse(records[0]["emotion_annotation_is_gold"])

    def test_invalid_emotion_does_not_fall_back_to_neutral(self):
        generator = Mock(return_value='{"emotion": "未知"}')
        with self.assertRaises(RuntimeError):
            feature_engineering.annotate_emotion("已完成通報。", generator=generator)
        self.assertEqual(generator.call_count, 3)

    def test_empty_description_is_rejected_before_generation(self):
        generator = Mock()
        with self.assertRaises(ValueError):
            feature_engineering.add_emotion_feature(
                [{"content": {"description": "  "}}], generator=generator
            )
        generator.assert_not_called()

    def test_invalid_label_response_cannot_contain_explanations_or_extra_fields(self):
        for raw in ('{"emotion": "中性", "reason": "平鋪直敘"}', "中性，因為平鋪直敘"):
            with self.assertRaises(ValueError):
                feature_engineering.parse_label_response(
                    raw, "emotion", feature_engineering.EMOTION_LABELS
                )

    def test_only_valid_idt_gold_labels_enter_accuracy_and_confusion_matrix(self):
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

    def test_emotion_target_is_judged_without_accuracy_or_idt_truth(self):
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


class InferenceLifecycleTests(unittest.TestCase):
    def test_models_are_released_before_the_next_phase_and_emotion_is_regenerated(self):
        events = []
        active = []
        config = RuntimeConfig(gemma_model="假 Gemma", judge_model="假 Qwen")

        class FakeSession:
            def __init__(self, model_id, runtime, adapter_path=None):
                self.model_id = model_id
                self.phase = "idt" if adapter_path is not None else (
                    "judge" if model_id == runtime.judge_model else "emotion"
                )

            def __enter__(self):
                if active:
                    raise AssertionError("同時載入兩個模型")
                active.append(self.phase)
                events.append(("enter", self.phase))
                return self

            def __exit__(self, *args):
                events.append(("exit", self.phase))
                active.pop()

            def generate(self, system_prompt, text, **kwargs):
                if self.phase == "idt":
                    return "個人"
                if self.phase == "emotion":
                    return '{"emotion": "中性"}'
                return json.dumps(valid_judgment())

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text(json.dumps([
                sample_record(idt_target="個人", emotion_target="憤怒")
            ]), encoding="utf-8")
            with patch.object(inference, "ModelSession", FakeSession), patch.object(
                evaluation, "ModelSession", FakeSession
            ):
                summaries = inference.run(str(input_path), directory, config=config)
            predictions = json.loads((Path(directory) / "inference_predictions.json").read_text())
            self.assertEqual(predictions[0]["emotion_pred"], "中性")
            self.assertEqual(predictions[0]["emotion_target"], "憤怒")
            self.assertEqual(summaries[0]["accuracy"], 1)
            self.assertNotIn("accuracy", summaries[1])
            self.assertEqual(summaries[1]["mean_overall_score"], 4)
        self.assertEqual(events, [
            ("enter", "idt"), ("exit", "idt"),
            ("enter", "emotion"), ("exit", "emotion"),
            ("enter", "judge"), ("exit", "judge"),
        ])

    def test_judge_failure_preserves_predictions_and_replaces_stale_evaluation(self):
        config = RuntimeConfig(gemma_model="假 Gemma", judge_model="假 Qwen")
        predictions = [sample_record(idt_pred="個人", emotion_pred="中性")]
        unscored = [{"task": "idt"}, {"task": "emotion", "evaluation_status": "not_scored"}]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text("[]", encoding="utf-8")
            evaluation_path = Path(directory) / "inference_evaluation.json"
            evaluation_path.write_text('[{"accuracy": 1}]', encoding="utf-8")
            with patch.object(inference, "run_inference", return_value=(predictions, unscored)), patch.object(
                inference, "evaluate_emotions", side_effect=RuntimeError("Qwen 失敗")
            ):
                with self.assertRaisesRegex(RuntimeError, "Qwen 失敗"):
                    inference.run(str(input_path), directory, config=config)
            self.assertTrue((Path(directory) / "inference_predictions.json").is_file())
            status = json.loads(evaluation_path.read_text())
            self.assertEqual(status[1]["evaluation_status"], "not_scored")


if __name__ == "__main__":
    unittest.main()
