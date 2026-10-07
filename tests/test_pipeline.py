"""僅用幾筆合成 Excel 資料驗證 CLI 各階段的交接，不讀正式資料。"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run
from src import data_augmentation, evaluation, feature_engineering, inference, trainer


class PipelineTests(unittest.TestCase):
    def test_all_pipeline_uses_updated_sheets_and_judges_fresh_emotion(self):
        sessions = []
        active = []

        class FakeSession:
            def __init__(self, model_id, config, adapter_path=None):
                self.model_id = model_id
                self.adapter_path = adapter_path

            def __enter__(self):
                if active:
                    raise AssertionError("同時載入兩個模型")
                active.append(self)
                sessions.append(self.model_id)
                return self

            def __exit__(self, *args):
                active.pop()

            def generate(self, prompt, text, **kwargs):
                if '"augmented"' in prompt:
                    payload = json.loads(text)
                    return json.dumps({"augmented": [payload["description"] + "（同義改寫）"]})
                if self.adapter_path is not None:
                    return "個人"
                if self.model_id == "fake-qwen":
                    return json.dumps({"scores": {name: 4 for name in evaluation.SCORE_DIMENSIONS},
                                       "rationale": "合成描述沒有明顯情緒。"})
                return '{"emotion": "中性"}'

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "updated.xlsx"
            with pd.ExcelWriter(raw) as writer:
                pd.DataFrame({
                    "IDT分析(個人,系統)": [" 個人 ", "系統"],
                    "事件描述": ["合成事件甲", "合成事件乙"], "批示": ["", ""],
                }).to_excel(writer, sheet_name="new-train", index=False)
                pd.DataFrame({
                    "IDT分析(個人,系統)": ["個人"],
                    "事件描述": ["合成測試事件丙"], "批示": [""],
                }).to_excel(writer, sheet_name="new-test", index=False)
            train_features = root / "features.json"
            result_dir = root / "results"
            with patch.object(data_augmentation, "ModelSession", FakeSession), patch.object(
                feature_engineering, "ModelSession", FakeSession
            ), patch.object(inference, "ModelSession", FakeSession), patch.object(
                evaluation, "ModelSession", FakeSession
            ), patch.object(trainer, "run", return_value="fake-adapter") as train_call, contextlib.redirect_stdout(io.StringIO()):
                summaries = run.main([
                    "all", "--raw-file", str(raw), "--train-sheets", "new-train",
                    "--test-sheets", "new-test", "--train-data", str(root / "train.json"),
                    "--train-augmented", str(root / "augmented.json"),
                    "--train-features", str(train_features), "--test-data", str(root / "test.json"),
                    "--models-dir", str(root / "models"), "--out-dir", str(result_dir),
                    "--augment-n", "1", "--gemma-model", "fake-gemma", "--judge-model", "fake-qwen",
                ])
            features = json.loads(train_features.read_text())
            self.assertEqual(len(features), 4)
            self.assertEqual([record["idt_target"] for record in features], ["個人", "系統", "個人", "系統"])
            self.assertTrue(all(record["emotion_target"] == "中性" for record in features))
            self.assertEqual(train_call.call_args.args, (str(train_features),))
            predictions = json.loads((result_dir / "inference_predictions.json").read_text())
            self.assertEqual(len(predictions), 1)
            self.assertFalse(predictions[0]["is_augmented"])
            self.assertEqual(predictions[0]["source_id"], "new-test:2")
            self.assertEqual(summaries[0]["accuracy"], 1)
            self.assertEqual(summaries[1]["mean_overall_score"], 4)
            self.assertNotIn("accuracy", summaries[1])
            self.assertEqual(sessions, ["fake-gemma"] * 4 + ["fake-qwen"])
            self.assertFalse(active)

    def test_dry_run_does_not_read_data_or_load_models(self):
        output = io.StringIO()
        with patch.object(data_preprocessing := run.data_preprocessing, "run") as load, patch.object(
            trainer, "run"
        ) as train_call, contextlib.redirect_stdout(output):
            run.main(["prepare", "--reuse-prepared", "--dry-run", "--raw-file", "nonexistent.xlsx"])
        load.assert_not_called()
        train_call.assert_not_called()
        self.assertEqual(len(json.loads(output.getvalue())["stages"]), 3)

    def test_overlapping_sheets_are_rejected_before_any_data_work(self):
        with patch.object(run.data_preprocessing, "run") as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run.main(["all", "--train-sheets", "same", "--test-sheets", "same", "--dry-run"])
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
