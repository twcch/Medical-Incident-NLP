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
    """確認 CLI 選項能傳遞到各階段，且訓練、測試與評分不混用資料。"""

    def test_all_pipeline_uses_updated_sheets_and_judges_fresh_emotion(self):
        """用新工作表與假模型走完流程，核對來源、固定增生及模型載入順序。"""
        sessions = []
        active = []

        class FakeSession:
            """記錄模型進出，若前一個模型尚未釋放就載入下一個則立即失敗。"""

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
                """依任務回傳合成答案，讓流程測試不依賴模型或網路。"""
                if '"augmented"' in prompt:
                    payload = json.loads(text)
                    return json.dumps({"augmented": [payload["description"] + "（同義改寫）"]})
                if self.adapter_path is not None:
                    return "個人"
                if self.model_id == "fake-qwen":
                    return json.dumps(
                        {
                            "scores": {name: 4 for name in evaluation.SCORE_DIMENSIONS},
                            "rationale": "合成描述沒有明顯情緒。",
                        }
                    )
                return '{"emotion": "中性"}'

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "updated.xlsx"
            # 月份刻意不同於預設值，確認 CLI 真正採用使用者指定的工作表。
            with pd.ExcelWriter(raw) as writer:
                pd.DataFrame(
                    {
                        "IDT分析(個人,系統)": [" 個人 ", "系統"],
                        "事件描述": ["合成事件甲", "合成事件乙"],
                        "批示": ["", ""],
                    }
                ).to_excel(writer, sheet_name="new-train", index=False)
                pd.DataFrame(
                    {
                        "IDT分析(個人,系統)": ["個人"],
                        "事件描述": ["合成測試事件丙"],
                        "批示": [""],
                    }
                ).to_excel(writer, sheet_name="new-test", index=False)
            train_features = root / "features.json"
            result_dir = root / "results"
            # 只替換模型與昂貴訓練；Excel 清洗、JSON 交接與評估仍使用真實程式。
            with (
                patch.object(data_augmentation, "ModelSession", FakeSession),
                patch.object(feature_engineering, "ModelSession", FakeSession),
                patch.object(inference, "ModelSession", FakeSession),
                patch.object(evaluation, "ModelSession", FakeSession),
                patch.object(trainer, "run", return_value="fake-adapter") as train_call,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summaries = run.main(
                    [
                        "all",
                        "--raw-file",
                        str(raw),
                        "--train-sheets",
                        "new-train",
                        "--test-sheets",
                        "new-test",
                        "--train-data",
                        str(root / "train.json"),
                        "--train-augmented",
                        str(root / "augmented.json"),
                        "--train-features",
                        str(train_features),
                        "--test-data",
                        str(root / "test.json"),
                        "--models-dir",
                        str(root / "models"),
                        "--out-dir",
                        str(result_dir),
                        "--augment-n",
                        "1",
                        "--gemma-model",
                        "fake-gemma",
                        "--judge-model",
                        "fake-qwen",
                    ]
                )
            features = json.loads(train_features.read_text())
            # 兩筆訓練原文各新增一筆；測試資料仍只有一筆且沒有增生。
            self.assertEqual(len(features), 4)
            self.assertEqual(
                [record["idt_target"] for record in features], ["個人", "系統", "個人", "系統"]
            )
            self.assertTrue(all(record["emotion_target"] == "中性" for record in features))
            self.assertEqual(train_call.call_args.args, (str(train_features),))
            predictions = json.loads((result_dir / "inference_predictions.json").read_text())
            self.assertEqual(len(predictions), 1)
            self.assertFalse(predictions[0]["is_augmented"])
            self.assertEqual(predictions[0]["source_id"], "new-test:2")
            self.assertEqual(summaries[0]["accuracy"], 1)
            self.assertEqual(summaries[1]["mean_overall_score"], 4)
            self.assertNotIn("accuracy", summaries[1])
            # 四次 Gemma 依序為增生、訓練情緒、IDT 推論、測試情緒，最後才載入 Qwen。
            self.assertEqual(sessions, ["fake-gemma"] * 4 + ["fake-qwen"])
            self.assertFalse(active)

    def test_dry_run_does_not_read_data_or_load_models(self):
        """即使原始檔不存在，預覽也應成功，prepare 的重用旗標不影響階段。"""
        output = io.StringIO()
        with (
            patch.object(run.data_preprocessing, "run") as load,
            patch.object(trainer, "run") as train_call,
            contextlib.redirect_stdout(output),
        ):
            run.main(["prepare", "--reuse-prepared", "--dry-run", "--raw-file", "nonexistent.xlsx"])
        load.assert_not_called()
        train_call.assert_not_called()
        self.assertEqual(len(json.loads(output.getvalue())["stages"]), 3)

    def test_reused_all_pipeline_still_predicts_emotion_and_can_skip_judging(self):
        """重用資料仍須產生新預測，但可將 Qwen 評分留給獨立 evaluate。"""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run.main(["all", "--reuse-prepared", "--skip-emotion-evaluation", "--dry-run"])
        self.assertEqual(
            json.loads(output.getvalue())["stages"],
            [
                "IDT LoRA 訓練",
                "IDT LoRA 推論",
                "Gemma 情緒標註",
            ],
        )

    def test_invalid_training_numbers_fail_before_any_pipeline_work(self):
        """非法數值應以 CLI 用法錯誤退出，不先讀取資料或產生中間結果。"""
        with (
            patch.object(run.data_preprocessing, "run") as load,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for option, value in (
                ("--epochs", "nan"),
                ("--learning-rate", "inf"),
                ("--max-new-tokens", "0"),
            ):
                with (
                    self.subTest(option=option, value=value),
                    self.assertRaises(SystemExit) as raised,
                ):
                    run.main(["all", option, value, "--dry-run"])
                self.assertEqual(raised.exception.code, 2)
        load.assert_not_called()

    def test_explicit_empty_sheets_do_not_fall_back_to_default_months(self):
        """空清單代表來源未指定完整，不應擅自改讀預設訓練或測試月份。"""
        with (
            patch.object(run.data_augmentation, "run") as augment,
            patch.object(run.predictor, "run") as predict,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(ValueError, "工作表"):
                run.prepare(train_sheets=[])
            with self.assertRaisesRegex(ValueError, "工作表"):
                run.inference(test_sheets=[])
        augment.assert_not_called()
        predict.assert_not_called()

    def test_overlapping_sheets_are_rejected_before_any_data_work(self):
        """月份重疊時先阻止執行，避免整個工作表同時成為訓練與測試資料。"""
        with (
            patch.object(run.data_preprocessing, "run") as load,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit):
                run.main(["all", "--train-sheets", "same", "--test-sheets", "same", "--dry-run"])
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
