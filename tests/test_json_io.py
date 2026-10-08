"""驗證不合法資料與寫入失敗不會覆蓋既有實驗結果。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.json_io import load_records, save_json


class JsonIOTests(unittest.TestCase):
    """確認 JSON 格式一致，且失敗時保留既有檔案與清除暫存檔。"""

    def test_chinese_records_round_trip(self):
        """中文內容以 UTF-8 直接保存，建立子目錄後仍可完整讀回。"""
        records = [{"content": {"description": "合成事件描述"}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "records.json"
            save_json(records, path)
            self.assertEqual(load_records(path), records)
            self.assertIn("合成事件描述", path.read_text(encoding="utf-8"))

    def test_invalid_record_shapes_and_nonfinite_values_are_rejected(self):
        """格式錯誤須指出外層或筆數，非有限數值則不可進入資料流程。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            for payload, error in (
                ('{"records": []}', "最外層"),
                ('[{}, "invalid"]', "第 2 筆"),
                ('[{"score": NaN}]', "NaN"),
                ('[{"score": Infinity}]', "Infinity"),
                ('[{"score": 1e400}]', "非有限"),
                ('[{"score": -1e400}]', "非有限"),
            ):
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_records(path)

    def test_serialization_failure_preserves_previous_output(self):
        """序列化失敗時逐位元保留舊結果，不留下部分內容或暫存檔。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            save_json({"experiment": "previous"}, path)
            previous = path.read_bytes()
            for value in (float("nan"), float("inf"), object()):
                with self.subTest(value=type(value).__name__):
                    with self.assertRaises((ValueError, TypeError)):
                        save_json({"score": value}, path)
                    self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_replace_failure_preserves_output_and_removes_temporary_file(self):
        """模擬最後替換失敗，驗證原子寫入的失敗路徑仍保護舊結果。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            save_json({"experiment": "previous"}, path)
            with patch("src.json_io.os.replace", side_effect=OSError("合成寫入失敗")):
                with self.assertRaisesRegex(OSError, "合成寫入失敗"):
                    save_json({"experiment": "new"}, path)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), {"experiment": "previous"}
            )
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
