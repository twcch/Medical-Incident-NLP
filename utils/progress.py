"""長時間批次的進度檔：每完成一筆追加一行 JSON，中斷後可從該處續跑。

第一行記錄本次的設定（模型與參數），續跑時必須一致，避免把不同模型的輸出混在一起。
每筆都帶 fingerprint（識別碼、描述與標籤的雜湊），資料重新切分後舊進度不會被誤用。
"""

import hashlib
import json
from pathlib import Path


def fingerprint(*parts: str | None) -> str:
    """把多個欄位串起來做雜湊，用來確認進度檔中的那筆與目前資料是同一筆。"""
    joined = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


class Progress:
    """JSONL 進度檔；entries 是 key → 已完成的結果。"""

    def __init__(self, path: str | Path, meta: dict, restart: bool = False):
        self.path = Path(path)
        # 經過 JSON 來回，tuple 與 list、int 與 float 的差異才不會造成誤判。
        self.meta = json.loads(json.dumps(meta))
        self.entries: dict[str, dict] = {}
        if restart and self.path.exists():
            self.path.unlink()
        if self.path.exists():
            self._load()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"meta": self.meta}, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        header = json.loads(lines[0]) if lines and lines[0].strip() else {}
        if header.get("meta") != self.meta:
            raise ValueError(
                f"進度檔 {self.path} 的設定與本次不同，請改用相同設定，或加 --restart 重新開始。\n"
                f"進度檔：{header.get('meta')}\n本次：{self.meta}"
            )
        for line in lines[1:]:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # 上次中斷時寫到一半的最後一行，略過即可，那筆會重做。
                continue
            self.entries[entry["key"]] = entry

    def get(self, key: str, fingerprint_value: str) -> dict | None:
        """只有識別碼與內容雜湊都相符才算已完成。"""
        entry = self.entries.get(key)
        if entry is not None and entry.get("fingerprint") == fingerprint_value:
            return entry
        return None

    def add(self, key: str, fingerprint_value: str, **data) -> dict:
        entry = {"key": key, "fingerprint": fingerprint_value, **data}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
        self.entries[key] = entry
        return entry

    def __len__(self) -> int:
        return len(self.entries)
