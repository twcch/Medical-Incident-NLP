"""長時間批次的進度檔：每完成一筆追加一行，中斷後可從該處續跑。"""

import json
from pathlib import Path

if __package__:
    from .json_io import parse_json, save_json
else:
    from json_io import parse_json, save_json


def completion_marker(out_path: str | Path) -> Path:
    """完成標記放在輸出檔旁，記錄產生該輸出時的輸入與設定指紋。"""
    return Path(f"{out_path}.done.json")


def is_complete(out_path: str | Path, header: dict) -> bool:
    """輸出檔與完成標記都在，且標記的指紋與目前相同，代表該階段已完成、可以略過。

    輸入或設定改變、標記損毀或輸出檔被刪除時回傳 False，由呼叫端照常重做。
    """
    marker = completion_marker(out_path)
    if not Path(out_path).is_file() or not marker.is_file():
        return False
    try:
        return parse_json(marker.read_text(encoding="utf-8")) == header
    except ValueError:
        return False


def mark_complete(out_path: str | Path, header: dict) -> None:
    """輸出檔寫完之後才呼叫；先寫標記會讓中斷的階段被誤認為已完成。"""
    save_json(header, completion_marker(out_path))


def format_eta(seconds: float) -> str:
    """把秒數轉成「X 小時 Y 分」，供長時間批次估計剩餘時間。"""
    minutes = int(seconds // 60)
    if minutes < 1:
        return "不到 1 分"
    return f"{minutes // 60} 小時 {minutes % 60} 分" if minutes >= 60 else f"{minutes} 分"


class ProgressFile:
    """第一行是檔頭（輸入與設定的指紋），其後每行是一筆已完成的結果。

    檔頭與目前的輸入或設定不符時拒絕沿用，避免把過期的結果套到新的資料上。
    先 load() 取得先前的進度，再用 with 開啟追加；全部完成後呼叫 remove()。
    """

    def __init__(self, path: str | Path, header: dict):
        self.path = Path(path)
        self.header = header
        # 進度檔裡已完成的紀錄數（含先前載入的），用來判斷中斷時是否真的有進度可續跑。
        self.count = 0
        self._handle = None

    def _is_empty(self) -> bool:
        return not self.path.is_file() or self.path.stat().st_size == 0

    def load(self) -> list[dict]:
        """回傳先前已完成的紀錄（不含檔頭）；沒有進度檔時回傳空清單。"""
        if self._is_empty():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        try:
            stored = parse_json(lines[0])
        except ValueError:
            if len(lines) > 1:
                raise ValueError(
                    f"進度檔 {self.path} 的檔頭損毀；確認不需要後請刪除它再重新執行。"
                ) from None
            # 第一次寫檔就被中斷，只剩殘缺的檔頭：等同沒有進度。
            self.path.unlink()
            return []
        if stored != self.header:
            raise ValueError(
                f"進度檔 {self.path} 與目前的輸入資料或設定不一致；確認不需要後請刪除它再重新執行。"
            )
        entries = []
        for position, line in enumerate(lines[1:], start=2):
            try:
                entries.append(parse_json(line))
            except ValueError:
                if position < len(lines):
                    raise
                # 程序被硬中斷時最後一行可能只寫到一半；從檔案移除，
                # 避免之後追加的內容接在殘缺的文字後面，造成下次續跑時整檔損毀。
                self.path.write_text(
                    "".join(f"{kept}\n" for kept in lines[:-1]), encoding="utf-8", newline="\n"
                )
        self.count = len(entries)
        return entries

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = self._is_empty()
        # 上次若在寫出換行之前就中斷，先補上換行，新的一筆才不會接在舊的一筆後面。
        needs_newline = False
        if not fresh:
            with self.path.open("rb") as existing:
                existing.seek(-1, 2)
                needs_newline = existing.read(1) != b"\n"
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        if needs_newline:
            self._handle.write("\n")
        if fresh:
            self._write(self.header)
        return self

    def _write(self, entry: dict) -> None:
        self._handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._handle.flush()

    def append(self, entry: dict) -> None:
        """寫入一筆並立刻 flush，讓程序被終止時已完成的結果仍在檔案裡。"""
        self._write(entry)
        self.count += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.close()
        self._handle = None
        return False

    def remove(self) -> None:
        """全部完成後刪除，避免下次執行誤用過期的進度。"""
        self.path.unlink(missing_ok=True)
