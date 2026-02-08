from dataclasses import dataclass
from pathlib import Path
import json
import time

@dataclass
class JsonlLogger:
    out_path: Path

    def __post_init__(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict) -> None:
        payload = dict(payload)
        payload["ts"] = time.time()
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
