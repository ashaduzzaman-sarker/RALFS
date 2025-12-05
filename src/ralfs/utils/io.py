# src/ralfs/utils/io.py
import json
from pathlib import Path
from typing import Any

def save_json(data: Any, path: Path | str, as_jsonl: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if not as_jsonl else "w"
    with open(path, mode, encoding="utf-8") as f:
        if as_jsonl:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)

