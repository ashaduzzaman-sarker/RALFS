# src/ralfs/utils/io.py
import json
from pathlib import Path
from typing import Any

def load_json(path: Path | str, as_jsonl: bool = False) -> Any:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        if as_jsonl:
            return [json.loads(line) for line in f]
        else:
            return json.load(f)

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

