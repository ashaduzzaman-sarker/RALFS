# ============================================================================
# File: ralfs/utils/io.py
# ============================================================================
"""I/O utilities for JSON/JSONL files with robust error handling."""

import json
from pathlib import Path
from typing import Any, List, Dict, Union
import logging

logger = logging.getLogger(__name__)


def load_json(
    path: Union[Path, str],
    as_jsonl: bool = False,
    encoding: str = "utf-8",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load JSON or JSONL file.
    
    Args:
        path: File path to load
        as_jsonl: If True, treat as JSONL (one JSON object per line)
        encoding: File encoding (default: utf-8)
    
    Returns:
        Loaded data (dict for JSON, list of dicts for JSONL)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        with open(path, "r", encoding=encoding) as f:
            if as_jsonl:
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                return data
            else:
                return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        raise


def save_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    path: Union[Path, str],
    as_jsonl: bool = False,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Save data as JSON or JSONL file.
    
    Args:
        data: Data to save (dict or list of dicts)
        path: Output file path
        as_jsonl: If True, save as JSONL (one JSON object per line)
        encoding: File encoding (default: utf-8)
        indent: Indentation for pretty JSON (ignored for JSONL)
        ensure_ascii: If True, escape non-ASCII characters
    
    Raises:
        TypeError: If data is not JSON-serializable
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, "w", encoding=encoding) as f:
            if as_jsonl:
                if not isinstance(data, list):
                    raise TypeError("Data must be a list for JSONL format")
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
            else:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        
        logger.info(f"Saved data to {path}")
    except Exception as e:
        logger.error(f"Failed to save data to {path}: {e}")
        raise


def load_jsonl(path: Union[Path, str], encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """Convenience function to load JSONL file."""
    return load_json(path, as_jsonl=True, encoding=encoding)


def save_jsonl(
    data: List[Dict[str, Any]],
    path: Union[Path, str],
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
) -> None:
    """Convenience function to save JSONL file."""
    save_json(data, path, as_jsonl=True, encoding=encoding, ensure_ascii=ensure_ascii)
