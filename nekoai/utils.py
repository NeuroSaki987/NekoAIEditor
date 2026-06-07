from __future__ import annotations

import json
import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", (text or "").strip())[:120] or "model"


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def zip_dir(src_dir: str | os.PathLike[str], zip_path: str | os.PathLike[str]) -> str:
    src = Path(src_dir).resolve()
    out = Path(zip_path).resolve()
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _dirs, files in os.walk(src):
            for name in files:
                f = Path(root) / name
                zf.write(f, f.relative_to(src.parent))
    return str(out)


@dataclass
class EditRecord:
    timestamp: str
    target: str
    index_expr: str
    mode: str
    value: float
    strength: float
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    note: str = ""


def tail(items, n: int = 20):
    values = list(items or [])
    return values[-int(n):]
