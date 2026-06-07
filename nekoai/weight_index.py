from __future__ import annotations

import re
from typing import Any

from .model_manager import ModelSession
from .tensor_ops import parse_index_expr, tensor_preview, tensor_stats


def _match(name: str, query: str, regex: bool, alias: str = "") -> bool:
    q = (query or "").strip()
    if not q:
        return True
    haystack = f"{name} {alias}".lower()
    if regex:
        try:
            return re.search(q, name) is not None or (alias and re.search(q, alias) is not None)
        except re.error as exc:
            raise ValueError(f"Bad regex: {exc}") from exc
    terms = [x.lower() for x in q.replace(",", " ").split() if x.strip()]
    return all(t in haystack for t in terms)


def search_parameters(session: ModelSession, query: str = "", regex: bool = False, limit: int = 300, stats: bool = False) -> list[dict[str, Any]]:
    session.require_loaded()
    rows: list[dict[str, Any]] = []
    for name, p in session.named_parameters():
        alias = session.aliases.get(name, "")
        if not _match(name, query, regex, alias):
            continue
        row = {
            "name": name,
            "alias": alias,
            "shape": "x".join(str(x) for x in p.shape),
            "ndim": int(p.ndim),
            "numel": int(p.numel()),
            "dtype": str(p.dtype).replace("torch.", ""),
            "device": str(p.device),
            "trainable": bool(p.requires_grad),
        }
        if stats:
            st = tensor_stats(p.data, sample=50_000)
            row.update({k: st.get(k) for k in ["mean", "std", "min", "max", "rms", "finite", "nan", "sampled"]})
        rows.append(row)
        if len(rows) >= int(limit):
            break
    return rows


def inspect_parameter(session: ModelSession, name: str, index_expr: str = ":") -> dict[str, Any]:
    p = session.get_parameter(name)
    idx = parse_index_expr(index_expr)
    view = p.data[idx] if idx else p.data
    return {"stats": tensor_stats(view), "preview": tensor_preview(view)}


def model_parameter_summary(session: ModelSession) -> dict[str, Any]:
    session.require_loaded()
    total = 0
    by_dtype: dict[str, int] = {}
    by_device: dict[str, int] = {}
    tensors = 0
    for _name, p in session.named_parameters():
        n = int(p.numel())
        total += n
        tensors += 1
        dtype = str(p.dtype).replace("torch.", "")
        device = str(p.device)
        by_dtype[dtype] = by_dtype.get(dtype, 0) + n
        by_device[device] = by_device.get(device, 0) + n
    return {"model_id": session.model_id, "parameters": total, "parameter_tensors": tensors, "aliases": len(session.aliases), "by_dtype": by_dtype, "by_device": by_device}
