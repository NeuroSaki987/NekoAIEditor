from __future__ import annotations

import re
from typing import Any

import torch

from .model_manager import ModelSession


def _param_count(module: torch.nn.Module, recurse: bool = True) -> int:
    return int(sum(int(p.numel()) for p in module.parameters(recurse=recurse)))


def _first_param(module: torch.nn.Module) -> torch.nn.Parameter | None:
    for p in module.parameters(recurse=True):
        return p
    return None


def _symbols(module_name: str, module: torch.nn.Module, limit: int = 4) -> str:
    vals: list[str] = []
    for i, (name, p) in enumerate(module.named_parameters(recurse=False)):
        if i >= limit:
            vals.append("...")
            break
        vals.append(f"{module_name}.{name}:{tuple(int(x) for x in p.shape)}")
    return "; ".join(vals)


def _matches(name: str, query: str, regex: bool) -> bool:
    q = (query or "").strip()
    if not q:
        return True
    if regex:
        return re.search(q, name) is not None
    terms = [t.lower() for t in q.replace(",", " ").split() if t.strip()]
    lower = name.lower()
    return all(t in lower for t in terms)


def disassemble_modules(session: ModelSession, query: str = "", regex: bool = False, leaf_only: bool = False, limit: int = 1000) -> list[dict[str, Any]]:
    session.require_loaded()
    rows: list[dict[str, Any]] = []
    for address, (name, module) in enumerate(session.model.named_modules()):
        if not name:
            continue
        children = list(module.children())
        if leaf_only and children:
            continue
        if not _matches(name, query, regex):
            continue
        first = _first_param(module)
        rows.append({
            "addr": f"0x{address:06X}",
            "name": name,
            "class": module.__class__.__name__,
            "depth": name.count("."),
            "leaf": len(children) == 0,
            "direct_params": _param_count(module, False),
            "recursive_params": _param_count(module, True),
            "dtype": str(first.dtype).replace("torch.", "") if first is not None else "",
            "device": str(first.device) if first is not None else "",
            "symbols": _symbols(name, module),
        })
        if len(rows) >= int(limit):
            break
    return rows
