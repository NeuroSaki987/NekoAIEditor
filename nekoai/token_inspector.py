from __future__ import annotations

from typing import Any

import torch

from .kv_cache_debugger import KVRuntimeState, iter_cache_layers
from .model_manager import ModelSession
from .tensor_ops import apply_tensor_edit_, tensor_stats
from .utils import now_ts

TOKEN_SEARCH_MODES = [
    "auto",
    "exact text span",
    "single token text contains",
    "token id",
    "show all context tokens",
]

TOKEN_COMPONENT_MODES = ["key", "value", "both"]


def _decode_token(session: ModelSession, token_id: int) -> str:
    try:
        return str(session.tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return str(token_id)


def _normalize_text(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.casefold()


def _context_ids(state: KVRuntimeState) -> list[int]:
    return [int(x) for x in (state.input_ids or [])]


def _cache_window_info(state: KVRuntimeState) -> tuple[int, int]:
    layers = iter_cache_layers(state.cache)
    if not layers:
        return 0, 0
    _layer, key, _value, _source = layers[0]
    cache_len = int(key.shape[-2]) if key.ndim >= 2 else 0
    total = len(_context_ids(state))
    offset = max(0, total - cache_len) if cache_len and total > cache_len else 0
    return offset, cache_len


def _absolute_to_kv_position(state: KVRuntimeState, position: int) -> int | None:
    offset, cache_len = _cache_window_info(state)
    if cache_len <= 0:
        return None
    kv_pos = int(position) - int(offset)
    if kv_pos < 0 or kv_pos >= cache_len:
        return None
    return int(kv_pos)


def context_token_rows(session: ModelSession, state: KVRuntimeState, max_rows: int = 5000) -> list[dict[str, Any]]:
    ids = _context_ids(state)
    generated_start = max(0, len(ids) - len(state.generated_ids or []))
    all_special = set(int(x) for x in getattr(session.tokenizer, "all_special_ids", []) or [])
    offset, cache_len = _cache_window_info(state)
    rows: list[dict[str, Any]] = []
    for pos, token_id in enumerate(ids[: int(max_rows)]):
        token_text = _decode_token(session, token_id)
        kv_pos = _absolute_to_kv_position(state, pos)
        rows.append(
            {
                "position": int(pos),
                "position_from_end": int(pos - len(ids)),
                "kv_position": "" if kv_pos is None else int(kv_pos),
                "token_id": int(token_id),
                "token_text": token_text,
                "token_repr": repr(token_text),
                "source": "generated" if pos >= generated_start else "prompt",
                "is_special": bool(token_id in all_special),
                "in_kv_cache": bool(kv_pos is not None),
                "kv_window_offset": int(offset),
                "cache_seq_len": int(cache_len),
            }
        )
    return rows


def query_tokenization_rows(session: ModelSession, text: str, add_space_variant: bool = True) -> list[dict[str, Any]]:
    session.require_loaded()
    query = text or ""
    variants: list[str] = []
    for candidate in [query, query.strip()]:
        if candidate and candidate not in variants:
            variants.append(candidate)
    if add_space_variant and query and not query.startswith(" "):
        spaced = " " + query
        if spaced not in variants:
            variants.append(spaced)
    rows: list[dict[str, Any]] = []
    for variant in variants or [query]:
        try:
            token_ids = session.tokenizer.encode(variant, add_special_tokens=False)
        except Exception as exc:
            rows.append({"variant": variant, "token_index": None, "token_id": None, "token_text": "", "token_repr": "", "error": str(exc)})
            continue
        if not token_ids:
            rows.append({"variant": variant, "token_index": None, "token_id": None, "token_text": "", "token_repr": "", "error": "tokenized to empty"})
        for i, token_id in enumerate(token_ids):
            token_text = _decode_token(session, int(token_id))
            rows.append(
                {
                    "variant": variant,
                    "variant_repr": repr(variant),
                    "token_index": int(i),
                    "token_id": int(token_id),
                    "token_text": token_text,
                    "token_repr": repr(token_text),
                    "sequence_len": int(len(token_ids)),
                    "sequence_ids": str([int(x) for x in token_ids]),
                }
            )
    return rows


def _encoded_variants(session: ModelSession, query: str) -> list[tuple[str, list[int]]]:
    rows = query_tokenization_rows(session, query)
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for row in rows:
        variant = str(row.get("variant") or "")
        tid = row.get("token_id")
        if tid is None:
            continue
        if variant not in grouped:
            grouped[variant] = []
            order.append(variant)
        grouped[variant].append(int(tid))
    return [(variant, grouped[variant]) for variant in order if grouped[variant]]


def _find_subsequence(haystack: list[int], needle: list[int]) -> list[tuple[int, int]]:
    if not needle or not haystack or len(needle) > len(haystack):
        return []
    spans: list[tuple[int, int]] = []
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            spans.append((i, i + n - 1))
    return spans


def search_context_tokens(
    session: ModelSession,
    state: KVRuntimeState,
    query: str,
    mode: str = "auto",
    case_sensitive: bool = False,
    max_rows: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search the active KV context by text, token text, token id, or exact tokenized span."""
    session.require_loaded()
    ids = _context_ids(state)
    normalized_mode = (mode or "auto").strip().lower()
    if not ids:
        meta = {"query": query or "", "mode": normalized_mode, "total_tokens": 0, "matched_positions": [], "span_matches": [], "message": "No active KV context tokens."}
        return [], meta

    show_all = normalized_mode in {"show all context tokens", "all", "context", ""} or not (query or "").strip()
    base_rows = context_token_rows(session, state, max_rows=max(len(ids), int(max_rows)))
    match_map: dict[int, list[dict[str, Any]]] = {}
    span_matches: list[dict[str, Any]] = []
    q = query or ""

    def add_match(pos: int, payload: dict[str, Any]) -> None:
        if 0 <= int(pos) < len(ids):
            match_map.setdefault(int(pos), []).append(payload)

    if not show_all:
        q_norm = _normalize_text(q, case_sensitive)
        if normalized_mode in {"auto", "exact text span", "span", "exact"}:
            for variant, token_ids in _encoded_variants(session, q):
                for start, end in _find_subsequence(ids, token_ids):
                    span = {"kind": "exact_text_span", "variant": variant, "start": int(start), "end": int(end), "token_ids": [int(x) for x in token_ids]}
                    span_matches.append(span)
                    for pos in range(start, end + 1):
                        add_match(pos, span)
        if normalized_mode in {"auto", "single token text contains", "token text contains", "contains", "text"} and q_norm:
            for row in base_rows:
                tok_norm = _normalize_text(str(row.get("token_text") or ""), case_sensitive)
                repr_norm = _normalize_text(str(row.get("token_repr") or ""), case_sensitive)
                if q_norm in tok_norm or q_norm in repr_norm:
                    add_match(int(row["position"]), {"kind": "single_token_text_contains", "query": q})
        if normalized_mode in {"auto", "token id", "id"} and q.strip().lstrip("-").isdigit():
            wanted = int(q.strip())
            for row in base_rows:
                if int(row["token_id"]) == wanted:
                    add_match(int(row["position"]), {"kind": "token_id", "query": wanted})

    out: list[dict[str, Any]] = []
    for row in base_rows:
        pos = int(row["position"])
        matches = match_map.get(pos, [])
        if show_all or matches:
            kinds = sorted(set(str(m.get("kind") or "match") for m in matches))
            starts = sorted(set(int(m.get("start")) for m in matches if m.get("start") is not None))
            ends = sorted(set(int(m.get("end")) for m in matches if m.get("end") is not None))
            new_row = dict(row)
            new_row.update(
                {
                    "match": bool(show_all or matches),
                    "match_kind": ",".join(kinds) if kinds else ("all" if show_all else ""),
                    "span_start": starts[0] if starts else "",
                    "span_end": ends[-1] if ends else "",
                    "kv_slice_all_heads": build_token_kv_slice(int(row.get("kv_position"))) if str(row.get("kv_position", "")).strip() != "" else "outside_current_kv_window",
                }
            )
            out.append(new_row)
            if len(out) >= int(max_rows):
                break
    matched_positions = [int(r["position"]) for r in out]
    meta = {
        "query": q,
        "mode": normalized_mode,
        "case_sensitive": bool(case_sensitive),
        "total_tokens": len(ids),
        "matched_count": len(matched_positions),
        "matched_positions": matched_positions,
        "span_matches": span_matches[:128],
        "limited_to": int(max_rows),
        "show_all": bool(show_all),
    }
    return out, meta


def parse_index_list(expr: str | None, count: int, default_all: bool = True, name: str = "index") -> list[int]:
    text = (expr or "").strip()
    if not text or text.lower() in {"all", "*", ":"}:
        if default_all:
            return list(range(int(count)))
        raise ValueError(f"{name} expression is empty.")
    out: list[int] = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if ":" in part:
            bits = part.split(":")
            if len(bits) > 3:
                raise ValueError(f"Bad {name} slice: {part}")
            start = int(bits[0]) if bits[0] else None
            stop = int(bits[1]) if len(bits) > 1 and bits[1] else None
            step = int(bits[2]) if len(bits) > 2 and bits[2] else None
            for i in range(*slice(start, stop, step).indices(int(count))):
                if i not in out:
                    out.append(i)
        else:
            i = int(part)
            if i < 0:
                i = int(count) + i
            if i < 0 or i >= int(count):
                raise IndexError(f"{name} {part} outside 0..{int(count) - 1}")
            if i not in out:
                out.append(i)
    return out


def _parse_basic_selector(expr: str | None, count: int, name: str) -> int | slice:
    text = (expr or "").strip()
    if not text or text.lower() in {"all", "*", ":"}:
        return slice(None)
    if "," in text:
        raise ValueError(f"{name} supports one int or one slice here; comma lists are intentionally disabled to keep edits as live tensor views.")
    if ":" in text:
        bits = text.split(":")
        if len(bits) > 3:
            raise ValueError(f"Bad {name} slice: {text}")
        start = int(bits[0]) if bits[0] else None
        stop = int(bits[1]) if len(bits) > 1 and bits[1] else None
        step = int(bits[2]) if len(bits) > 2 and bits[2] else None
        return slice(start, stop, step)
    idx = int(text)
    if idx < 0:
        idx = int(count) + idx
    if idx < 0 or idx >= int(count):
        raise IndexError(f"{name} {text} outside 0..{int(count) - 1}")
    return idx


def _components(component_mode: str) -> list[str]:
    c = (component_mode or "key").strip().lower()
    if c in {"both", "key+value", "key,value", "all"}:
        return ["key", "value"]
    if c.startswith("v"):
        return ["value"]
    return ["key"]


def build_token_kv_slice(position: int, head_expr: str = ":", dim_expr: str = ":") -> str:
    head = (head_expr or ":").strip() or ":"
    dim = (dim_expr or ":").strip() or ":"
    return f":, {head}, {int(position)}, {dim}"


def _component_tensor(key: torch.Tensor, value: torch.Tensor, component: str) -> torch.Tensor:
    return key if component == "key" else value


def token_layer_info(
    session: ModelSession,
    state: KVRuntimeState,
    positions: list[int],
    layer_expr: str = "all",
    component_mode: str = "both",
    max_rows: int = 1000,
) -> list[dict[str, Any]]:
    session.require_loaded()
    if not state.supported or state.cache is None:
        return []
    layers = iter_cache_layers(state.cache)
    layer_indices = parse_index_list(layer_expr, len(layers), default_all=True, name="layer")
    ids = _context_ids(state)
    rows: list[dict[str, Any]] = []
    for local_layer_index in layer_indices:
        layer_id, key, value, source = layers[local_layer_index]
        for component in _components(component_mode):
            tensor = _component_tensor(key, value, component)
            if tensor.ndim < 4:
                rows.append({"layer": int(layer_id), "component": component, "error": f"Expected [batch, heads, seq, head_dim], got {list(tensor.shape)}"})
                continue
            seq = int(tensor.shape[-2])
            for raw_pos in positions:
                pos = int(raw_pos)
                if pos < 0:
                    pos = seq + pos
                if pos < 0 or pos >= seq:
                    rows.append({"layer": int(layer_id), "component": component, "position": int(raw_pos), "error": f"position outside cache seq_len {seq}"})
                    continue
                view = tensor[:, :, pos, :]
                stats = tensor_stats(view, sample=4096)
                head_rms_min = head_rms_max = head_rms_argmax = None
                try:
                    head_view = view.detach().float()
                    # Reduce batch/head_dim, keep head axis. Shape is [batch, heads, head_dim].
                    rms_by_head = torch.sqrt(torch.mean(head_view * head_view, dim=(0, 2))).cpu()
                    if rms_by_head.numel() > 0:
                        head_rms_min = float(rms_by_head.min().item())
                        head_rms_max = float(rms_by_head.max().item())
                        head_rms_argmax = int(torch.argmax(rms_by_head).item())
                except Exception:
                    pass
                token_id = int(ids[pos]) if 0 <= pos < len(ids) else None
                token_text = _decode_token(session, token_id) if token_id is not None else ""
                rows.append(
                    {
                        "position": int(pos),
                        "token_id": token_id,
                        "token_text": token_text,
                        "token_repr": repr(token_text),
                        "layer": int(layer_id),
                        "component": component,
                        "source": source,
                        "shape": str(list(view.shape)),
                        "dtype": str(tensor.dtype).replace("torch.", ""),
                        "device": str(tensor.device),
                        "heads": int(tensor.shape[1]),
                        "head_dim": int(tensor.shape[-1]),
                        "mean": stats.get("mean"),
                        "std": stats.get("std"),
                        "rms": stats.get("rms"),
                        "min": stats.get("min"),
                        "max": stats.get("max"),
                        "nan": stats.get("nan"),
                        "head_rms_min": head_rms_min,
                        "head_rms_max": head_rms_max,
                        "head_rms_argmax": head_rms_argmax,
                        "kv_slice_all_heads": build_token_kv_slice(pos),
                    }
                )
                if len(rows) >= int(max_rows):
                    return rows
    return rows


def edit_token_positions(
    cache: Any,
    positions: list[int],
    layer_expr: str = "all",
    component_mode: str = "both",
    head_expr: str = ":",
    dim_expr: str = ":",
    mode: str = "add",
    value: float = 0.0,
    strength: float = 0.1,
    max_targets: int = 4096,
) -> dict[str, Any]:
    if cache is None:
        raise RuntimeError("No active KV cache.")
    layers = iter_cache_layers(cache)
    if not layers:
        raise RuntimeError("No editable KV cache layers found.")
    layer_indices = parse_index_list(layer_expr, len(layers), default_all=True, name="layer")
    components = _components(component_mode)
    unique_positions = []
    for p in positions:
        pi = int(p)
        if pi not in unique_positions:
            unique_positions.append(pi)
    applied: list[dict[str, Any]] = []
    before_first: dict[str, Any] | None = None
    after_last: dict[str, Any] | None = None
    with torch.no_grad():
        for local_layer_index in layer_indices:
            layer_id, key, value_tensor, _source = layers[local_layer_index]
            for component in components:
                tensor = _component_tensor(key, value_tensor, component)
                if tensor.ndim < 4:
                    raise ValueError(f"Layer {layer_id} {component} expected [batch, heads, seq, head_dim], got {list(tensor.shape)}")
                head_sel = _parse_basic_selector(head_expr, int(tensor.shape[1]), "head")
                dim_sel = _parse_basic_selector(dim_expr, int(tensor.shape[-1]), "dim")
                seq = int(tensor.shape[-2])
                for raw_pos in unique_positions:
                    if len(applied) >= int(max_targets):
                        raise RuntimeError(f"Stopped after max_targets={max_targets}; narrow your search/layers.")
                    pos = int(raw_pos)
                    if pos < 0:
                        pos = seq + pos
                    if pos < 0 or pos >= seq:
                        raise IndexError(f"position {raw_pos} outside cache seq_len {seq}")
                    view = tensor[:, head_sel, pos, dim_sel]
                    before = tensor_stats(view, sample=2048)
                    if before_first is None:
                        before_first = before
                    apply_tensor_edit_(view, mode, float(value), float(strength))
                    after = tensor_stats(view, sample=2048)
                    after_last = after
                    applied.append(
                        {
                            "layer": int(layer_id),
                            "component": component,
                            "position": int(pos),
                            "slice": build_token_kv_slice(pos, head_expr, dim_expr),
                            "before_rms": before.get("rms"),
                            "after_rms": after.get("rms"),
                            "shape": before.get("shape"),
                        }
                    )
    if not applied:
        raise RuntimeError("No token-position KV targets were edited.")
    return {
        "timestamp": now_ts(),
        "target": "kv.token_positions",
        "mode": mode,
        "value": float(value),
        "strength": float(strength),
        "layer_expr": layer_expr,
        "component_mode": component_mode,
        "head_expr": head_expr,
        "dim_expr": dim_expr,
        "positions": unique_positions,
        "target_count": len(applied),
        "before": before_first,
        "after": after_last,
        "targets": applied[:128],
    }
