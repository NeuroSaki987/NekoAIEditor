from __future__ import annotations

import inspect
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import zipfile

import torch

from .generation import apply_logit_bias, parse_logit_bias, parse_token_override, sample_token, top_tokens
from .model_manager import ModelSession
from .tensor_ops import (
    apply_tensor_edit_,
    dataframe_to_matrix,
    diff_matrices,
    matrix_with_cell,
    parse_index_expr,
    preview_cell_to_local_index,
    tensor_preview,
    tensor_stats,
)
from .utils import ensure_dir, now_ts, safe_name, write_json, zip_dir


@dataclass
class KVRuntimeState:
    model_id: str = ""
    prompt: str = ""
    input_ids: list[int] = field(default_factory=list)
    generated_ids: list[int] = field(default_factory=list)
    attention_mask: Any = None
    cache: Any = None
    logits: Any = None
    queued_token_id: int | None = None
    steps: int = 0
    done: bool = False
    supported: bool = False
    status: str = "No KV session."
    warnings: list[str] = field(default_factory=list)


def resolve_dtype(session: ModelSession, dtype_mode: str) -> torch.dtype | None:
    mode = (dtype_mode or "keep").lower().strip()
    if mode == "keep":
        return None
    if mode == "model":
        session.require_loaded()
        for p in session.model.parameters():
            return p.dtype
        return None
    mapping = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    if mode not in mapping:
        raise ValueError("dtype must be keep/model/float32/float16/bfloat16")
    return mapping[mode]


def _get_attr_tensor(obj: Any, names: list[str]) -> torch.Tensor | None:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if torch.is_tensor(value):
                return value
    return None


def iter_cache_layers(cache: Any) -> list[tuple[int, torch.Tensor, torch.Tensor, str]]:
    """Return mutable key/value tensor references when possible.

    Handles legacy tuple/list caches, DynamicCache key_cache/value_cache storage,
    and newer Cache.layers storage. It intentionally returns live tensor refs so
    edit operations act like debugger memory writes.
    """
    if cache is None:
        return []
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        keys = list(getattr(cache, "key_cache"))
        values = list(getattr(cache, "value_cache"))
        rows = []
        for i, (k, v) in enumerate(zip(keys, values)):
            if torch.is_tensor(k) and torch.is_tensor(v):
                rows.append((i, k, v, f"{type(cache).__name__}.key_cache[{i}]"))
        if rows:
            return rows
    if hasattr(cache, "layers"):
        rows = []
        for i, layer in enumerate(list(getattr(cache, "layers"))):
            k = _get_attr_tensor(layer, ["keys", "key", "key_cache", "key_states"])
            v = _get_attr_tensor(layer, ["values", "value", "value_cache", "value_states"])
            if k is not None and v is not None:
                rows.append((i, k, v, f"{type(cache).__name__}.layers[{i}]"))
        if rows:
            return rows
    if isinstance(cache, (tuple, list)):
        rows = []
        for i, layer in enumerate(cache):
            if isinstance(layer, (tuple, list)) and len(layer) >= 2 and torch.is_tensor(layer[0]) and torch.is_tensor(layer[1]):
                rows.append((i, layer[0], layer[1], "legacy_tuple"))
        return rows
    if hasattr(cache, "to_legacy_cache"):
        try:
            return iter_cache_layers(cache.to_legacy_cache())
        except Exception:
            return []
    return []


def normalize_cache_for_edit(cache: Any) -> Any:
    if iter_cache_layers(cache):
        return cache
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return cache


def cache_format_name(cache: Any) -> str:
    if cache is None:
        return "none"
    return type(cache).__name__ if not isinstance(cache, (tuple, list)) else "legacy_tuple"


def _map_cache_tensors(cache: Any, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Any:
    if cache is None:
        return None
    if torch.is_tensor(cache):
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None and torch.is_floating_point(cache):
            kwargs["dtype"] = dtype
        return cache.to(**kwargs) if kwargs else cache
    if isinstance(cache, tuple):
        return tuple(_map_cache_tensors(x, device, dtype) for x in cache)
    if isinstance(cache, list):
        return [_map_cache_tensors(x, device, dtype) for x in cache]
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        keys = getattr(cache, "key_cache")
        values = getattr(cache, "value_cache")
        for i in range(len(keys)):
            if torch.is_tensor(keys[i]):
                keys[i] = _map_cache_tensors(keys[i], device, dtype)
            if torch.is_tensor(values[i]):
                values[i] = _map_cache_tensors(values[i], device, dtype)
        return cache
    if hasattr(cache, "layers"):
        for layer in list(getattr(cache, "layers")):
            for key_name in ["keys", "key", "key_cache", "key_states"]:
                if hasattr(layer, key_name) and torch.is_tensor(getattr(layer, key_name)):
                    setattr(layer, key_name, _map_cache_tensors(getattr(layer, key_name), device, dtype))
                    break
            for value_name in ["values", "value", "value_cache", "value_states"]:
                if hasattr(layer, value_name) and torch.is_tensor(getattr(layer, value_name)):
                    setattr(layer, value_name, _map_cache_tensors(getattr(layer, value_name), device, dtype))
                    break
        return cache
    if hasattr(cache, "to_legacy_cache"):
        try:
            return _map_cache_tensors(cache.to_legacy_cache(), device, dtype)
        except Exception:
            return cache
    return cache


def cast_cache(cache: Any, session: ModelSession, dtype_mode: str = "keep", device_mode: str = "model") -> Any:
    dtype = resolve_dtype(session, dtype_mode)
    device = session.input_device() if (device_mode or "model") == "model" else torch.device(device_mode)
    return _map_cache_tensors(cache, device, dtype)


def cache_seq_len(cache: Any) -> int:
    layers = iter_cache_layers(cache)
    if not layers:
        return 0
    _i, key, _value, _source = layers[0]
    return int(key.shape[-2]) if key.ndim >= 2 else 0


def cache_summary(cache: Any, sample: int = 4096) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_idx, key, value, source in iter_cache_layers(cache):
        ks = tensor_stats(key, sample=sample)
        vs = tensor_stats(value, sample=sample)
        rows.append({
            "layer": int(layer_idx),
            "source": source,
            "key_shape": str(list(key.shape)),
            "value_shape": str(list(value.shape)),
            "key_dtype": str(key.dtype).replace("torch.", ""),
            "value_dtype": str(value.dtype).replace("torch.", ""),
            "key_device": str(key.device),
            "value_device": str(value.device),
            "k_heads": int(key.shape[1]) if key.ndim >= 4 else None,
            "v_heads": int(value.shape[1]) if value.ndim >= 4 else None,
            "k_seq": int(key.shape[-2]) if key.ndim >= 2 else None,
            "v_seq": int(value.shape[-2]) if value.ndim >= 2 else None,
            "head_dim": int(key.shape[-1]) if key.ndim >= 1 else None,
            "k_mean": ks.get("mean"),
            "k_std": ks.get("std"),
            "k_rms": ks.get("rms"),
            "k_nan": ks.get("nan"),
            "v_mean": vs.get("mean"),
            "v_std": vs.get("std"),
            "v_rms": vs.get("rms"),
            "v_nan": vs.get("nan"),
        })
    return rows


def cache_anomalies(cache: Any) -> list[dict[str, Any]]:
    rows = cache_summary(cache, sample=8192)
    rms_vals = [float(x) for r in rows for x in (r.get("k_rms"), r.get("v_rms")) if x is not None]
    mean = sum(rms_vals) / len(rms_vals) if rms_vals else 0.0
    var = sum((x - mean) ** 2 for x in rms_vals) / len(rms_vals) if rms_vals else 0.0
    std = var ** 0.5
    out: list[dict[str, Any]] = []
    for r in rows:
        flags: list[str] = []
        if (r.get("k_nan") or 0) > 0 or (r.get("v_nan") or 0) > 0:
            flags.append("nan")
        for key in ["k_rms", "v_rms"]:
            val = r.get(key)
            if val is not None and std > 0 and abs(float(val) - mean) > 3.0 * std:
                flags.append(f"{key}_outlier")
        if flags:
            out.append({"layer": r.get("layer"), "flags": ",".join(sorted(set(flags))), "k_rms": r.get("k_rms"), "v_rms": r.get("v_rms"), "k_nan": r.get("k_nan"), "v_nan": r.get("v_nan")})
    return out


def get_cache_tensor(cache: Any, layer_idx: int, component: str) -> torch.Tensor:
    layers = iter_cache_layers(cache)
    for i, key, value, _source in layers:
        if int(i) == int(layer_idx):
            return key if component.lower().startswith("k") else value
    raise IndexError(f"KV cache layer not found: {layer_idx}")


def inspect_cache_slice(cache: Any, layer_idx: int, component: str, index_expr: str = ":") -> dict[str, Any]:
    tensor = get_cache_tensor(cache, layer_idx, component)
    idx = parse_index_expr(index_expr)
    view = tensor[idx] if idx else tensor
    return {"stats": tensor_stats(view), "preview": tensor_preview(view)}


def inspect_head_vector(cache: Any, layer_idx: int, component: str, head: int, token_pos: int, dim_start: int = 0, dim_count: int = 128) -> dict[str, Any]:
    tensor = get_cache_tensor(cache, layer_idx, component)
    if tensor.ndim < 4:
        raise ValueError("Expected KV tensor shape [batch, heads, seq, head_dim].")
    seq = int(tensor.shape[-2])
    pos = int(token_pos)
    if pos < 0:
        pos = seq + pos
    stop = min(int(tensor.shape[-1]), int(dim_start) + int(dim_count))
    view = tensor[:, int(head), pos, int(dim_start):stop]
    return {"stats": tensor_stats(view), "preview": tensor_preview(view), "slice": f":, {int(head)}, {pos}, {int(dim_start)}:{stop}"}



def _decode_token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def _token_piece(tokenizer: Any, token_id: int) -> str:
    try:
        piece = tokenizer.convert_ids_to_tokens(int(token_id))
        return str(piece)
    except Exception:
        return _decode_token_text(tokenizer, int(token_id))


def _short(text: str, limit: int = 240) -> str:
    value = str(text)
    return value if len(value) <= int(limit) else value[: int(limit) - 1] + "…"


def _token_window(tokenizer: Any, ids: list[int], position: int, radius: int = 5) -> str:
    if not ids:
        return ""
    pos = max(0, min(int(position), len(ids) - 1))
    radius = max(0, int(radius))
    start = max(0, pos - radius)
    end = min(len(ids), pos + radius + 1)
    parts: list[str] = []
    for i in range(start, end):
        marker = "▶" if i == pos else " "
        parts.append(f"{marker}{i}:{ids[i]}:{_decode_token_text(tokenizer, ids[i])!r}")
    return " | ".join(parts)


def tokenize_text(session: ModelSession, text: str, add_special_tokens: bool = False) -> list[dict[str, Any]]:
    session.require_loaded()
    query = text or ""
    ids = session.tokenizer.encode(query, add_special_tokens=bool(add_special_tokens))
    rows: list[dict[str, Any]] = []
    for i, tid in enumerate(ids):
        rows.append({
            "query_index": int(i),
            "token_id": int(tid),
            "token_text": repr(_decode_token_text(session.tokenizer, int(tid))),
            "token_piece": _token_piece(session.tokenizer, int(tid)),
        })
    return rows


def _cache_shape_info(cache: Any) -> dict[str, Any]:
    layers = iter_cache_layers(cache)
    if not layers:
        return {"cache_layers": 0, "cache_seq_len": 0, "heads": None, "head_dim": None, "available_layers": ""}
    i, key, _value, _source = layers[0]
    return {
        "cache_layers": len(layers),
        "cache_seq_len": int(key.shape[-2]) if key.ndim >= 2 else 0,
        "heads": int(key.shape[1]) if key.ndim >= 4 else None,
        "head_dim": int(key.shape[-1]) if key.ndim >= 1 else None,
        "available_layers": f"0..{len(layers) - 1}",
    }


def _append_match(
    rows: list[dict[str, Any]],
    seen: set[tuple[int, int, str]],
    tokenizer: Any,
    ids: list[int],
    position: int,
    length: int,
    match_type: str,
    context_radius: int,
    cache_info: dict[str, Any],
) -> None:
    key = (int(position), int(length), str(match_type))
    if key in seen:
        return
    seen.add(key)
    token_id = int(ids[int(position)])
    matched_ids = ids[int(position) : int(position) + int(length)]
    matched_text = tokenizer.decode(matched_ids, skip_special_tokens=False) if matched_ids else ""
    end_pos = int(position) + int(length) - 1
    in_kv = bool(cache_info.get("cache_seq_len") and end_pos < int(cache_info.get("cache_seq_len") or 0))
    rows.append({
        "match_id": len(rows),
        "position": int(position),
        "end_position": int(end_pos),
        "length": int(length),
        "token_id": token_id,
        "token_text": repr(_decode_token_text(tokenizer, token_id)),
        "token_piece": _token_piece(tokenizer, token_id),
        "matched_text": repr(_short(matched_text, 160)),
        "match_type": match_type,
        "in_kv_cache": in_kv,
        "cache_layers": cache_info.get("cache_layers"),
        "available_layers": cache_info.get("available_layers"),
        "cache_seq_len": cache_info.get("cache_seq_len"),
        "heads": cache_info.get("heads"),
        "head_dim": cache_info.get("head_dim"),
        "slice_all_heads": f":, :, {int(position)}, :",
        "slice_head0": f":, 0, {int(position)}, :",
        "context": _short(_token_window(tokenizer, ids, int(position), context_radius), 360),
    })


def search_tokens_in_state(
    session: ModelSession,
    state: KVRuntimeState,
    query: str,
    mode: str = "auto",
    max_matches: int = 200,
    context_radius: int = 5,
    add_special_tokens: bool = False,
) -> list[dict[str, Any]]:
    """Search current decoded/tokenized KV context by token id, token text, or encoded phrase."""
    session.require_loaded()
    ids = [int(x) for x in (state.input_ids or [])]
    if not ids:
        raise RuntimeError("No active token context. Prefill or import a KV state first.")
    q = (query or "").strip()
    if not q:
        raise ValueError("Enter token text, a phrase, or a numeric token id to search.")
    mode_norm = (mode or "auto").lower().strip()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    cache_info = _cache_shape_info(state.cache)
    limit = max(1, int(max_matches))

    def maybe_stop() -> bool:
        return len(rows) >= limit

    # Numeric token-id search.
    if mode_norm in {"auto", "token id", "id", "token_id"} and q.lstrip("-").isdigit():
        target = int(q)
        for pos, tid in enumerate(ids):
            if tid == target:
                _append_match(rows, seen, session.tokenizer, ids, pos, 1, "token_id", context_radius, cache_info)
                if maybe_stop():
                    return rows[:limit]

    # Encoded phrase search. This allows search text that is not already in the top-token list.
    if mode_norm in {"auto", "encoded phrase", "phrase", "tokenized phrase", "sequence"}:
        qids = [int(x) for x in session.tokenizer.encode(q, add_special_tokens=bool(add_special_tokens))]
        if qids:
            n = len(qids)
            for pos in range(0, max(0, len(ids) - n + 1)):
                if ids[pos : pos + n] == qids:
                    _append_match(rows, seen, session.tokenizer, ids, pos, n, "encoded_phrase", context_radius, cache_info)
                    if maybe_stop():
                        return rows[:limit]

    # Per-token decoded text search. This is useful for GPT-style tokens with leading spaces.
    if mode_norm in {"auto", "decoded token contains", "contains", "token contains", "decoded contains"}:
        q_low = q.lower()
        for pos, tid in enumerate(ids):
            decoded = _decode_token_text(session.tokenizer, tid)
            piece = _token_piece(session.tokenizer, tid)
            if q_low in decoded.lower() or q_low in piece.lower():
                _append_match(rows, seen, session.tokenizer, ids, pos, 1, "decoded_contains", context_radius, cache_info)
                if maybe_stop():
                    return rows[:limit]

    if mode_norm in {"decoded token exact", "exact", "token exact", "decoded exact"}:
        for pos, tid in enumerate(ids):
            decoded = _decode_token_text(session.tokenizer, tid)
            piece = _token_piece(session.tokenizer, tid)
            if q == decoded or q == piece or q == repr(decoded):
                _append_match(rows, seen, session.tokenizer, ids, pos, 1, "decoded_exact", context_radius, cache_info)
                if maybe_stop():
                    return rows[:limit]

    return rows[:limit]


def current_context_tokens(session: ModelSession, state: KVRuntimeState, limit: int = 4096, context_radius: int = 0) -> list[dict[str, Any]]:
    session.require_loaded()
    ids = [int(x) for x in (state.input_ids or [])]
    cache_info = _cache_shape_info(state.cache)
    rows: list[dict[str, Any]] = []
    for pos, tid in enumerate(ids[: max(1, int(limit))]):
        rows.append({
            "position": int(pos),
            "token_id": int(tid),
            "token_text": repr(_decode_token_text(session.tokenizer, int(tid))),
            "token_piece": _token_piece(session.tokenizer, int(tid)),
            "in_kv_cache": bool(cache_info.get("cache_seq_len") and int(pos) < int(cache_info.get("cache_seq_len") or 0)),
            "slice_all_heads": f":, :, {int(pos)}, :",
            "context": _short(_token_window(session.tokenizer, ids, int(pos), int(context_radius)), 360) if context_radius else "",
        })
    return rows


def _normalize_token_position(tensor: torch.Tensor, token_pos: int) -> int:
    if tensor.ndim < 2:
        raise ValueError("KV tensor does not expose a sequence dimension.")
    seq = int(tensor.shape[-2])
    pos = int(token_pos)
    if pos < 0:
        pos = seq + pos
    if pos < 0 or pos >= seq:
        raise IndexError(f"Token position {token_pos} is outside KV seq length {seq}.")
    return pos


def token_kv_layer_stats(cache: Any, token_pos: int, sample: int = 4096) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_idx, key, value, source in iter_cache_layers(cache):
        pos = _normalize_token_position(key, int(token_pos))
        key_index = [slice(None)] * key.ndim
        value_index = [slice(None)] * value.ndim
        key_index[-2] = pos
        value_index[-2] = pos
        k_view = key[tuple(key_index)]
        v_view = value[tuple(value_index)]
        ks = tensor_stats(k_view, sample=sample)
        vs = tensor_stats(v_view, sample=sample)
        rows.append({
            "layer": int(layer_idx),
            "source": source,
            "position": int(pos),
            "key_shape_at_token": str(list(k_view.shape)),
            "value_shape_at_token": str(list(v_view.shape)),
            "key_dtype": str(key.dtype).replace("torch.", ""),
            "value_dtype": str(value.dtype).replace("torch.", ""),
            "key_device": str(key.device),
            "value_device": str(value.device),
            "heads": int(key.shape[1]) if key.ndim >= 4 else None,
            "head_dim": int(key.shape[-1]) if key.ndim >= 1 else None,
            "k_mean": ks.get("mean"),
            "k_std": ks.get("std"),
            "k_rms": ks.get("rms"),
            "k_min": ks.get("min"),
            "k_max": ks.get("max"),
            "k_nan": ks.get("nan"),
            "v_mean": vs.get("mean"),
            "v_std": vs.get("std"),
            "v_rms": vs.get("rms"),
            "v_min": vs.get("min"),
            "v_max": vs.get("max"),
            "v_nan": vs.get("nan"),
        })
    return rows


def token_kv_index(cache: Any, layer_idx: int, component: str, token_pos: int, head: int | None = None, dim_start: int = 0, dim_count: int = 0) -> str:
    tensor = get_cache_tensor(cache, int(layer_idx), component)
    pos = _normalize_token_position(tensor, int(token_pos))
    head_part = ":" if head is None or int(head) < 0 else str(int(head))
    if int(dim_count) and int(dim_count) > 0:
        start = max(0, int(dim_start))
        stop = min(int(tensor.shape[-1]), start + int(dim_count))
        dim_part = f"{start}:{stop}"
    else:
        dim_part = ":"
    if tensor.ndim < 4:
        # Fallback for unusual cache layouts: select sequence dimension and all remaining dims.
        parts = [":"] * tensor.ndim
        parts[-2] = str(pos)
        return ", ".join(parts)
    return f":, {head_part}, {pos}, {dim_part}"


def inspect_token_kv_slice(cache: Any, token_pos: int, layer_idx: int, component: str, head: int | None = None, dim_start: int = 0, dim_count: int = 128) -> dict[str, Any]:
    index_expr = token_kv_index(cache, int(layer_idx), component, int(token_pos), head=head, dim_start=int(dim_start), dim_count=int(dim_count))
    data = inspect_cache_slice(cache, int(layer_idx), component, index_expr)
    data["slice"] = index_expr
    data["token_pos"] = int(token_pos)
    return data


def edit_cache_token_positions(
    cache: Any,
    positions: list[int],
    layer_idx: int,
    component: str,
    mode: str,
    value: float,
    strength: float,
    head: int | None = None,
    dim_start: int = 0,
    dim_count: int = 0,
    max_positions: int = 256,
) -> dict[str, Any]:
    unique_positions: list[int] = []
    seen: set[int] = set()
    for pos in positions:
        p = int(pos)
        if p not in seen:
            seen.add(p)
            unique_positions.append(p)
        if len(unique_positions) >= int(max_positions):
            break
    if not unique_positions:
        raise ValueError("No token positions supplied for KV token edit.")
    records: list[dict[str, Any]] = []
    for pos in unique_positions:
        index_expr = token_kv_index(cache, int(layer_idx), component, pos, head=head, dim_start=int(dim_start), dim_count=int(dim_count))
        rec = edit_cache_slice(cache, int(layer_idx), component, index_expr, mode, float(value), float(strength))
        rec["token_pos"] = int(pos)
        records.append(rec)
    return {
        "timestamp": now_ts(),
        "target": f"kv.layer{int(layer_idx)}.{component}.token_positions",
        "mode": mode,
        "value": float(value),
        "strength": float(strength),
        "positions": unique_positions,
        "position_count": len(unique_positions),
        "head": None if head is None or int(head) < 0 else int(head),
        "dim_start": int(dim_start),
        "dim_count": int(dim_count),
        "edits": records[:32],
    }

def edit_cache_slice(cache: Any, layer_idx: int, component: str, index_expr: str, mode: str, value: float, strength: float) -> dict[str, Any]:
    tensor = get_cache_tensor(cache, layer_idx, component)
    idx = parse_index_expr(index_expr)
    view = tensor[idx] if idx else tensor
    before = tensor_stats(view)
    apply_tensor_edit_(view, mode, float(value), float(strength))
    after = tensor_stats(view)
    return {"timestamp": now_ts(), "target": f"kv.layer{layer_idx}.{component}[{index_expr or ':'}]", "mode": mode, "value": float(value), "strength": float(strength), "before": before, "after": after}


def edit_cache_preview(cache: Any, layer_idx: int, component: str, index_expr: str, old_preview: Any, new_preview: Any, max_changes: int = 256) -> dict[str, Any] | None:
    tensor = get_cache_tensor(cache, layer_idx, component)
    idx = parse_index_expr(index_expr)
    view = tensor[idx] if idx else tensor
    old = dataframe_to_matrix(old_preview)
    new = dataframe_to_matrix(new_preview)
    changes = diff_matrices(old, new, max_changes=max_changes)
    if not changes:
        return None
    before = tensor_stats(view)
    applied: list[dict[str, Any]] = []
    with torch.no_grad():
        for row, col, old_val, new_val in changes:
            local_idx = preview_cell_to_local_index(view.shape, row, col)
            try:
                numeric = float(new_val)
            except Exception as exc:
                raise ValueError(f"KV preview cell [{row}, {col}] must be numeric, got {new_val!r}.") from exc
            target = view[local_idx]
            if not torch.is_floating_point(target):
                raise TypeError("Only floating point KV cache tensors can be edited from the preview grid.")
            old_tensor_value = float(target.detach().float().cpu().item()) if target.numel() == 1 else None
            target.copy_(torch.as_tensor(numeric, device=target.device, dtype=target.dtype))
            applied.append({"row": row, "col": col, "local_index": local_idx, "old_preview": old_val, "old_tensor": old_tensor_value, "new": numeric})
    after = tensor_stats(view)
    return {"timestamp": now_ts(), "target": f"kv.layer{layer_idx}.{component}[{index_expr or ':'}]", "mode": "preview_grid_set", "value": 0.0, "strength": 1.0, "before": before, "after": after, "cells": applied[:64], "cell_count": len(applied)}


def _legacy_layers_for_export(cache: Any, dtype: torch.dtype | None = None) -> list[dict[str, Any]]:
    layers = []
    for i, key, value, source in iter_cache_layers(cache):
        k = key.detach().cpu()
        v = value.detach().cpu()
        if dtype is not None and torch.is_floating_point(k):
            k = k.to(dtype=dtype)
        if dtype is not None and torch.is_floating_point(v):
            v = v.to(dtype=dtype)
        layers.append({"layer": int(i), "source": source, "key": k.clone(), "value": v.clone()})
    return layers



def edit_cache_preview_cell(cache: Any, layer_idx: int, component: str, index_expr: str, old_preview: Any, row: int, col: int, value: Any) -> dict[str, Any] | None:
    """Write one visible KV preview cell into the live cache tensor."""
    old = dataframe_to_matrix(old_preview)
    new = matrix_with_cell(old, int(row), int(col), value)
    return edit_cache_preview(cache, int(layer_idx), component, index_expr or ":", old, new)


def _records_from_table(table: Any) -> list[dict[str, Any]]:
    if table is None:
        return []
    if hasattr(table, "to_dict"):
        try:
            return [dict(x) for x in table.to_dict("records")]
        except Exception:
            pass
    if isinstance(table, dict):
        if "rows" in table and isinstance(table["rows"], list):
            return [dict(x) for x in table["rows"] if isinstance(x, dict)]
        data = table.get("data") or table.get("value") or []
        headers = table.get("headers") or []
        if isinstance(data, list) and headers:
            return [dict(zip(headers, row)) for row in data]
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return [dict(x) for x in data]
    if isinstance(table, list) and all(isinstance(x, dict) for x in table):
        return [dict(x) for x in table]
    return []


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except Exception:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return int(float(value))
    except Exception:
        return None


def _same_float(a: Any, b: Any, eps: float = 1e-12) -> bool:
    af = _float_or_none(a)
    bf = _float_or_none(b)
    if af is None or bf is None:
        return str(a) == str(b)
    return abs(af - bf) <= eps


def _set_token_probability_(logits: torch.Tensor, token_id: int, probability: float) -> float:
    p = min(max(float(probability), 1e-8), 1.0 - 1e-8)
    row = logits[0].float()
    tid = int(token_id)
    if tid < 0 or tid >= row.shape[-1]:
        raise IndexError(f"token_id {tid} outside vocab size {row.shape[-1]}")
    others = row.clone()
    others[tid] = -float("inf")
    logsum_other = torch.logsumexp(others, dim=-1)
    new_logit = (torch.log(torch.tensor(p, device=row.device)) - torch.log(torch.tensor(1.0 - p, device=row.device)) + logsum_other).to(device=logits.device, dtype=logits.dtype)
    logits[0, tid].copy_(new_logit)
    return float(new_logit.detach().float().cpu().item())


def edit_logits_value(state: KVRuntimeState, token_id: int, mode: str, value: float, strength: float = 1.0) -> dict[str, Any]:
    if state.logits is None:
        raise RuntimeError("No logits are available. Prefill or execute one token first.")
    tid = int(token_id)
    if tid < 0 or tid >= int(state.logits.shape[-1]):
        raise IndexError(f"token_id {tid} outside vocab size {state.logits.shape[-1]}")
    mode = (mode or "set_logit").lower().strip()
    s = max(0.0, min(1.0, float(strength)))
    before = float(state.logits[0, tid].detach().float().cpu().item())
    with torch.no_grad():
        if mode in {"set", "set_logit", "logit"}:
            new_value = before * (1.0 - s) + float(value) * s
            state.logits[0, tid].copy_(torch.as_tensor(new_value, device=state.logits.device, dtype=state.logits.dtype))
        elif mode in {"add", "add_delta", "delta"}:
            state.logits[0, tid].add_(float(value) * s)
        elif mode == "multiply":
            factor = 1.0 + (float(value) - 1.0) * s
            state.logits[0, tid].mul_(factor)
        elif mode == "boost":
            state.logits[0, tid].add_(abs(float(value)) * s)
        elif mode == "suppress":
            target = -1.0e4 if float(value) == 0.0 else -abs(float(value))
            new_value = before * (1.0 - s) + target * s
            state.logits[0, tid].copy_(torch.as_tensor(new_value, device=state.logits.device, dtype=state.logits.dtype))
        elif mode in {"target_probability", "probability"}:
            _set_token_probability_(state.logits, tid, float(value))
        else:
            raise ValueError("Logit mode must be set_logit/add_delta/multiply/boost/suppress/target_probability")
    after = float(state.logits[0, tid].detach().float().cpu().item())
    state.status = f"Edited logit for token {tid}: {before:.6g} -> {after:.6g}."
    return {"timestamp": now_ts(), "target": f"logits[{tid}]", "mode": mode, "value": float(value), "strength": s, "before": before, "after": after}


def edit_logits_from_table(state: KVRuntimeState, old_table: Any, new_table: Any, max_changes: int = 256) -> dict[str, Any] | None:
    if state.logits is None:
        raise RuntimeError("No logits are available. Prefill or execute one token first.")
    old_rows = _records_from_table(old_table)
    new_rows = _records_from_table(new_table)
    old_by_rank: dict[int, dict[str, Any]] = {}
    for row in old_rows:
        rank = _int_or_none(row.get("rank"))
        if rank is not None:
            old_by_rank[rank] = row
    applied: list[dict[str, Any]] = []
    for row in new_rows:
        if len(applied) >= int(max_changes):
            break
        tid = _int_or_none(row.get("token_id"))
        if tid is None:
            continue
        if tid < 0 or tid >= int(state.logits.shape[-1]):
            raise IndexError(f"token_id {tid} outside vocab size {state.logits.shape[-1]}")
        old = old_by_rank.get(_int_or_none(row.get("rank")) or -1, {})
        edited_logit = _float_or_none(row.get("edited_logit"))
        raw_logit = _float_or_none(row.get("logit"))
        delta = _float_or_none(row.get("logit_delta"))
        target_p = _float_or_none(row.get("target_probability"))
        changed = False
        if target_p is not None and not _same_float(row.get("target_probability"), old.get("target_probability")):
            rec = edit_logits_value(state, tid, "target_probability", target_p, 1.0)
            rec["source"] = "top_table.target_probability"
            applied.append(rec)
            changed = True
        elif edited_logit is not None and not _same_float(row.get("edited_logit"), old.get("edited_logit")):
            rec = edit_logits_value(state, tid, "set_logit", edited_logit, 1.0)
            rec["source"] = "top_table.edited_logit"
            applied.append(rec)
            changed = True
        elif raw_logit is not None and not _same_float(row.get("logit"), old.get("logit")):
            rec = edit_logits_value(state, tid, "set_logit", raw_logit, 1.0)
            rec["source"] = "top_table.logit"
            applied.append(rec)
            changed = True
        if not changed and delta is not None and abs(delta) > 1e-12 and not _same_float(row.get("logit_delta"), old.get("logit_delta")):
            rec = edit_logits_value(state, tid, "add_delta", delta, 1.0)
            rec["source"] = "top_table.logit_delta"
            applied.append(rec)
    if not applied:
        return None
    state.status = f"Applied {len(applied)} top-token logit table edit(s)."
    return {"timestamp": now_ts(), "target": "logits.top_table", "mode": "top_table_edit", "value": 0.0, "strength": 1.0, "cell_count": len(applied), "edits": applied[:64]}


def _decode_one(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        try:
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            return str(token)
        except Exception:
            return str(token_id)


def _raw_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.convert_ids_to_tokens(int(token_id)))
    except Exception:
        return _decode_one(tokenizer, int(token_id))


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    value = text or ""
    try:
        ids = tokenizer.encode(value, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(value)
    return [int(x) for x in (ids or [])]


def _norm_text(text: Any, case_sensitive: bool = False) -> str:
    out = str(text if text is not None else "")
    return out if case_sensitive else out.lower()


def query_tokenization(session: ModelSession, text: str) -> list[dict[str, Any]]:
    """Tokenize arbitrary text with the loaded tokenizer for UI inspection."""
    session.require_loaded()
    ids = _encode_text(session.tokenizer, text or "")
    rows: list[dict[str, Any]] = []
    for i, tid in enumerate(ids):
        decoded = _decode_one(session.tokenizer, tid)
        rows.append({
            "query_index": i,
            "token_id": tid,
            "token_text": decoded,
            "token_repr": repr(decoded),
            "raw_token": _raw_token(session.tokenizer, tid),
        })
    return rows


def _span_decoded(session: ModelSession, ids: list[int], start: int, end: int) -> str:
    try:
        return str(session.tokenizer.decode(ids[int(start):int(end)], skip_special_tokens=False))
    except Exception:
        return "".join(_decode_one(session.tokenizer, tid) for tid in ids[int(start):int(end)])


def _find_subsequences(ids: list[int], needle: list[int], max_matches: int = 200) -> list[tuple[int, int]]:
    if not ids or not needle or len(needle) > len(ids):
        return []
    out: list[tuple[int, int]] = []
    width = len(needle)
    for start in range(0, len(ids) - width + 1):
        if ids[start:start + width] == needle:
            out.append((start, start + width))
            if len(out) >= int(max_matches):
                break
    return out


def search_context_tokens(
    session: ModelSession,
    state: KVRuntimeState,
    query_text: str = "",
    match_mode: str = "auto",
    case_sensitive: bool = False,
    max_matches: int = 200,
) -> list[dict[str, Any]]:
    """Search the current debugger context and map text to token positions.

    Returned rows are token-position rows rather than only a few sampled logits. A
    multi-token text match therefore returns one row for every token in the span.
    """
    session.require_loaded()
    ids = [int(x) for x in (state.input_ids or [])]
    if not ids:
        return []
    mode = (match_mode or "auto").lower().strip()
    query = query_text or ""
    query_ids = _encode_text(session.tokenizer, query) if query else []
    query_token_texts = [_decode_one(session.tokenizer, tid) for tid in query_ids]
    cache_len = cache_seq_len(state.cache)
    layer_count = len(iter_cache_layers(state.cache))
    rows: list[dict[str, Any]] = []
    seen_spans: set[tuple[str, int, int]] = set()
    match_id = 0

    def add_span(kind: str, start: int, end: int) -> None:
        nonlocal match_id
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > len(ids):
            return
        key = (kind, start, end)
        if key in seen_spans:
            return
        seen_spans.add(key)
        decoded_span = _span_decoded(session, ids, start, end)
        token_slice = f":, :, {start}:{end}, :" if end != start + 1 else f":, :, {start}, :"
        for pos in range(start, end):
            tid = int(ids[pos])
            token_text = _decode_one(session.tokenizer, tid)
            rows.append({
                "match_id": match_id,
                "match_type": kind,
                "start_pos": start,
                "end_pos_exclusive": end,
                "token_pos": pos,
                "offset_in_match": pos - start,
                "span_len": end - start,
                "token_id": tid,
                "token_text": token_text,
                "token_repr": repr(token_text),
                "raw_token": _raw_token(session.tokenizer, tid),
                "decoded_span": decoded_span,
                "query_token_ids": str(query_ids),
                "query_token_texts": repr(query_token_texts),
                "in_cache": bool(cache_len <= 0 or pos < cache_len),
                "cache_seq_len": cache_len,
                "cache_layers": layer_count,
                "layer_range": f"0..{layer_count - 1}" if layer_count else "none",
                "recommended_key_slice": token_slice,
                "recommended_value_slice": token_slice,
            })
        match_id += 1

    limit = max(1, int(max_matches))
    if not query:
        for pos in range(min(len(ids), limit)):
            add_span("context_token", pos, pos + 1)
        return rows[:limit]

    if mode in {"auto", "tokenized_sequence", "sequence", "exact_text"} and query_ids:
        for start, end in _find_subsequences(ids, query_ids, limit):
            add_span("tokenized_sequence", start, end)

    if mode in {"auto", "token_text_contains", "contains", "decoded_token_contains"}:
        q = _norm_text(query, case_sensitive)
        for pos, tid in enumerate(ids):
            if len(rows) >= limit:
                break
            decoded = _decode_one(session.tokenizer, tid)
            raw = _raw_token(session.tokenizer, tid)
            if q in _norm_text(decoded, case_sensitive) or q in _norm_text(raw, case_sensitive):
                add_span("token_text_contains", pos, pos + 1)

    if mode in {"token_id", "id"}:
        try:
            needle_id = int(query.strip())
            for pos, tid in enumerate(ids):
                if len(rows) >= limit:
                    break
                if int(tid) == needle_id:
                    add_span("token_id", pos, pos + 1)
        except Exception:
            pass

    return rows[:limit]


def _axis_index(expr: str | None) -> Any:
    text = (expr or ":").strip()
    if text in {"", ":", "all"}:
        return slice(None)
    parsed = parse_index_expr(text)
    if len(parsed) != 1:
        raise ValueError(f"Axis selector must be a single integer/slice, got {expr!r}.")
    return parsed[0]


def _axis_label(expr: str | None) -> str:
    text = (expr or ":").strip()
    return ":" if text in {"", "all"} else text


def parse_layer_spec(cache: Any, layer_spec: str | None = "current", current_layer: int | None = 0) -> list[int]:
    available = [int(i) for i, _k, _v, _src in iter_cache_layers(cache)]
    if not available:
        return []
    text = (layer_spec or "current").strip().lower()
    if text in {"", "current", "selected"}:
        layer = available[0] if current_layer is None else int(current_layer)
        if layer not in available:
            raise IndexError(f"Current layer {layer} is not in available layers {available[:8]}...")
        return [layer]
    if text in {"all", "*", ":"}:
        return available
    out: list[int] = []
    for part in text.replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item and not item.startswith("-"):
            left, right = item.split("-", 1)
            start = int(left.strip())
            stop = int(right.strip())
            step = 1 if stop >= start else -1
            out.extend(range(start, stop + step, step))
        else:
            out.append(int(item))
    uniq = []
    for layer in out:
        if layer not in available:
            raise IndexError(f"Layer {layer} is not available. Available layers: {available[:20]}{'...' if len(available) > 20 else ''}")
        if layer not in uniq:
            uniq.append(layer)
    return uniq


def _unique_match_spans(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        mid = int(row.get("match_id", 0))
        if mid not in by_id:
            by_id[mid] = {
                "match_id": mid,
                "match_type": row.get("match_type"),
                "start_pos": int(row.get("start_pos", 0)),
                "end_pos_exclusive": int(row.get("end_pos_exclusive", 0)),
                "span_len": int(row.get("span_len", 1)),
                "decoded_span": row.get("decoded_span", ""),
            }
    return [by_id[k] for k in sorted(by_id)]


def select_match_spans(rows: list[dict[str, Any]], occurrence: str | int | None = "first") -> list[dict[str, Any]]:
    spans = _unique_match_spans(rows)
    if not spans:
        return []
    text = str(occurrence if occurrence is not None else "first").strip().lower()
    if text in {"", "first"}:
        return [spans[0]]
    if text in {"last", "latest", "-1"}:
        return [spans[-1]]
    if text in {"all", "*"}:
        return spans
    index = int(float(text))
    if index < 0:
        index = len(spans) + index
    if index < 0 or index >= len(spans):
        raise IndexError(f"Occurrence {occurrence!r} outside 0..{len(spans) - 1}.")
    return [spans[index]]


def token_layer_details(
    session: ModelSession,
    state: KVRuntimeState,
    query_text: str = "",
    match_mode: str = "auto",
    case_sensitive: bool = False,
    layer_spec: str = "all",
    current_layer: int = 0,
    max_matches: int = 50,
    max_rows: int = 500,
) -> list[dict[str, Any]]:
    """Return per-token/per-layer K/V statistics for text matches."""
    session.require_loaded()
    if not state.supported or state.cache is None:
        return []
    matches = search_context_tokens(session, state, query_text, match_mode, case_sensitive, max_matches=max_matches)
    positions: list[int] = []
    pos_meta: dict[int, dict[str, Any]] = {}
    for row in matches:
        pos = int(row.get("token_pos", 0))
        if pos not in positions:
            positions.append(pos)
            pos_meta[pos] = row
    layers = parse_layer_spec(state.cache, layer_spec=layer_spec, current_layer=current_layer)
    out: list[dict[str, Any]] = []
    for pos in positions:
        for layer in layers:
            if len(out) >= int(max_rows):
                return out
            try:
                key = get_cache_tensor(state.cache, layer, "key")
                value = get_cache_tensor(state.cache, layer, "value")
                if key.ndim < 4 or value.ndim < 4:
                    out.append({"token_pos": pos, "layer": layer, "error": "KV tensor is not 4D [batch, heads, seq, head_dim]."})
                    continue
                if pos < 0 or pos >= int(key.shape[-2]) or pos >= int(value.shape[-2]):
                    out.append({"token_pos": pos, "layer": layer, "error": "Token position is outside cache seq length."})
                    continue
                k_view = key[:, :, pos:pos + 1, :]
                v_view = value[:, :, pos:pos + 1, :]
                ks = tensor_stats(k_view, sample=4096)
                vs = tensor_stats(v_view, sample=4096)
                meta = pos_meta.get(pos, {})
                out.append({
                    "match_id": meta.get("match_id"),
                    "token_pos": pos,
                    "token_id": meta.get("token_id"),
                    "token_text": meta.get("token_text"),
                    "decoded_span": meta.get("decoded_span"),
                    "layer": layer,
                    "heads": int(key.shape[1]),
                    "head_dim": int(key.shape[-1]),
                    "seq_len": int(key.shape[-2]),
                    "key_slice": f":, :, {pos}, :",
                    "value_slice": f":, :, {pos}, :",
                    "k_shape": str(list(k_view.shape)),
                    "v_shape": str(list(v_view.shape)),
                    "k_dtype": ks.get("dtype"),
                    "v_dtype": vs.get("dtype"),
                    "k_device": ks.get("device"),
                    "v_device": vs.get("device"),
                    "k_mean": ks.get("mean"),
                    "k_std": ks.get("std"),
                    "k_rms": ks.get("rms"),
                    "k_nan": ks.get("nan"),
                    "v_mean": vs.get("mean"),
                    "v_std": vs.get("std"),
                    "v_rms": vs.get("rms"),
                    "v_nan": vs.get("nan"),
                })
            except Exception as exc:
                out.append({"token_pos": pos, "layer": layer, "error": f"{type(exc).__name__}: {exc}"})
    return out


def search_vocabulary_tokens(
    session: ModelSession,
    state: KVRuntimeState | None,
    query_text: str,
    case_sensitive: bool = False,
    max_results: int = 200,
) -> list[dict[str, Any]]:
    """Search tokenizer vocabulary by text/id and optionally show current logits."""
    session.require_loaded()
    tokenizer = session.tokenizer
    query = query_text or ""
    q = _norm_text(query, case_sensitive)
    query_ids = set(_encode_text(tokenizer, query)) if query else set()
    vocab_items: list[tuple[str, int]] = []
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab_items = [(str(tok), int(tid)) for tok, tid in tokenizer.get_vocab().items()]
        except Exception:
            vocab_items = []
    if not vocab_items:
        size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        vocab_items = [(_raw_token(tokenizer, i), i) for i in range(size)]
    vocab_items.sort(key=lambda x: x[1])
    logits = getattr(state, "logits", None) if state is not None else None
    probs = None
    if torch.is_tensor(logits):
        try:
            probs = torch.softmax(logits.float(), dim=-1)
        except Exception:
            probs = None
    out: list[dict[str, Any]] = []
    for raw, tid in vocab_items:
        decoded = _decode_one(tokenizer, tid)
        hit = False
        if query:
            hit = q in _norm_text(raw, case_sensitive) or q in _norm_text(decoded, case_sensitive) or tid in query_ids
        else:
            hit = True
        if not hit:
            continue
        row: dict[str, Any] = {
            "token_id": tid,
            "token_text": decoded,
            "token_repr": repr(decoded),
            "raw_token": raw,
            "from_query_tokenization": bool(tid in query_ids),
        }
        if torch.is_tensor(logits) and 0 <= tid < int(logits.shape[-1]):
            row["current_logit"] = float(logits[0, tid].detach().float().cpu().item())
            if torch.is_tensor(probs):
                row["current_probability"] = float(probs[0, tid].detach().float().cpu().item())
        out.append(row)
        if len(out) >= int(max_results):
            break
    return out


def edit_cache_by_token_text(
    session: ModelSession,
    state: KVRuntimeState,
    query_text: str,
    match_mode: str,
    occurrence: str | int,
    layer_spec: str,
    current_layer: int,
    component_spec: str,
    current_component: str,
    head_expr: str,
    dim_expr: str,
    mode: str,
    value: float,
    strength: float,
    max_matches: int = 200,
) -> dict[str, Any]:
    """Apply an edit to KV cache positions found by token text search."""
    session.require_loaded()
    if not state.supported or state.cache is None:
        raise RuntimeError("KV debugger is not active or this model has no inspectable cache.")
    if not (query_text or "").strip():
        raise ValueError("Token text query is empty.")
    matches = search_context_tokens(session, state, query_text, match_mode, False, max_matches=max_matches)
    spans = select_match_spans(matches, occurrence)
    if not spans:
        raise LookupError(f"No current-context token position matched {query_text!r}.")
    layers = parse_layer_spec(state.cache, layer_spec=layer_spec, current_layer=int(current_layer))
    comp_text = (component_spec or "current").strip().lower()
    if comp_text in {"current", "selected"}:
        components = [current_component or "key"]
    elif comp_text == "both":
        components = ["key", "value"]
    elif comp_text in {"key", "value"}:
        components = [comp_text]
    else:
        raise ValueError("Component selection must be current/key/value/both.")
    h = _axis_index(head_expr)
    d = _axis_index(dim_expr)
    h_label = _axis_label(head_expr)
    d_label = _axis_label(dim_expr)
    applied: list[dict[str, Any]] = []
    for span in spans:
        start = int(span["start_pos"])
        end = int(span["end_pos_exclusive"])
        pos_index = slice(start, end) if end != start + 1 else start
        pos_label = f"{start}:{end}" if end != start + 1 else str(start)
        for layer in layers:
            for component in components:
                tensor = get_cache_tensor(state.cache, layer, component)
                if tensor.ndim < 4:
                    raise ValueError("KV tensor must be 4D [batch, heads, seq, head_dim] for token-text editing.")
                if start < 0 or end > int(tensor.shape[-2]):
                    raise IndexError(f"Matched token span {start}:{end} outside layer {layer} cache seq length {tensor.shape[-2]}.")
                view = tensor[(slice(None), h, pos_index, d)]
                before = tensor_stats(view, sample=4096)
                apply_tensor_edit_(view, mode, float(value), float(strength))
                after = tensor_stats(view, sample=4096)
                applied.append({
                    "timestamp": now_ts(),
                    "target": f"kv.layer{layer}.{component}[:, {h_label}, {pos_label}, {d_label}]",
                    "query_text": query_text,
                    "match_id": span.get("match_id"),
                    "decoded_span": span.get("decoded_span"),
                    "layer": int(layer),
                    "component": component,
                    "head_expr": h_label,
                    "token_pos": pos_label,
                    "dim_expr": d_label,
                    "mode": mode,
                    "value": float(value),
                    "strength": float(strength),
                    "before": before,
                    "after": after,
                })
    state.status = f"Applied {len(applied)} KV token-text edit(s) for {query_text!r}."
    return {
        "timestamp": now_ts(),
        "target": "kv.token_text_search",
        "query_text": query_text,
        "match_mode": match_mode,
        "occurrence": occurrence,
        "layer_spec": layer_spec,
        "component_spec": component_spec,
        "edit_count": len(applied),
        "matches": spans,
        "edits": applied[:128],
    }


def export_kv_state(state: KVRuntimeState, export_root: str = "exports/kv_cache", export_dtype: str = "keep") -> str:
    if not state.supported or state.cache is None:
        raise RuntimeError("No active KV cache to export.")
    dtype = resolve_dtype(ModelSession(), export_dtype) if export_dtype not in {"keep", "model"} else None
    out_dir = ensure_dir(Path(export_root) / f"kv-{safe_name(state.model_id)}-{now_ts()}")
    payload = {
        "format": "nekoai-kv-cache-v1",
        "model_id": state.model_id,
        "prompt": state.prompt,
        "input_ids": state.input_ids,
        "generated_ids": state.generated_ids,
        "attention_mask": state.attention_mask.detach().cpu() if torch.is_tensor(state.attention_mask) else None,
        "logits": state.logits.detach().cpu() if torch.is_tensor(state.logits) else None,
        "queued_token_id": state.queued_token_id,
        "steps": state.steps,
        "done": state.done,
        "cache_format": cache_format_name(state.cache),
        "layers": _legacy_layers_for_export(state.cache, dtype=dtype),
    }
    torch.save(payload, out_dir / "cache.pt")
    manifest = {k: v for k, v in payload.items() if k not in {"attention_mask", "logits", "layers"}}
    manifest["layer_count"] = len(payload["layers"])
    manifest["summary"] = cache_summary(state.cache, sample=1024)
    manifest["anomalies"] = cache_anomalies(state.cache)
    write_json(out_dir / "manifest.json", manifest)
    return zip_dir(out_dir, str(out_dir) + ".zip")


def _safe_extract_zip(zf: zipfile.ZipFile, target: Path) -> None:
    root = target.resolve()
    for member in zf.infolist():
        dest = (target / member.filename).resolve()
        if not str(dest).startswith(str(root)):
            raise ValueError("Unsafe zip path detected.")
    zf.extractall(target)


def _find_cache_pt(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix.lower() == ".pt":
        return p
    if p.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="nekoai-kv-import-"))
        with zipfile.ZipFile(p, "r") as zf:
            _safe_extract_zip(zf, tmp)
        matches = list(tmp.rglob("cache.pt"))
        if not matches:
            raise FileNotFoundError("No cache.pt found in zip.")
        return matches[0]
    raise ValueError("Import a NekoAI KV .zip or cache.pt file.")


def import_kv_state(session: ModelSession, path: str | Path, dtype_mode: str = "model") -> KVRuntimeState:
    session.require_loaded()
    pt = _find_cache_pt(path)
    payload = torch.load(pt, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "nekoai-kv-cache-v1":
        raise ValueError("Unsupported KV cache file format.")
    layers = payload.get("layers") or []
    legacy = []
    for layer in layers:
        legacy.append((layer["key"], layer["value"]))
    state = KVRuntimeState(
        model_id=str(payload.get("model_id") or session.model_id),
        prompt=str(payload.get("prompt") or ""),
        input_ids=[int(x) for x in payload.get("input_ids") or []],
        generated_ids=[int(x) for x in payload.get("generated_ids") or []],
        attention_mask=payload.get("attention_mask"),
        cache=tuple(legacy),
        logits=payload.get("logits"),
        queued_token_id=payload.get("queued_token_id"),
        steps=int(payload.get("steps") or 0),
        done=bool(payload.get("done") or False),
        supported=True,
        status=f"Imported KV cache from {pt}.",
    )
    state.cache = cast_cache(state.cache, session, dtype_mode=dtype_mode)
    if torch.is_tensor(state.attention_mask):
        state.attention_mask = state.attention_mask.to(device=session.input_device())
    if torch.is_tensor(state.logits):
        state.logits = state.logits.to(device=session.input_device())
    return state


def prefill_kv(session: ModelSession, prompt: str, dtype_mode: str = "model", force_cache: bool = True) -> KVRuntimeState:
    session.require_loaded()
    tokenizer = session.tokenizer
    text = prompt or " "
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {k: v.to(session.input_device()) for k, v in encoded.items()}
    warnings: list[str] = []
    if hasattr(session.model, "config") and getattr(session.model.config, "is_encoder_decoder", False):
        warnings.append("Encoder-decoder models may not expose decoder-only KV cache in the same format.")
    if hasattr(session.model, "config") and getattr(session.model.config, "use_cache", None) is False:
        warnings.append("model.config.use_cache is False; forcing use_cache=True for inference debugger.")
    with torch.no_grad():
        outputs = session.model(**encoded, use_cache=force_cache, return_dict=True)
    cache = getattr(outputs, "past_key_values", None)
    if cache is None or not iter_cache_layers(normalize_cache_for_edit(cache)):
        return KVRuntimeState(
            model_id=session.model_id,
            prompt=text,
            input_ids=[int(x) for x in encoded["input_ids"][0].detach().cpu().tolist()],
            supported=False,
            status="This model did not return an inspectable KV cache. Use weight editor/static IDA mode or a causal Transformer with use_cache support.",
            warnings=warnings,
        )
    cache = normalize_cache_for_edit(cache)
    cache = cast_cache(cache, session, dtype_mode=dtype_mode)
    logits = outputs.logits[:, -1, :].detach()
    state = KVRuntimeState(
        model_id=session.model_id,
        prompt=text,
        input_ids=[int(x) for x in encoded["input_ids"][0].detach().cpu().tolist()],
        attention_mask=encoded.get("attention_mask"),
        cache=cache,
        logits=logits,
        supported=True,
        status=f"KV prefill complete. cache={cache_format_name(cache)}, layers={len(iter_cache_layers(cache))}.",
        warnings=warnings,
    )
    return state


def decoded_text(session: ModelSession, state: KVRuntimeState) -> str:
    if not state.input_ids:
        return ""
    return session.tokenizer.decode(state.input_ids, skip_special_tokens=False)


def queue_next_token(
    session: ModelSession,
    state: KVRuntimeState,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.95,
    token_override: str | None = None,
    logit_bias_json: str | None = None,
) -> KVRuntimeState:
    if not state.supported:
        raise RuntimeError("KV debugger is not active or unsupported for this model.")
    override = parse_token_override(session.tokenizer, token_override)
    if override is not None:
        state.queued_token_id = int(override)
        state.status = f"Queued override token id {override}: {repr(session.tokenizer.decode([override], skip_special_tokens=False))}"
        return state
    if state.logits is None:
        raise RuntimeError("No logits available. Queue a token id manually or prefill again.")
    logits = state.logits
    bias = parse_logit_bias(session.tokenizer, logit_bias_json)
    if bias:
        logits = apply_logit_bias(logits, bias)
    tid = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    state.queued_token_id = int(tid)
    state.status = f"Queued sampled token id {tid}: {repr(session.tokenizer.decode([tid], skip_special_tokens=False))}"
    return state


def _forward_accepts(model: Any, name: str) -> bool:
    try:
        sig = inspect.signature(model.forward)
        return name in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        return False


def execute_queued_token(session: ModelSession, state: KVRuntimeState, dtype_mode: str = "model") -> KVRuntimeState:
    if not state.supported or state.cache is None:
        raise RuntimeError("KV debugger is not active.")
    if state.queued_token_id is None:
        raise RuntimeError("Queue a token first.")
    token_id = int(state.queued_token_id)
    device = session.input_device()
    state.cache = cast_cache(state.cache, session, dtype_mode=dtype_mode)
    old_cache_len = cache_seq_len(state.cache) or len(state.input_ids)
    token = torch.tensor([[token_id]], dtype=torch.long, device=device)
    if torch.is_tensor(state.attention_mask):
        mask = state.attention_mask.to(device=device)
        # Some imported states may have shorter masks; repair them before append.
        if mask.shape[-1] < old_cache_len:
            pad = torch.ones((mask.shape[0], old_cache_len - mask.shape[-1]), dtype=mask.dtype, device=device)
            mask = torch.cat([mask, pad], dim=-1)
        state.attention_mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=device)], dim=-1)
    else:
        state.attention_mask = torch.ones((1, old_cache_len + 1), dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {
        "input_ids": token,
        "attention_mask": state.attention_mask,
        "past_key_values": state.cache,
        "use_cache": True,
        "return_dict": True,
    }
    if _forward_accepts(session.model, "cache_position"):
        kwargs["cache_position"] = torch.arange(old_cache_len, old_cache_len + 1, dtype=torch.long, device=device)
    if _forward_accepts(session.model, "position_ids"):
        kwargs["position_ids"] = torch.tensor([[old_cache_len]], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = session.model(**kwargs)
    cache = getattr(outputs, "past_key_values", None)
    if cache is None:
        raise RuntimeError("Model stopped returning past_key_values during decode.")
    state.cache = normalize_cache_for_edit(cache)
    state.cache = cast_cache(state.cache, session, dtype_mode=dtype_mode)
    state.logits = outputs.logits[:, -1, :].detach()
    state.input_ids.append(token_id)
    state.generated_ids.append(token_id)
    state.queued_token_id = None
    state.steps += 1
    eos = getattr(session.tokenizer, "eos_token_id", None)
    if eos is not None and token_id == int(eos):
        state.done = True
    state.status = f"Executed one token and paused at KV boundary. steps={state.steps}, cache_layers={len(iter_cache_layers(state.cache))}"
    return state


def auto_step(
    session: ModelSession,
    state: KVRuntimeState,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.95,
    dtype_mode: str = "model",
    token_override: str | None = None,
    logit_bias_json: str | None = None,
) -> KVRuntimeState:
    state = queue_next_token(session, state, temperature=temperature, top_k=top_k, top_p=top_p, token_override=token_override, logit_bias_json=logit_bias_json)
    return execute_queued_token(session, state, dtype_mode=dtype_mode)


def kv_top_tokens(session: ModelSession, state: KVRuntimeState, temperature: float = 0.0, top_k: int = 50, top_p: float = 0.95) -> list[dict[str, Any]]:
    return top_tokens(session.tokenizer, state.logits, k=10, temperature=temperature, top_k=top_k, top_p=top_p) if state.supported else []



# ---------------------------------------------------------------------------
# Token text / KV position explorer helpers
# ---------------------------------------------------------------------------

def _escape_cell_text(value: Any, max_len: int = 240) -> str:
    text = str(value).replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _decode_token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def _tokenizer_token_text(tokenizer: Any, token_id: int) -> str:
    try:
        value = tokenizer.convert_ids_to_tokens(int(token_id))
        return str(value)
    except Exception:
        return ""


def _encode_text(tokenizer: Any, text: str, add_special_tokens: bool = False) -> list[int]:
    try:
        ids = tokenizer.encode(text or "", add_special_tokens=bool(add_special_tokens))
    except TypeError:
        ids = tokenizer.encode(text or "")
    return [int(x) for x in ids]


def tokenization_rows(session: ModelSession, text: str, add_special_tokens: bool = False) -> list[dict[str, Any]]:
    """Return tokenizer output for arbitrary text without requiring a KV session."""
    session.require_loaded()
    tokenizer = session.tokenizer
    ids = _encode_text(tokenizer, text or "", add_special_tokens=add_special_tokens)
    rows: list[dict[str, Any]] = []
    offset_rows: list[tuple[int | None, int | None]] = [(None, None)] * len(ids)
    try:
        encoded = tokenizer(text or "", add_special_tokens=bool(add_special_tokens), return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
        if offsets and len(offsets) == len(ids):
            offset_rows = [(int(a), int(b)) for a, b in offsets]
    except Exception:
        pass
    for i, token_id in enumerate(ids):
        start, end = offset_rows[i]
        rows.append({
            "index": i,
            "token_id": int(token_id),
            "token_text": _escape_cell_text(_decode_token_text(tokenizer, token_id)),
            "repr": repr(_decode_token_text(tokenizer, token_id)),
            "tokenizer_token": _escape_cell_text(_tokenizer_token_text(tokenizer, token_id)),
            "char_start": start,
            "char_end": end,
        })
    return rows


def _normalise_search_mode(mode: str | None) -> str:
    value = (mode or "auto").lower().strip().replace(" ", "_").replace("-", "_")
    aliases = {
        "contains": "token_text_contains",
        "exact": "token_text_exact",
        "id": "token_id",
        "sequence": "token_sequence",
        "text_sequence": "token_sequence",
        "vocab": "vocab_contains",
    }
    return aliases.get(value, value or "auto")


def _kv_window_offset(state: KVRuntimeState) -> tuple[int, int]:
    """Return (absolute_offset, cache_len) for mapping absolute token positions to KV positions."""
    cache_len = cache_seq_len(state.cache)
    total = len(state.input_ids)
    if cache_len <= 0 or total <= cache_len:
        return 0, cache_len or total
    # Sliding-window or imported caches may retain only the last cache_len tokens.
    return total - cache_len, cache_len


def _abs_to_kv_pos(state: KVRuntimeState, abs_pos: int) -> int | None:
    offset, cache_len = _kv_window_offset(state)
    kv_pos = int(abs_pos) - int(offset)
    if kv_pos < 0 or kv_pos >= int(cache_len):
        return None
    return int(kv_pos)


def _context_snippet(tokenizer: Any, ids: list[int], start: int, end: int, radius: int = 6) -> str:
    lo = max(0, int(start) - int(radius))
    hi = min(len(ids), int(end) + int(radius))
    try:
        left = tokenizer.decode(ids[lo:int(start)], skip_special_tokens=False)
        mid = tokenizer.decode(ids[int(start):int(end)], skip_special_tokens=False)
        right = tokenizer.decode(ids[int(end):hi], skip_special_tokens=False)
        return _escape_cell_text(f"{left}⟦{mid}⟧{right}")
    except Exception:
        return _escape_cell_text(str(ids[lo:hi]))


def build_kv_token_index_expr(kv_pos: int, kv_end: int | None = None, head: str | int | None = "all", dim_start: int | None = None, dim_count: int | None = None) -> str:
    """Build a safe KV tensor index expression for [batch, heads, seq, head_dim]."""
    start = int(kv_pos)
    stop = int(kv_end) if kv_end is not None else start + 1
    if stop <= start:
        stop = start + 1
    h_text = str(head if head is not None else "all").strip().lower()
    if h_text in {"", "all", ":", "*", "none", "-1"}:
        head_expr = ":"
    else:
        h = int(float(h_text))
        head_expr = f"{h}:{h + 1}"
    if dim_start is None or int(dim_start) < 0:
        dim_expr = ":"
    else:
        ds = int(dim_start)
        if dim_count is None or int(dim_count) <= 0:
            dim_expr = f"{ds}:"
        else:
            dim_expr = f"{ds}:{ds + int(dim_count)}"
    return f":, {head_expr}, {start}:{stop}, {dim_expr}"


def _row_for_context_match(
    tokenizer: Any,
    state: KVRuntimeState,
    ids: list[int],
    match_no: int,
    start: int,
    end: int,
    matched_by: str,
) -> dict[str, Any]:
    token_ids = ids[int(start):int(end)]
    kv_start = _abs_to_kv_pos(state, int(start))
    kv_last = _abs_to_kv_pos(state, int(end) - 1)
    if kv_start is None or kv_last is None:
        kv_end = None
        all_heads = "outside_current_kv_window"
    else:
        kv_end = int(kv_last) + 1
        all_heads = build_kv_token_index_expr(kv_start, kv_end, "all")
    decoded = ""
    try:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        decoded = str(token_ids)
    token_id_value: int | str | None = int(token_ids[0]) if len(token_ids) == 1 else ",".join(str(x) for x in token_ids)
    tokenizer_tok = _tokenizer_token_text(tokenizer, token_ids[0]) if len(token_ids) == 1 else "sequence"
    return {
        "match": int(match_no),
        "position": int(start),
        "end_position": int(end) - 1,
        "kv_pos": kv_start,
        "kv_end": kv_end,
        "token_count": len(token_ids),
        "token_id": token_id_value,
        "token_text": _escape_cell_text(decoded),
        "repr": repr(decoded),
        "tokenizer_token": _escape_cell_text(tokenizer_tok),
        "is_generated": bool(int(start) >= (len(state.input_ids) - len(state.generated_ids))),
        "context": _context_snippet(tokenizer, ids, int(start), int(end)),
        "slice_all_heads": all_heads,
        "matched_by": matched_by,
    }


def search_context_tokens(
    session: ModelSession,
    state: KVRuntimeState,
    query: str,
    mode: str = "auto",
    case_sensitive: bool = False,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Search current decoded context/KV positions by token text, token id, or tokenized text sequence."""
    session.require_loaded()
    tokenizer = session.tokenizer
    ids = [int(x) for x in (state.input_ids or [])]
    if not ids:
        return []
    q = query or ""
    if not q.strip():
        return []
    mode = _normalise_search_mode(mode)
    limit = max(1, int(limit))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()

    def add(start: int, end: int, why: str) -> None:
        if len(rows) >= limit:
            return
        key = (int(start), int(end), why)
        if key in seen:
            return
        seen.add(key)
        rows.append(_row_for_context_match(tokenizer, state, ids, len(rows) + 1, int(start), int(end), why))

    q_cmp = q if case_sensitive else q.lower()

    def text_match(text: str, exact: bool = False) -> bool:
        candidate = text if case_sensitive else text.lower()
        return candidate == q_cmp if exact else q_cmp in candidate

    # Numeric search works in auto mode too.
    if mode in {"auto", "token_id"} and q.strip().lstrip("-").isdigit():
        target = int(q.strip())
        for pos, token_id in enumerate(ids):
            if int(token_id) == target:
                add(pos, pos + 1, "token_id")
        if mode == "token_id":
            return rows

    # Sequence search: search the tokenized query as a contiguous span.
    if mode in {"auto", "token_sequence"}:
        needle = _encode_text(tokenizer, q, add_special_tokens=False)
        if needle:
            n = len(needle)
            for pos in range(0, max(0, len(ids) - n + 1)):
                if ids[pos:pos + n] == needle:
                    add(pos, pos + n, "token_sequence")
        if mode == "token_sequence":
            return rows

    # Single-token text search over decoded token text and tokenizer-native token string.
    exact = mode == "token_text_exact"
    if mode in {"auto", "token_text_contains", "token_text_exact", "tokenizer_token_contains", "tokenizer_token_exact"}:
        for pos, token_id in enumerate(ids):
            decoded = _decode_token_text(tokenizer, token_id)
            tok = _tokenizer_token_text(tokenizer, token_id)
            if mode in {"tokenizer_token_contains", "tokenizer_token_exact"}:
                ok = text_match(tok, exact=mode.endswith("_exact"))
                why = mode
            else:
                ok = text_match(decoded, exact=exact) or text_match(repr(decoded), exact=exact)
                why = "token_text_exact" if exact else "token_text_contains"
            if ok:
                add(pos, pos + 1, why)
            if len(rows) >= limit:
                break
    return rows[:limit]


def search_vocab_tokens(
    session: ModelSession,
    state: KVRuntimeState | None,
    query: str,
    mode: str = "vocab_contains",
    case_sensitive: bool = False,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Search the tokenizer vocabulary by token id/native token/decoded text. Includes current logits when available."""
    session.require_loaded()
    tokenizer = session.tokenizer
    q = query or ""
    if not q.strip():
        return []
    mode = _normalise_search_mode(mode)
    limit = max(1, int(limit))
    q_cmp = q if case_sensitive else q.lower()
    logits = getattr(state, "logits", None) if state is not None else None
    log_norm = None
    if torch.is_tensor(logits):
        try:
            log_norm = torch.logsumexp(logits[0].float(), dim=-1)
        except Exception:
            log_norm = None

    def row(token_id: int, rank: int, matched_by: str) -> dict[str, Any]:
        decoded = _decode_token_text(tokenizer, token_id)
        tok = _tokenizer_token_text(tokenizer, token_id)
        logit_val = None
        prob_val = None
        if torch.is_tensor(logits) and 0 <= int(token_id) < int(logits.shape[-1]):
            try:
                raw = logits[0, int(token_id)].detach().float()
                logit_val = float(raw.cpu().item())
                if log_norm is not None:
                    prob_val = float(torch.exp(raw - log_norm).detach().cpu().item())
            except Exception:
                pass
        return {
            "rank": int(rank),
            "token_id": int(token_id),
            "token_text": _escape_cell_text(decoded),
            "repr": repr(decoded),
            "tokenizer_token": _escape_cell_text(tok),
            "logit": logit_val,
            "probability": prob_val,
            "matched_by": matched_by,
        }

    # Direct id lookup.
    if mode in {"auto", "token_id"} and q.strip().lstrip("-").isdigit():
        tid = int(q.strip())
        try:
            vocab_n = len(tokenizer)
        except Exception:
            vocab_n = getattr(tokenizer, "vocab_size", tid + 1)
        if 0 <= tid < int(vocab_n):
            return [row(tid, 1, "token_id")]
        return []

    candidates: list[int] = []
    try:
        vocab_n = len(tokenizer)
        candidates = list(range(int(vocab_n)))
    except Exception:
        try:
            vocab = tokenizer.get_vocab()
            candidates = sorted({int(v) for v in vocab.values()})
        except Exception:
            candidates = []
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    exact = mode in {"vocab_exact", "token_text_exact", "tokenizer_token_exact"}
    tokenizer_only = mode in {"tokenizer_token_contains", "tokenizer_token_exact"}
    for tid in candidates:
        if len(out) >= limit:
            break
        if tid in seen:
            continue
        decoded = _decode_token_text(tokenizer, tid)
        tok = _tokenizer_token_text(tokenizer, tid)
        dec_cmp = decoded if case_sensitive else decoded.lower()
        repr_cmp = repr(decoded) if case_sensitive else repr(decoded).lower()
        tok_cmp = tok if case_sensitive else tok.lower()
        if tokenizer_only:
            ok = tok_cmp == q_cmp if exact else q_cmp in tok_cmp
            why = "tokenizer_token_exact" if exact else "tokenizer_token_contains"
        else:
            ok = (dec_cmp == q_cmp or repr_cmp == q_cmp or tok_cmp == q_cmp) if exact else (q_cmp in dec_cmp or q_cmp in repr_cmp or q_cmp in tok_cmp)
            why = "vocab_exact" if exact else "vocab_contains"
        if ok:
            seen.add(tid)
            out.append(row(tid, len(out) + 1, why))
    return out


def token_layer_stats(
    cache: Any,
    kv_pos: int,
    head: str | int | None = "all",
    dim_start: int | None = None,
    dim_count: int | None = None,
    sample: int = 8192,
) -> list[dict[str, Any]]:
    """Per-layer K/V statistics for a token position in the current cache window."""
    rows: list[dict[str, Any]] = []
    for layer_idx, key, value, _source in iter_cache_layers(cache):
        if key.ndim < 4 or value.ndim < 4:
            continue
        seq = int(key.shape[-2])
        pos = int(kv_pos)
        if pos < 0:
            pos = seq + pos
        if pos < 0 or pos >= seq:
            continue
        index_expr = build_kv_token_index_expr(pos, pos + 1, head=head, dim_start=dim_start, dim_count=dim_count)
        idx = parse_index_expr(index_expr)
        k_view = key[idx]
        v_view = value[idx]
        ks = tensor_stats(k_view, sample=sample)
        vs = tensor_stats(v_view, sample=sample)
        rows.append({
            "layer": int(layer_idx),
            "kv_pos": int(pos),
            "head": str(head if head is not None else "all"),
            "slice": index_expr,
            "key_shape": str(list(k_view.shape)),
            "value_shape": str(list(v_view.shape)),
            "k_mean": ks.get("mean"),
            "k_std": ks.get("std"),
            "k_rms": ks.get("rms"),
            "k_min": ks.get("min"),
            "k_max": ks.get("max"),
            "k_nan": ks.get("nan"),
            "v_mean": vs.get("mean"),
            "v_std": vs.get("std"),
            "v_rms": vs.get("rms"),
            "v_min": vs.get("min"),
            "v_max": vs.get("max"),
            "v_nan": vs.get("nan"),
        })
    return rows


def edit_cache_token_matches(
    cache: Any,
    match_rows: list[dict[str, Any]],
    layer_idx: int,
    component: str,
    mode: str,
    value: float,
    strength: float,
    head: str | int | None = "all",
    dim_start: int | None = None,
    dim_count: int | None = None,
    all_layers: bool = False,
    max_edits: int = 512,
) -> dict[str, Any]:
    """Apply a scalar KV edit to token positions returned by search_context_tokens."""
    if not match_rows:
        raise ValueError("No token match rows to edit. Search tokens first.")
    layers = [int(i) for i, _k, _v, _s in iter_cache_layers(cache)] if all_layers else [int(layer_idx)]
    components = ["key", "value"] if str(component).lower() == "both" else [component]
    applied: list[dict[str, Any]] = []
    for row in match_rows:
        if len(applied) >= int(max_edits):
            break
        kv_pos = row.get("kv_pos")
        kv_end = row.get("kv_end")
        if kv_pos is None or kv_end is None:
            continue
        for layer in layers:
            for comp in components:
                if len(applied) >= int(max_edits):
                    break
                expr = build_kv_token_index_expr(int(kv_pos), int(kv_end), head=head, dim_start=dim_start, dim_count=dim_count)
                rec = edit_cache_slice(cache, int(layer), comp, expr, mode, float(value), float(strength))
                rec.update({
                    "source": "token_text_match",
                    "position": row.get("position"),
                    "end_position": row.get("end_position"),
                    "kv_pos": int(kv_pos),
                    "kv_end": int(kv_end),
                    "token_text": row.get("token_text"),
                    "matched_by": row.get("matched_by"),
                })
                applied.append(rec)
    if not applied:
        raise ValueError("No editable matches were inside the current KV cache window.")
    return {
        "timestamp": now_ts(),
        "target": "kv.token_text_matches",
        "mode": mode,
        "value": float(value),
        "strength": float(strength),
        "layer": "all" if all_layers else int(layer_idx),
        "component": component,
        "head": str(head),
        "dim_start": dim_start,
        "dim_count": dim_count,
        "edit_count": len(applied),
        "edits": applied[:32],
    }


def _safe_decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return f"<token:{int(token_id)}>"


def _safe_encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    except TypeError:
        return [int(x) for x in tokenizer.encode(text)]


def _unique_sequences(seqs: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for seq in seqs:
        key = tuple(int(x) for x in seq)
        if key and key not in seen:
            seen.add(key)
            out.append(list(key))
    return out


def token_query_sequences(session: ModelSession, query: str) -> list[list[int]]:
    """Return useful tokenizer encodings for a text query.

    Decoder-only BPE/SentencePiece models often represent a word differently at
    the beginning of a string and after a space.  For usability, the token text
    search checks both the literal query and a leading-space variant.
    """
    tokenizer = session.tokenizer
    text = query or ""
    seqs = [_safe_encode(tokenizer, text)]
    if text and not text.startswith((" ", "\n", "\t")):
        seqs.append(_safe_encode(tokenizer, " " + text))
    return _unique_sequences(seqs)


def context_token_rows(session: ModelSession, state: KVRuntimeState, limit: int = 4096) -> list[dict[str, Any]]:
    """Return a token-position table for the current KV debugger context."""
    ids = [int(x) for x in (state.input_ids or [])]
    seq_len = cache_seq_len(state.cache) or len(ids)
    rows: list[dict[str, Any]] = []
    char = 0
    generated_start = max(0, len(ids) - len(state.generated_ids or []))
    for pos, tid in enumerate(ids[: max(0, int(limit))]):
        piece = _safe_decode_token(session.tokenizer, tid)
        start = char
        char += len(piece)
        rows.append({
            "position": int(pos),
            "cache_position": int(pos),
            "in_cache": bool(pos < seq_len),
            "token_id": int(tid),
            "token_text": piece,
            "token_repr": repr(piece),
            "generated": bool(pos >= generated_start),
            "char_start": int(start),
            "char_end": int(char),
            "layer_count": int(len(iter_cache_layers(state.cache))),
            "cache_seq_len": int(seq_len),
            "kv_index_all_heads": f":, :, {int(pos)}, :",
        })
    return rows


def _sequence_matches(ids: list[int], query_ids: list[int]) -> list[tuple[int, int]]:
    if not query_ids or len(query_ids) > len(ids):
        return []
    q = [int(x) for x in query_ids]
    n = len(q)
    return [(i, i + n - 1) for i in range(0, len(ids) - n + 1) if ids[i : i + n] == q]


def find_token_positions(
    session: ModelSession,
    state: KVRuntimeState,
    query: str,
    mode: str = "auto",
    case_sensitive: bool = False,
    limit: int = 512,
) -> list[dict[str, Any]]:
    """Find current-context token positions by text, token id, or encoded sequence.

    The result is intentionally position-centric so rows can be bound directly
    to KV cache slices such as ``:, :, position, :``.
    """
    if not state.input_ids:
        return []
    q = (query or "").strip()
    rows = context_token_rows(session, state, limit=max(len(state.input_ids), 1))
    ids = [int(x) for x in state.input_ids]
    seq_len = cache_seq_len(state.cache) or len(ids)
    max_rows = max(1, int(limit))
    mode = (mode or "auto").strip().lower()
    if not q:
        return rows[:max_rows]

    def text_cmp(value: str, needle: str) -> bool:
        if case_sensitive:
            return needle in value
        return needle.lower() in value.lower()

    found: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int, int]] = set()

    def add_match(base: dict[str, Any], kind: str, match_start: int, match_end: int, encoded_ids: list[int] | None = None) -> None:
        if len(found) >= max_rows:
            return
        pos = int(base["position"])
        key = (pos, kind, int(match_start), int(match_end))
        if key in seen:
            return
        seen.add(key)
        rec = dict(base)
        rec.update({
            "match_kind": kind,
            "match_start": int(match_start),
            "match_end": int(match_end),
            "sequence_length": int(match_end - match_start + 1),
            "encoded_query_ids": str(encoded_ids or []),
            "selected": bool(len(found) == 0),
            "in_cache": bool(pos < seq_len),
        })
        found.append(rec)

    numeric_tid: int | None = None
    if q.lstrip("-").isdigit():
        try:
            numeric_tid = int(q)
        except Exception:
            numeric_tid = None

    if mode in {"auto", "token_id"} and numeric_tid is not None:
        for base in rows:
            if int(base["token_id"]) == numeric_tid:
                add_match(base, "token_id", int(base["position"]), int(base["position"]), [numeric_tid])

    if mode in {"auto", "encoded_sequence", "sequence", "tokenized_text"}:
        for seq in token_query_sequences(session, q):
            for start, end in _sequence_matches(ids, seq):
                for pos in range(start, end + 1):
                    add_match(rows[pos], "encoded_sequence", start, end, seq)

    if mode in {"auto", "token_piece_contains", "token_text_contains", "piece"}:
        for base in rows:
            if text_cmp(str(base.get("token_text", "")), q):
                pos = int(base["position"])
                add_match(base, "token_piece_contains", pos, pos, [int(base["token_id"])])

    if mode in {"auto", "decoded_text_contains", "decoded_text"}:
        joined = "".join(str(x.get("token_text", "")) for x in rows)
        haystack = joined if case_sensitive else joined.lower()
        needle = q if case_sensitive else q.lower()
        start = 0
        while needle and len(found) < max_rows:
            idx = haystack.find(needle, start)
            if idx < 0:
                break
            end_char = idx + len(needle)
            for base in rows:
                if int(base["char_start"]) < end_char and int(base["char_end"]) > idx:
                    add_match(base, "decoded_text_contains", int(base["position"]), int(base["position"]), [int(base["token_id"])])
            start = idx + 1

    return found[:max_rows]


def token_position_detail(session: ModelSession, state: KVRuntimeState, token_pos: int) -> dict[str, Any]:
    rows = context_token_rows(session, state, limit=max(len(state.input_ids), 1))
    pos = int(token_pos)
    if pos < 0:
        pos = len(rows) + pos
    if pos < 0 or pos >= len(rows):
        raise IndexError(f"Token position {token_pos} outside current context length {len(rows)}.")
    rec = dict(rows[pos])
    rec["left_context"] = "".join(str(r["token_text"]) for r in rows[max(0, pos - 8) : pos])
    rec["right_context"] = "".join(str(r["token_text"]) for r in rows[pos + 1 : min(len(rows), pos + 9)])
    rec["decoded_context_window"] = rec["left_context"] + str(rec["token_text"]) + rec["right_context"]
    return rec


def _component_names(component: str) -> list[str]:
    comp = (component or "key").strip().lower()
    if comp in {"both", "key+value", "k+v", "all"}:
        return ["key", "value"]
    if comp.startswith("v"):
        return ["value"]
    return ["key"]


def _layer_ids(cache: Any, layer_scope: str, layer_idx: int | None = None) -> list[int]:
    layers = [int(i) for i, _k, _v, _src in iter_cache_layers(cache)]
    scope = (layer_scope or "current layer").strip().lower()
    if scope in {"all", "all layers", "every layer", "*"}:
        return layers
    target = int(layer_idx or 0)
    if target not in layers:
        raise IndexError(f"Layer {target} not found. Available layers: {layers[:16]}{'...' if len(layers) > 16 else ''}")
    return [target]


def _head_slice_text(head_selector: str | int | None) -> str:
    text = str(head_selector if head_selector is not None else ":").strip()
    if text.lower() in {"", "all", "*", ":"}:
        return ":"
    int(text)  # validate
    return text


def _dim_slice_text(dim_selector: str | None, dim_start: int | None = None, dim_count: int | None = None) -> str:
    text = str(dim_selector or "").strip()
    if text:
        if text.lower() in {"all", "*"}:
            return ":"
        if ":" in text:
            return text
        int(text)  # validate one dim
        return text
    if dim_count is None or int(dim_count) <= 0:
        return ":"
    start = int(dim_start or 0)
    return f"{start}:{start + int(dim_count)}"


def build_token_kv_index(token_position: int, head_selector: str | int | None = ":", dim_selector: str | None = ":") -> str:
    return f":, {_head_slice_text(head_selector)}, {int(token_position)}, {_dim_slice_text(dim_selector)}"


def token_kv_info(
    session: ModelSession,
    state: KVRuntimeState,
    query: str,
    mode: str = "auto",
    component: str = "both",
    layer_scope: str = "all layers",
    layer_idx: int | None = None,
    case_sensitive: bool = False,
    max_matches: int = 64,
    sample: int = 8192,
) -> list[dict[str, Any]]:
    """Return per-token/per-layer KV stats for Token Inspector."""
    matches = find_token_positions(session, state, query, mode=mode, case_sensitive=case_sensitive, limit=max_matches)
    layers = _layer_ids(state.cache, layer_scope, layer_idx)
    comps = _component_names(component)
    out: list[dict[str, Any]] = []
    for m in matches[: max(1, int(max_matches))]:
        pos = int(m["position"])
        for layer in layers:
            for comp in comps:
                tensor = get_cache_tensor(state.cache, layer, comp)
                if tensor.ndim < 3:
                    continue
                if pos < 0 or pos >= int(tensor.shape[-2]):
                    rec = dict(m)
                    rec.update({"layer": layer, "component": comp, "available": False, "reason": f"position {pos} outside seq_len {tensor.shape[-2]}"})
                    out.append(rec)
                    continue
                view = tensor[..., pos : pos + 1, :]
                stats = tensor_stats(view, sample=sample)
                rec = dict(m)
                rec.update({
                    "layer": int(layer),
                    "component": comp,
                    "available": True,
                    "tensor_shape": str(list(tensor.shape)),
                    "heads": int(tensor.shape[1]) if tensor.ndim >= 4 else None,
                    "head_dim": int(tensor.shape[-1]) if tensor.ndim >= 1 else None,
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "device": str(tensor.device),
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "rms": stats.get("rms"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "nan": stats.get("nan"),
                    "kv_index_all_heads": f":, :, {pos}, :",
                })
                out.append(rec)
    return out


def _pick_token_positions(matches: list[dict[str, Any]], policy: str, explicit_position: int | None = None) -> list[int]:
    if explicit_position is not None and int(explicit_position) >= 0:
        return [int(explicit_position)]
    if not matches:
        return []
    policy_norm = (policy or "first").strip().lower()
    positions = [int(m["position"]) for m in matches]
    if policy_norm in {"all", "all matches", "every"}:
        return sorted(set(positions))
    if policy_norm in {"last", "last match"}:
        return [positions[-1]]
    return [positions[0]]


def edit_cache_tokens_by_text(
    session: ModelSession,
    state: KVRuntimeState,
    query: str,
    mode_search: str = "auto",
    layer_scope: str = "current layer",
    layer_idx: int = 0,
    component: str = "key",
    match_policy: str = "first",
    explicit_position: int | None = None,
    head_selector: str | int | None = ":",
    dim_selector: str | None = ":",
    edit_mode: str = "add",
    value: float = 0.0,
    strength: float = 0.1,
    case_sensitive: bool = False,
    max_matches: int = 256,
    max_writes: int = 4096,
) -> dict[str, Any]:
    """Search token text/id and edit corresponding KV slices.

    This is the high-level memory-write path used by the UI so the user does not
    have to manually type tensor indices for common token-oriented edits.
    """
    if not state.supported or state.cache is None:
        raise RuntimeError("KV debugger is not active.")
    matches = find_token_positions(session, state, query, mode=mode_search, case_sensitive=case_sensitive, limit=max_matches)
    positions = _pick_token_positions(matches, match_policy, explicit_position)
    if not positions:
        raise ValueError(f"No token positions matched query {query!r}.")
    layers = _layer_ids(state.cache, layer_scope, int(layer_idx))
    comps = _component_names(component)
    head_text = _head_slice_text(head_selector)
    dim_text = _dim_slice_text(dim_selector)
    records: list[dict[str, Any]] = []
    writes = 0
    for pos in positions:
        for layer in layers:
            for comp in comps:
                if writes >= int(max_writes):
                    raise RuntimeError(f"Stopped after {max_writes} writes. Narrow your token/layer/component selection.")
                tensor = get_cache_tensor(state.cache, layer, comp)
                if pos < 0 or tensor.ndim < 3 or pos >= int(tensor.shape[-2]):
                    continue
                index_expr = f":, {head_text}, {int(pos)}, {dim_text}"
                rec = edit_cache_slice(state.cache, layer, comp, index_expr, edit_mode, float(value), float(strength))
                rec.update({
                    "token_position": int(pos),
                    "token_text": _safe_decode_token(session.tokenizer, int(state.input_ids[pos])) if pos < len(state.input_ids) else "",
                    "layer": int(layer),
                    "component": comp,
                    "index_expr": index_expr,
                    "source": "token_text_kv_edit",
                })
                records.append(rec)
                writes += 1
    state.status = f"Edited KV cache by token search {query!r}: positions={positions}, writes={writes}."
    return {
        "timestamp": now_ts(),
        "target": "kv.token_text_search",
        "query": query,
        "mode_search": mode_search,
        "positions": positions,
        "layers": layers,
        "components": comps,
        "head_selector": head_text,
        "dim_selector": dim_text,
        "edit_mode": edit_mode,
        "value": float(value),
        "strength": float(strength),
        "write_count": writes,
        "records": records[:64],
    }



# -----------------------------
# Token text / KV position tools
# -----------------------------

TOKEN_SEARCH_MODES = [
    "auto",
    "contains token text",
    "exact token text",
    "tokenized text sequence",
    "token id",
    "all tokens",
]


def _decode_ids(tokenizer: Any, ids: list[int]) -> str:
    try:
        return str(tokenizer.decode(ids, skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(ids))


def _decode_one(tokenizer: Any, token_id: int) -> str:
    return _decode_ids(tokenizer, [int(token_id)])


def _normalize_token_text(text: Any) -> str:
    return str(text).replace("▁", " ").replace("Ġ", " ").lower()


def tokenize_text_for_display(session: ModelSession, text: str | None) -> list[dict[str, Any]]:
    """Tokenize arbitrary text for UI display without adding special tokens."""
    session.require_loaded()
    query = text or ""
    try:
        ids = session.tokenizer.encode(query, add_special_tokens=False)
    except TypeError:
        encoded = session.tokenizer(query)
        ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else encoded
        if ids and isinstance(ids[0], list):
            ids = ids[0]
    rows: list[dict[str, Any]] = []
    for i, tid in enumerate([int(x) for x in ids]):
        tok = _decode_one(session.tokenizer, tid)
        rows.append({
            "query_index": i,
            "token_id": tid,
            "token_text": tok,
            "token_repr": repr(tok),
        })
    return rows


def _query_token_ids(session: ModelSession, query: str) -> list[int]:
    return [int(r["token_id"]) for r in tokenize_text_for_display(session, query)]


def _resolve_search_mode(session: ModelSession, query: str, mode: str) -> tuple[str, list[int]]:
    m = (mode or "auto").lower().strip()
    q = (query or "").strip()
    if m in {"all", "all tokens", "*"} or not q:
        return "all tokens", []
    if m in {"id", "token id", "token_id"}:
        return "token id", [int(q)]
    qids = _query_token_ids(session, q)
    if m == "auto":
        if q.lstrip("-").isdigit():
            return "token id", [int(q)]
        if len(qids) > 1:
            return "tokenized text sequence", qids
        return "contains token text", qids
    if m in {"contains", "contains token text", "text contains"}:
        return "contains token text", qids
    if m in {"exact", "exact token text"}:
        return "exact token text", qids
    if m in {"sequence", "tokenized text sequence", "text sequence", "phrase"}:
        return "tokenized text sequence", qids
    raise ValueError(f"Unknown token search mode: {mode}")


def _prompt_token_len(state: KVRuntimeState) -> int:
    return max(0, len(state.input_ids) - len(state.generated_ids))


def _cache_visible_start(state: KVRuntimeState) -> int:
    seq = cache_seq_len(state.cache)
    if seq <= 0:
        return len(state.input_ids)
    return max(0, len(state.input_ids) - seq)


def _context_text(tokenizer: Any, ids: list[int], start: int, stop: int) -> str:
    start = max(0, int(start))
    stop = min(len(ids), int(stop))
    if start >= stop:
        return ""
    return _decode_ids(tokenizer, [int(x) for x in ids[start:stop]])


def current_context_token_rows(
    session: ModelSession,
    state: KVRuntimeState,
    query: str | None = "",
    mode: str = "auto",
    max_results: int = 500,
    context_window: int = 5,
) -> list[dict[str, Any]]:
    """Search the current decoded context and map token positions to live KV positions.

    `pos` is the absolute token index in `state.input_ids`; `kv_pos` is the
    corresponding index inside the currently retained cache. They differ for
    sliding-window / cropped caches.
    """
    session.require_loaded()
    ids = [int(x) for x in state.input_ids]
    if not ids:
        return []
    resolved_mode, qids = _resolve_search_mode(session, query or "", mode)
    qnorm = _normalize_token_text(query or "")
    prompt_len = _prompt_token_len(state)
    visible_start = _cache_visible_start(state)
    cache_len = cache_seq_len(state.cache)

    matched: list[tuple[int, int, int, str, int | None]] = []
    # tuple: token position, span_start, span_end, kind, query_token_index
    if resolved_mode == "all tokens":
        matched = [(i, i, i + 1, "all", None) for i in range(len(ids))]
    elif resolved_mode == "token id":
        target = int(qids[0])
        matched = [(i, i, i + 1, "token_id", None) for i, tid in enumerate(ids) if int(tid) == target]
    elif resolved_mode == "tokenized text sequence":
        if not qids:
            matched = []
        else:
            n = len(qids)
            for start in range(0, max(0, len(ids) - n + 1)):
                if ids[start:start + n] == qids:
                    for offset in range(n):
                        matched.append((start + offset, start, start + n, "sequence", offset))
    else:
        for i, tid in enumerate(ids):
            token_text = _decode_one(session.tokenizer, tid)
            norm = _normalize_token_text(token_text)
            stripped_norm = norm.strip()
            stripped_query = qnorm.strip()
            if resolved_mode == "exact token text":
                ok = norm == qnorm or stripped_norm == stripped_query
            else:
                ok = stripped_query in stripped_norm or qnorm in norm
            if ok:
                matched.append((i, i, i + 1, resolved_mode, None))

    rows: list[dict[str, Any]] = []
    max_n = max(1, int(max_results))
    for pos, span_start, span_end, kind, q_index in matched[:max_n]:
        tid = int(ids[pos])
        token_text = _decode_one(session.tokenizer, tid)
        kv_pos: int | None = None
        visible = bool(cache_len > 0 and visible_start <= pos < visible_start + cache_len)
        if visible:
            kv_pos = int(pos - visible_start)
        rows.append({
            "pos": int(pos),
            "kv_pos": -1 if kv_pos is None else int(kv_pos),
            "cache_visible": visible,
            "token_id": tid,
            "token_text": token_text,
            "token_repr": repr(token_text),
            "source": "generated" if pos >= prompt_len else "prompt",
            "match_kind": kind,
            "query_token_index": "" if q_index is None else int(q_index),
            "span": f"{span_start}:{span_end}",
            "span_text": _context_text(session.tokenizer, ids, span_start, span_end),
            "left_context": repr(_context_text(session.tokenizer, ids, pos - context_window, pos)),
            "right_context": repr(_context_text(session.tokenizer, ids, pos + 1, pos + 1 + context_window)),
        })
    return rows


def _available_layer_ids(cache: Any) -> list[int]:
    return [int(i) for i, _k, _v, _src in iter_cache_layers(cache)]


def _parse_int_ranges(spec: str | None, available: list[int]) -> list[int]:
    text = (spec or "all").strip().lower()
    if text in {"", "all", "*", ":"}:
        return list(available)
    out: list[int] = []
    available_set = set(int(x) for x in available)
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and not part.startswith("-"):
            a, b = part.split("-", 1)
            start, stop = int(a), int(b)
            step = 1 if stop >= start else -1
            for v in range(start, stop + step, step):
                if v in available_set:
                    out.append(v)
        else:
            v = int(part)
            if v in available_set:
                out.append(v)
    # Preserve order, remove duplicates.
    seen: set[int] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def _component_list(component_scope: str | None) -> list[str]:
    text = (component_scope or "both").lower().strip()
    if text in {"both", "all", "key+value", "kv"}:
        return ["key", "value"]
    if text.startswith("k"):
        return ["key"]
    if text.startswith("v"):
        return ["value"]
    raise ValueError("Component must be key/value/both.")


def _parse_heads(head_spec: str | int | float | None, head_count: int) -> list[int | None]:
    if head_spec is None:
        return [None]
    text = str(head_spec).strip().lower()
    if text in {"", "all", "*", ":", "none"}:
        return [None]
    heads = _parse_int_ranges(text, list(range(int(head_count))))
    if not heads:
        raise ValueError(f"No valid heads matched {head_spec!r}; head_count={head_count}")
    return [int(x) for x in heads]


def _dim_slice(dim_start: int | float | None, dim_count: int | float | None, dim_size: int) -> slice:
    start = 0 if dim_start is None else int(dim_start)
    start = max(0, min(start, int(dim_size)))
    count = 0 if dim_count is None else int(dim_count)
    if count <= 0:
        return slice(start, None)
    stop = max(start, min(int(dim_size), start + count))
    return slice(start, stop)


def token_kv_slice_expr(kv_pos: int, head: str | int | None = "all", dim_start: int = 0, dim_count: int = 0) -> str:
    h_text = str(head if head is not None else "all").strip().lower()
    head_expr = ":" if h_text in {"", "all", "*", ":", "none"} else f"{int(float(h_text))}:{int(float(h_text)) + 1}"
    if int(dim_count or 0) <= 0:
        dim_expr = f"{int(dim_start or 0)}:" if int(dim_start or 0) > 0 else ":"
    else:
        start = int(dim_start or 0)
        dim_expr = f"{start}:{start + int(dim_count)}"
    return f":, {head_expr}, {int(kv_pos)}:{int(kv_pos) + 1}, {dim_expr}"


def _kv_token_view(tensor: torch.Tensor, kv_pos: int, head: int | None = None, dim_start: int = 0, dim_count: int = 0) -> torch.Tensor:
    if tensor.ndim < 4:
        raise ValueError("Expected KV tensor shape [batch, heads, seq, head_dim].")
    seq = int(tensor.shape[-2])
    pos = int(kv_pos)
    if pos < 0:
        pos = seq + pos
    if pos < 0 or pos >= seq:
        raise IndexError(f"kv_pos {kv_pos} outside cache seq len {seq}")
    head_count = int(tensor.shape[1])
    if head is None:
        head_slice = slice(None)
    else:
        h = int(head)
        if h < 0:
            h = head_count + h
        if h < 0 or h >= head_count:
            raise IndexError(f"head {head} outside head count {head_count}")
        head_slice = slice(h, h + 1)
    dims = _dim_slice(dim_start, dim_count, int(tensor.shape[-1]))
    return tensor[:, head_slice, pos:pos + 1, dims]


def token_kv_layer_details(
    session: ModelSession,
    state: KVRuntimeState,
    query: str | None,
    search_mode: str = "auto",
    layers: str = "all",
    component_scope: str = "both",
    head_spec: str = "all",
    dim_start: int = 0,
    dim_count: int = 0,
    max_matches: int = 32,
    max_rows: int = 800,
) -> list[dict[str, Any]]:
    if not state.supported or state.cache is None:
        raise RuntimeError("KV debugger is not active or this model did not expose KV cache.")
    matches = [r for r in current_context_token_rows(session, state, query, search_mode, max_results=max_matches) if bool(r.get("cache_visible"))]
    layer_ids = _parse_int_ranges(layers, _available_layer_ids(state.cache))
    comps = _component_list(component_scope)
    rows: list[dict[str, Any]] = []
    for m in matches:
        if len(rows) >= int(max_rows):
            break
        kv_pos = int(m["kv_pos"])
        for layer_id in layer_ids:
            for comp in comps:
                tensor = get_cache_tensor(state.cache, layer_id, comp)
                head_count = int(tensor.shape[1]) if tensor.ndim >= 4 else 0
                heads = _parse_heads(head_spec, head_count) if head_count else [None]
                for h in heads:
                    if len(rows) >= int(max_rows):
                        break
                    view = _kv_token_view(tensor, kv_pos, h, int(dim_start), int(dim_count))
                    st = tensor_stats(view, sample=4096)
                    rows.append({
                        "pos": int(m["pos"]),
                        "kv_pos": kv_pos,
                        "token_id": int(m["token_id"]),
                        "token_text": m.get("token_text", ""),
                        "layer": int(layer_id),
                        "component": comp,
                        "head": "all" if h is None else int(h),
                        "slice": token_kv_slice_expr(kv_pos, "all" if h is None else h, int(dim_start), int(dim_count)),
                        "shape": st.get("shape"),
                        "dtype": st.get("dtype"),
                        "device": st.get("device"),
                        "mean": st.get("mean"),
                        "std": st.get("std"),
                        "rms": st.get("rms"),
                        "min": st.get("min"),
                        "max": st.get("max"),
                        "nan": st.get("nan"),
                    })
    return rows


def edit_cache_by_token_text(
    session: ModelSession,
    state: KVRuntimeState,
    query: str | None,
    search_mode: str,
    layers: str,
    component_scope: str,
    head_spec: str,
    dim_start: int,
    dim_count: int,
    edit_mode: str,
    value: float,
    strength: float,
    max_matches: int = 64,
    max_targets: int = 2048,
) -> dict[str, Any]:
    """Apply an in-place KV edit to token positions found by text search."""
    if not state.supported or state.cache is None:
        raise RuntimeError("KV debugger is not active or this model did not expose KV cache.")
    matches = [r for r in current_context_token_rows(session, state, query, search_mode, max_results=max_matches) if bool(r.get("cache_visible"))]
    if not matches:
        raise ValueError("No matching token positions are currently visible in the KV cache. For sliding-window caches, older context may have been evicted.")
    layer_ids = _parse_int_ranges(layers, _available_layer_ids(state.cache))
    if not layer_ids:
        raise ValueError("No KV layers selected.")
    comps = _component_list(component_scope)
    targets: list[dict[str, Any]] = []
    before_sample: list[dict[str, Any]] = []
    after_sample: list[dict[str, Any]] = []
    count = 0
    with torch.no_grad():
        for m in matches:
            kv_pos = int(m["kv_pos"])
            for layer_id in layer_ids:
                for comp in comps:
                    tensor = get_cache_tensor(state.cache, layer_id, comp)
                    head_count = int(tensor.shape[1]) if tensor.ndim >= 4 else 0
                    heads = _parse_heads(head_spec, head_count) if head_count else [None]
                    for h in heads:
                        if count >= int(max_targets):
                            raise RuntimeError(f"Token KV edit stopped at max_targets={max_targets}. Narrow your layer/head/component selection.")
                        view = _kv_token_view(tensor, kv_pos, h, int(dim_start), int(dim_count))
                        target = {
                            "pos": int(m["pos"]),
                            "kv_pos": kv_pos,
                            "token_id": int(m["token_id"]),
                            "token_text": m.get("token_text", ""),
                            "layer": int(layer_id),
                            "component": comp,
                            "head": "all" if h is None else int(h),
                            "slice": token_kv_slice_expr(kv_pos, "all" if h is None else h, int(dim_start), int(dim_count)),
                        }
                        if len(before_sample) < 16:
                            before_sample.append({**target, "stats": tensor_stats(view, sample=2048)})
                        apply_tensor_edit_(view, edit_mode, float(value), float(strength))
                        if len(after_sample) < 16:
                            after_sample.append({**target, "stats": tensor_stats(view, sample=2048)})
                        if len(targets) < 128:
                            targets.append(target)
                        count += 1
    state.status = f"Applied token-text KV edit to {count} target view(s) from {len(matches)} matched token position(s)."
    return {
        "timestamp": now_ts(),
        "target": "kv.token_text_search",
        "query": query or "",
        "search_mode": search_mode,
        "layers": layers,
        "components": component_scope,
        "heads": head_spec,
        "dim_start": int(dim_start),
        "dim_count": int(dim_count),
        "mode": edit_mode,
        "value": float(value),
        "strength": float(strength),
        "matched_positions": len(matches),
        "target_count": count,
        "targets": targets,
        "before_sample": before_sample,
        "after_sample": after_sample,
    }

def runtime_manifest(state: KVRuntimeState) -> dict[str, Any]:
    return {
        "model_id": state.model_id,
        "prompt_chars": len(state.prompt),
        "total_tokens": len(state.input_ids),
        "generated_tokens": len(state.generated_ids),
        "queued_token_id": state.queued_token_id,
        "steps": state.steps,
        "done": state.done,
        "supported": state.supported,
        "cache_format": cache_format_name(state.cache),
        "cache_layers": len(iter_cache_layers(state.cache)),
        "cache_seq_len": cache_seq_len(state.cache),
        "warnings": state.warnings,
        "status": state.status,
    }
