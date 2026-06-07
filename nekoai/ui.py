from __future__ import annotations

import base64
import inspect
import json
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
import pandas as pd

from .generation import DEFAULT_SYSTEM_PROMPT, build_prompt
from .kv_cache_debugger import (
    KVRuntimeState,
    auto_step,
    cache_anomalies,
    cache_summary,
    current_context_token_rows,
    current_context_tokens,
    decoded_text as kv_decoded_text,
    edit_cache_preview,
    edit_cache_preview_cell,
    edit_cache_slice,
    edit_cache_by_token_text,
    edit_cache_token_positions,
    edit_logits_from_table,
    edit_logits_value,
    execute_queued_token,
    export_kv_state,
    import_kv_state,
    inspect_cache_slice,
    inspect_head_vector,
    inspect_token_kv_slice,
    kv_top_tokens,
    prefill_kv,
    queue_next_token,
    runtime_manifest,
    search_tokens_in_state,
    token_kv_layer_details,
    token_kv_slice_expr,
    tokenize_text,
    tokenize_text_for_display,
    token_kv_index,
    token_kv_layer_stats,
)
from .model_manager import ModelSession, compatibility_report, default_export_dir, load_model_session
from .module_debugger import disassemble_modules
from .plotting import empty_figure, kv_rms_figure, weight_heatmap, weight_histogram
from .tensor_ops import dataframe_to_matrix
from .token_inspector import (
    TOKEN_COMPONENT_MODES,
    TOKEN_SEARCH_MODES,
    build_token_kv_slice,
    edit_token_positions,
    query_tokenization_rows,
    search_context_tokens,
    token_layer_info,
)
from .utils import tail, zip_dir
from .weight_index import inspect_parameter, model_parameter_summary, search_parameters
from .weight_ops import apply_preview_cell_edit, apply_preview_table_edits, apply_scalar_edit

ACTIVE_SESSION = ModelSession()
KV_STATE = KVRuntimeState()
KV_EDIT_LOG: list[dict[str, Any]] = []

EDIT_MODES = [
    "set",
    "add",
    "subtract",
    "multiply",
    "lerp_to",
    "zero",
    "clip_abs",
    "soft_clip",
    "noise",
    "center",
    "normalize_rms",
    "standardize",
    "nan_to_num",
]

LOGIT_EDIT_MODES = ["set_logit", "add_delta", "multiply", "boost", "suppress", "target_probability"]
TOP_TOKEN_COLUMNS = ["rank", "token_id", "token", "logit", "filtered_logit", "probability", "edited_logit", "logit_delta", "target_probability"]
TOP_TOKEN_TYPES = ["number", "number", "str", "number", "number", "number", "number", "number", "str"]
KV_TEXT_SEARCH_MODES = ["auto", "contains token text", "exact token text", "tokenized text sequence", "token id", "all tokens"]
KV_TEXT_COMPONENTS = ["both", "key", "value"]
KV_TEXT_TOKENIZE_COLUMNS = ["query_index", "token_id", "token_text", "token_repr"]
KV_TEXT_MATCH_COLUMNS = [
    "pos",
    "kv_pos",
    "cache_visible",
    "token_id",
    "token_text",
    "token_repr",
    "source",
    "match_kind",
    "query_token_index",
    "span",
    "span_text",
    "left_context",
    "right_context",
]
KV_TEXT_LAYER_COLUMNS = [
    "pos",
    "kv_pos",
    "token_id",
    "token_text",
    "layer",
    "component",
    "head",
    "slice",
    "shape",
    "dtype",
    "device",
    "mean",
    "std",
    "rms",
    "min",
    "max",
    "nan",
]
TOKEN_SEARCH_COLUMNS = [
    "position",
    "position_from_end",
    "kv_position",
    "token_id",
    "token_text",
    "token_repr",
    "source",
    "is_special",
    "in_kv_cache",
    "match",
    "match_kind",
    "span_start",
    "span_end",
    "kv_slice_all_heads",
    "kv_window_offset",
    "cache_seq_len",
]
TOKEN_SEARCH_TYPES = ["number", "number", "number", "number", "str", "str", "str", "bool", "bool", "bool", "str", "str", "str", "str", "number", "number"]
TOKENIZE_COLUMNS = ["variant", "variant_repr", "token_index", "token_id", "token_text", "token_repr", "sequence_len", "sequence_ids", "error"]
TOKEN_LAYER_COLUMNS = [
    "position",
    "token_id",
    "token_text",
    "token_repr",
    "layer",
    "component",
    "source",
    "shape",
    "dtype",
    "device",
    "heads",
    "head_dim",
    "mean",
    "std",
    "rms",
    "min",
    "max",
    "nan",
    "head_rms_min",
    "head_rms_max",
    "head_rms_argmax",
    "kv_slice_all_heads",
    "error",
]
CONTEXT_TOKEN_COLUMNS = TOKEN_SEARCH_COLUMNS



def _session() -> ModelSession:
    return ACTIVE_SESSION


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _top_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    for col in TOP_TOKEN_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[TOP_TOKEN_COLUMNS]


def _table_records(table: Any) -> list[dict[str, Any]]:
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


def _top_state(df: Any) -> dict[str, Any]:
    return {"rows": _table_records(df)}


def _ordered_df(rows: list[dict[str, Any]] | None, columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns]


def _token_search_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    return _ordered_df(rows, TOKEN_SEARCH_COLUMNS)


def _tokenize_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    return _ordered_df(rows, TOKENIZE_COLUMNS)


def _token_layer_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    return _ordered_df(rows, TOKEN_LAYER_COLUMNS)


def _context_token_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    return _ordered_df(rows, CONTEXT_TOKEN_COLUMNS)


def _kv_text_match_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    return _ordered_df(rows, KV_TEXT_MATCH_COLUMNS)


def _kv_text_layer_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    return _ordered_df(rows, KV_TEXT_LAYER_COLUMNS)


def _kv_text_tokenize_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    return _ordered_df(rows, KV_TEXT_TOKENIZE_COLUMNS)


def _positions_from_table(table: Any) -> list[int]:
    """Return KV-local positions from a table; fall back to absolute positions for old tables."""
    out: list[int] = []
    for row in _table_records(table):
        value = row.get("kv_position")
        if value is None or str(value).strip() == "":
            value = row.get("position")
        try:
            out.append(int(float(value)))
        except Exception:
            continue
    return out


def _kv_position_from_row(row: dict[str, Any]) -> int:
    value = row.get("kv_position")
    if value is None or str(value).strip() == "":
        value = row.get("position")
    return int(float(value))


def _selected_row_dict(evt: gr.SelectData, columns: list[str]) -> dict[str, Any]:
    row = _selected_row_value(evt)
    return {col: row[i] if i < len(row) else None for i, col in enumerate(columns)}


def _selected_row_value(evt: gr.SelectData) -> list[Any]:
    row = getattr(evt, "row_value", None)
    if row is not None:
        return list(row)
    value = getattr(evt, "value", None)
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _event_index(evt: gr.SelectData) -> tuple[int, int]:
    idx = getattr(evt, "index", None)
    if isinstance(idx, (list, tuple)) and len(idx) >= 2:
        return int(idx[0]), int(idx[1])
    if isinstance(idx, int):
        return int(idx), 0
    return 0, 0


def _event_value(evt: gr.SelectData) -> Any:
    return getattr(evt, "value", "")


def _err(prefix: str, exc: BaseException) -> str:
    return f"❌ {prefix}: {type(exc).__name__}: {exc}"


def _preview_state(name: str, index_expr: str, df: Any) -> dict[str, Any]:
    return {"parameter": (name or "").strip(), "index_expr": index_expr or ":", "preview": dataframe_to_matrix(df)}


def _kv_slice_state(layer: int, component: str, index_expr: str, df: Any) -> dict[str, Any]:
    return {"layer": int(layer), "component": component, "index_expr": index_expr or ":", "preview": dataframe_to_matrix(df)}


def _kv_outputs(temperature: float = 0.0, top_k: int = 50, top_p: float = 0.95):
    rows = cache_summary(KV_STATE.cache) if KV_STATE.supported else []
    warnings = "\n".join([f"- {w}" for w in KV_STATE.warnings])
    status = KV_STATE.status + ("\n\nWarnings:\n" + warnings if warnings else "")
    try:
        queued = "None" if KV_STATE.queued_token_id is None else f"{KV_STATE.queued_token_id} / {repr(_session().tokenizer.decode([KV_STATE.queued_token_id], skip_special_tokens=False))}"
    except Exception:
        queued = str(KV_STATE.queued_token_id)
    try:
        decoded = kv_decoded_text(_session(), KV_STATE) if _session().loaded and KV_STATE.input_ids else ""
    except Exception as exc:
        decoded = f"Decode failed: {exc}"
    try:
        top = kv_top_tokens(_session(), KV_STATE, temperature, int(top_k), float(top_p)) if _session().loaded and KV_STATE.supported else []
    except Exception as exc:
        top = [{"error": str(exc)}]
    try:
        anomalies = cache_anomalies(KV_STATE.cache) if KV_STATE.supported else []
    except Exception as exc:
        anomalies = [{"error": str(exc)}]
    top_df = _top_df(top if isinstance(top, list) else [])
    return (
        status,
        _json(runtime_manifest(KV_STATE)),
        _df(rows),
        kv_rms_figure(rows),
        top_df,
        decoded,
        queued,
        _json(tail(KV_EDIT_LOG, 100)),
        _df(anomalies),
        _top_state(top_df),
    )


def _kv_error_outputs(message: str):
    return (
        f"❌ {message}",
        _json(runtime_manifest(KV_STATE)),
        _empty_df(),
        empty_figure("KV error"),
        _top_df([]),
        "",
        "None",
        _json(tail(KV_EDIT_LOG, 100)),
        _empty_df(),
        _top_state(_top_df([])),
    )


def load_model_cb(model_id: str, device_mode: str, dtype_name: str, trust_remote_code: bool, use_device_map: bool, revision: str):
    global ACTIVE_SESSION, KV_STATE, KV_EDIT_LOG
    try:
        ACTIVE_SESSION = load_model_session(model_id, device_mode, dtype_name, trust_remote_code, use_device_map, revision.strip() or None)
        KV_STATE = KVRuntimeState()
        KV_EDIT_LOG = []
        summary = model_parameter_summary(ACTIVE_SESSION)
        return f"✅ Loaded {model_id}. Parameters: {summary['parameters']:,}.", _json(summary), compatibility_report(), default_export_dir(model_id)
    except Exception as exc:
        return _err("Load model failed", exc), "{}", compatibility_report(), default_export_dir(model_id or "model")


def refresh_compat_cb() -> str:
    try:
        return compatibility_report()
    except Exception as exc:
        return _err("Hardware report failed", exc)


def build_chat_prompt_cb(system_prompt: str, user_prompt: str) -> str:
    try:
        _session().require_loaded()
        return build_prompt(_session().tokenizer, system_prompt, user_prompt)
    except Exception:
        # Still useful before loading a tokenizer.
        return f"{system_prompt or DEFAULT_SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"


def prefill_kv_cb(prompt: str, dtype_mode: str, temperature: float, top_k: int, top_p: float):
    global KV_STATE
    try:
        KV_STATE = prefill_kv(_session(), prompt, dtype_mode=dtype_mode, force_cache=True)
        return _kv_outputs(temperature, int(top_k), float(top_p))
    except Exception as exc:
        return _kv_error_outputs(_err("Prefill failed", exc))


def queue_token_cb(temperature: float, top_k: int, top_p: float, token_override: str, logit_bias_json: str):
    global KV_STATE
    try:
        KV_STATE = queue_next_token(_session(), KV_STATE, temperature=float(temperature), top_k=int(top_k), top_p=float(top_p), token_override=token_override, logit_bias_json=logit_bias_json)
        return _kv_outputs(temperature, int(top_k), float(top_p))
    except Exception as exc:
        return _kv_error_outputs(_err("Queue token failed", exc))


def execute_token_cb(dtype_mode: str, temperature: float, top_k: int, top_p: float):
    global KV_STATE
    try:
        KV_STATE = execute_queued_token(_session(), KV_STATE, dtype_mode=dtype_mode)
        return _kv_outputs(temperature, int(top_k), float(top_p))
    except Exception as exc:
        return _kv_error_outputs(_err("Execute token failed", exc))


def auto_step_cb(temperature: float, top_k: int, top_p: float, dtype_mode: str, token_override: str, logit_bias_json: str):
    global KV_STATE
    try:
        KV_STATE = auto_step(_session(), KV_STATE, temperature=float(temperature), top_k=int(top_k), top_p=float(top_p), dtype_mode=dtype_mode, token_override=token_override, logit_bias_json=logit_bias_json)
        return _kv_outputs(temperature, int(top_k), float(top_p))
    except Exception as exc:
        return _kv_error_outputs(_err("Auto step failed", exc))


def run_steps_cb(n_steps: int, temperature: float, top_k: int, top_p: float, dtype_mode: str, logit_bias_json: str) -> Iterator[tuple[Any, ...]]:
    global KV_STATE
    for _ in range(max(1, int(n_steps))):
        if KV_STATE.done:
            break
        try:
            KV_STATE = auto_step(_session(), KV_STATE, temperature=float(temperature), top_k=int(top_k), top_p=float(top_p), dtype_mode=dtype_mode, token_override="", logit_bias_json=logit_bias_json)
            yield _kv_outputs(temperature, int(top_k), float(top_p))
        except Exception as exc:
            yield _kv_error_outputs(_err("Run steps failed", exc))
            break


def inspect_kv_cb(layer: int, component: str, index_expr: str):
    try:
        data = inspect_cache_slice(KV_STATE.cache, int(layer), component, index_expr or ":")
        df = pd.DataFrame(data["preview"])
        return _json(data["stats"]), df, _kv_slice_state(int(layer), component, index_expr or ":", df)
    except Exception as exc:
        return _json({"error": _err("Inspect KV failed", exc)}), _empty_df(), {}


def inspect_kv_head_cb(layer: int, component: str, head: int, token_pos: int, dim_start: int, dim_count: int):
    try:
        data = inspect_head_vector(KV_STATE.cache, int(layer), component, int(head), int(token_pos), int(dim_start), int(dim_count))
        df = pd.DataFrame(data["preview"])
        idx = data["slice"]
        return idx, _json(data["stats"]), df, _kv_slice_state(int(layer), component, idx, df)
    except Exception as exc:
        return ":", _json({"error": _err("Inspect head vector failed", exc)}), _empty_df(), {}


def edit_kv_cb(layer: int, component: str, index_expr: str, mode: str, value: float, strength: float, temperature: float, top_k: int, top_p: float):
    try:
        record = edit_cache_slice(KV_STATE.cache, int(layer), component, index_expr or ":", mode, float(value), float(strength))
        KV_EDIT_LOG.append(record)
        stats = inspect_cache_slice(KV_STATE.cache, int(layer), component, index_expr or ":")
        df = pd.DataFrame(stats["preview"])
        outputs = _kv_outputs(temperature, int(top_k), float(top_p))
        return (*outputs, _json(stats["stats"]), df, _kv_slice_state(int(layer), component, index_expr or ":", df))
    except Exception as exc:
        return (*_kv_error_outputs(_err("Apply KV edit failed", exc)), _json({"error": str(exc)}), _empty_df(), {})


def edit_kv_preview_cb(new_preview: Any, layer: int, component: str, index_expr: str, preview_state: dict[str, Any], temperature: float, top_k: int, top_p: float):
    try:
        state = preview_state or {}
        old_preview = state.get("preview")
        if not old_preview:
            stats = inspect_cache_slice(KV_STATE.cache, int(layer), component, index_expr or ":")
            df = pd.DataFrame(stats["preview"])
            return (*_kv_outputs(temperature, int(top_k), float(top_p)), _json(stats["stats"]), df, _kv_slice_state(int(layer), component, index_expr or ":", df))
        record = edit_cache_preview(KV_STATE.cache, int(layer), component, index_expr or ":", old_preview, new_preview)
        if record:
            KV_EDIT_LOG.append(record)
        stats = inspect_cache_slice(KV_STATE.cache, int(layer), component, index_expr or ":")
        df = pd.DataFrame(stats["preview"])
        return (*_kv_outputs(temperature, int(top_k), float(top_p)), _json(stats["stats"]), df, _kv_slice_state(int(layer), component, index_expr or ":", df))
    except Exception as exc:
        return (*_kv_error_outputs(_err("KV preview grid edit failed", exc)), _json({"error": str(exc)}), _empty_df(), {})


def edit_kv_cell_cb(preview_table: Any, row: int, col: int, value: float, layer: int, component: str, index_expr: str, preview_state: dict[str, Any], temperature: float, top_k: int, top_p: float):
    try:
        state = preview_state or {}
        old_preview = state.get("preview") or dataframe_to_matrix(preview_table)
        record = edit_cache_preview_cell(KV_STATE.cache, int(layer), component, index_expr or ":", old_preview, int(row), int(col), value)
        if record:
            KV_EDIT_LOG.append(record)
        stats = inspect_cache_slice(KV_STATE.cache, int(layer), component, index_expr or ":")
        df = pd.DataFrame(stats["preview"])
        status = _json(stats["stats"])
        return (*_kv_outputs(temperature, int(top_k), float(top_p)), status, df, _kv_slice_state(int(layer), component, index_expr or ":", df))
    except Exception as exc:
        return (*_kv_error_outputs(_err("KV cell write failed", exc)), _json({"error": str(exc)}), _empty_df(), {})


def select_kv_preview_cell_cb(evt: gr.SelectData):
    try:
        row, col = _event_index(evt)
        return row, col, _event_value(evt)
    except Exception:
        return 0, 0, 0.0


def select_weight_preview_cell_cb(evt: gr.SelectData):
    try:
        row, col = _event_index(evt)
        return row, col, _event_value(evt)
    except Exception:
        return 0, 0, 0.0


def select_top_token_row_cb(evt: gr.SelectData):
    try:
        row = _selected_row_value(evt)
        token_id = int(float(row[1])) if len(row) > 1 else 0
        token_text = str(row[2]) if len(row) > 2 else ""
        logit = float(row[3]) if len(row) > 3 else 0.0
        return token_id, token_text, logit, f"✅ Selected token {token_id} {token_text}"
    except Exception as exc:
        return 0, "", 0.0, _err("Top-token row select failed", exc)


def apply_top_table_edits_cb(top_table: Any, top_state: dict[str, Any], temperature: float, top_k: int, top_p: float):
    try:
        old_table = top_state or {"rows": []}
        record = edit_logits_from_table(KV_STATE, old_table, top_table)
        if record:
            KV_EDIT_LOG.append(record)
        elif KV_STATE.supported:
            KV_STATE.status = "No top-token table edit detected. Edit edited_logit, logit_delta, or target_probability, then apply."
        return _kv_outputs(temperature, int(top_k), float(top_p))
    except Exception as exc:
        return _kv_error_outputs(_err("Top-token table edit failed", exc))


def apply_top_logit_cb(token_id: int, mode: str, value: float, strength: float, temperature: float, top_k: int, top_p: float):
    try:
        record = edit_logits_value(KV_STATE, int(token_id), mode, float(value), float(strength))
        KV_EDIT_LOG.append(record)
        return _kv_outputs(temperature, int(top_k), float(top_p))
    except Exception as exc:
        return _kv_error_outputs(_err("Top-token logit edit failed", exc))


def queue_selected_top_token_cb(token_id: int, temperature: float, top_k: int, top_p: float):
    try:
        if not KV_STATE.supported:
            raise RuntimeError("KV debugger is not active.")
        KV_STATE.queued_token_id = int(token_id)
        try:
            token_text = repr(_session().tokenizer.decode([int(token_id)], skip_special_tokens=False))
        except Exception:
            token_text = ""
        KV_STATE.status = f"Queued selected top-table token id {int(token_id)} {token_text}"
        return _kv_outputs(temperature, int(top_k), float(top_p))
    except Exception as exc:
        return _kv_error_outputs(_err("Queue selected token failed", exc))


def select_kv_row_cb(evt: gr.SelectData):
    try:
        row = evt.row_value or []
        layer = int(row[0] if row else evt.value)
        data = inspect_cache_slice(KV_STATE.cache, layer, "key", ":")
        df = pd.DataFrame(data["preview"])
        return layer, "key", ":", _json(data["stats"]), df, _kv_slice_state(layer, "key", ":", df)
    except Exception as exc:
        return 0, "key", ":", _json({"error": _err("KV row select failed", exc)}), _empty_df(), {}


TOKEN_MATCH_MODES = ["auto", "encoded phrase", "token id", "decoded token contains", "decoded token exact"]
TOKEN_KV_SCOPES = ["all heads / full dim", "selected head / full dim", "all heads / dim range", "selected head / dim range"]


def _scope_to_head_dim(scope: str, head: int | float | None, dim_count: int | float | None) -> tuple[int | None, int]:
    text = (scope or "all heads / full dim").lower()
    selected_head = "selected head" in text
    dim_range = "dim range" in text
    h = int(head) if selected_head and head is not None else None
    if h is not None and h < 0:
        h = None
    count = int(dim_count or 0) if dim_range else 0
    return h, max(0, count)


def _first_position_from_table(table: Any) -> int | None:
    positions = _positions_from_table(table)
    return positions[0] if positions else None


def search_kv_tokens_cb(query: str, mode: str, max_matches: int, context_radius: int):
    try:
        rows = search_tokens_in_state(_session(), KV_STATE, query, mode=mode, max_matches=int(max_matches), context_radius=int(context_radius))
        msg = f"✅ Found {len(rows)} token position match(es). Click a row to build a KV slice, or apply an edit to all matches."
        if not rows:
            msg = "No token positions matched. Try mode='encoded phrase' for full text, or 'decoded token contains' for a token fragment."
        return _token_search_df(rows), msg
    except Exception as exc:
        return _token_search_df([]), _err("Token search failed", exc)


def select_kv_token_row_cb(evt: gr.SelectData, layer: int, component: str, scope: str, head: int, dim_start: int, dim_count: int):
    try:
        row = _selected_row_dict(evt, TOKEN_SEARCH_COLUMNS)
        pos = int(float(row.get("position")))
        h, count = _scope_to_head_dim(scope, head, dim_count)
        index_expr = token_kv_index(KV_STATE.cache, int(layer), component, pos, head=h, dim_start=int(dim_start), dim_count=count)
        data = inspect_cache_slice(KV_STATE.cache, int(layer), component, index_expr)
        df = pd.DataFrame(data["preview"])
        text = row.get("token_text") or ""
        msg = f"✅ Selected token position {pos} {text}; KV slice set to `{index_expr}`."
        return pos, index_expr, _json(data["stats"]), df, _kv_slice_state(int(layer), component, index_expr, df), msg
    except Exception as exc:
        return -1, ":", _json({"error": _err("Token row select failed", exc)}), _empty_df(), {}, _err("Token row select failed", exc)


def apply_kv_token_matches_cb(token_table: Any, scope: str, layer: int, component: str, mode: str, value: float, strength: float, head: int, dim_start: int, dim_count: int, temperature: float, top_k: int, top_p: float):
    try:
        positions = _positions_from_table(token_table)
        if not positions:
            raise ValueError("Search for token text first; the result table is empty.")
        h, count = _scope_to_head_dim(scope, head, dim_count)
        record = edit_cache_token_positions(KV_STATE.cache, positions, int(layer), component, mode, float(value), float(strength), head=h, dim_start=int(dim_start), dim_count=count)
        KV_EDIT_LOG.append(record)
        first = int(record["positions"][0])
        index_expr = token_kv_index(KV_STATE.cache, int(layer), component, first, head=h, dim_start=int(dim_start), dim_count=count)
        data = inspect_cache_slice(KV_STATE.cache, int(layer), component, index_expr)
        df = pd.DataFrame(data["preview"])
        status = f"✅ Applied `{mode}` to {record['position_count']} matched token position(s) on layer {int(layer)} {component}. First slice: `{index_expr}`"
        return (status, *_kv_outputs(temperature, int(top_k), float(top_p)), _json(data["stats"]), df, _kv_slice_state(int(layer), component, index_expr, df))
    except Exception as exc:
        return (_err("Token-match KV edit failed", exc), *_kv_error_outputs(_err("Token-match KV edit failed", exc)), _json({"error": str(exc)}), _empty_df(), {})


def token_inspector_search_cb(query: str, mode: str, add_special_tokens: bool, max_matches: int, context_radius: int, context_limit: int):
    try:
        token_rows = tokenize_text(_session(), query or "", add_special_tokens=bool(add_special_tokens)) if (query or "").strip() else []
        context_rows = current_context_tokens(_session(), KV_STATE, limit=int(context_limit), context_radius=0)
        match_rows = []
        if (query or "").strip():
            match_rows = search_tokens_in_state(_session(), KV_STATE, query, mode=mode, max_matches=int(max_matches), context_radius=int(context_radius), add_special_tokens=bool(add_special_tokens))
        info = {
            "query": query,
            "query_token_count": len(token_rows),
            "match_count": len(match_rows),
            "context_token_count_shown": len(context_rows),
            "runtime": runtime_manifest(KV_STATE),
        }
        status = f"✅ Tokenized into {len(token_rows)} token(s); found {len(match_rows)} active-context match(es)."
        if not (query or "").strip():
            status = f"✅ Showing {len(context_rows)} current context token(s). Enter text to search positions."
        return _tokenize_df(token_rows), _token_search_df(match_rows), _context_token_df(context_rows), _token_layer_df([]), _json(info), status
    except Exception as exc:
        return _tokenize_df([]), _token_search_df([]), _context_token_df([]), _token_layer_df([]), _json({"error": _err("Token inspector search failed", exc)}), _err("Token inspector search failed", exc)


def _inspect_token_position_outputs(pos: int, layer: int, component: str, scope: str, head: int, dim_start: int, dim_count: int, status_prefix: str = "✅ Inspected"):
    h, count = _scope_to_head_dim(scope, head, dim_count)
    layer_rows = token_kv_layer_stats(KV_STATE.cache, int(pos))
    data = inspect_token_kv_slice(KV_STATE.cache, int(pos), int(layer), component, head=h, dim_start=int(dim_start), dim_count=count)
    df = pd.DataFrame(data["preview"])
    index_expr = str(data["slice"])
    token_id = KV_STATE.input_ids[int(pos)] if 0 <= int(pos) < len(KV_STATE.input_ids) else None
    token_text = _session().tokenizer.decode([int(token_id)], skip_special_tokens=False) if token_id is not None else ""
    info = {
        "position": int(pos),
        "token_id": token_id,
        "token_text": token_text,
        "layer": int(layer),
        "component": component,
        "scope": scope,
        "head": h,
        "dim_start": int(dim_start),
        "dim_count": count,
        "slice": index_expr,
        "runtime": runtime_manifest(KV_STATE),
    }
    return (
        int(pos),
        int(token_id) if token_id is not None else -1,
        repr(token_text),
        _token_layer_df(layer_rows),
        _json(info),
        index_expr,
        _json(data["stats"]),
        df,
        _kv_slice_state(int(layer), component, index_expr, df),
        f"{status_prefix} token position {int(pos)} {repr(token_text)} at layer {int(layer)} {component}.",
    )


def select_token_inspector_row_cb(evt: gr.SelectData, layer: int, component: str, scope: str, head: int, dim_start: int, dim_count: int):
    try:
        row = _selected_row_dict(evt, TOKEN_SEARCH_COLUMNS)
        pos_value = row.get("position")
        if pos_value is None:
            row_ctx = _selected_row_dict(evt, CONTEXT_TOKEN_COLUMNS)
            pos_value = row_ctx.get("position")
        pos = int(float(pos_value))
        return _inspect_token_position_outputs(pos, int(layer), component, scope, int(head), int(dim_start), int(dim_count), "✅ Selected")
    except Exception as exc:
        return (-1, -1, "", _token_layer_df([]), _json({"error": _err("Token selection failed", exc)}), ":", _json({"error": str(exc)}), _empty_df(), {}, _err("Token selection failed", exc))


def select_context_inspector_row_cb(evt: gr.SelectData, layer: int, component: str, scope: str, head: int, dim_start: int, dim_count: int):
    try:
        row = _selected_row_dict(evt, CONTEXT_TOKEN_COLUMNS)
        pos = int(float(row.get("position")))
        return _inspect_token_position_outputs(pos, int(layer), component, scope, int(head), int(dim_start), int(dim_count), "✅ Selected")
    except Exception as exc:
        return (-1, -1, "", _token_layer_df([]), _json({"error": _err("Context token selection failed", exc)}), ":", _json({"error": str(exc)}), _empty_df(), {}, _err("Context token selection failed", exc))


def inspect_selected_token_cb(pos: int, layer: int, component: str, scope: str, head: int, dim_start: int, dim_count: int):
    try:
        return _inspect_token_position_outputs(int(pos), int(layer), component, scope, int(head), int(dim_start), int(dim_count), "✅ Inspected")
    except Exception as exc:
        return (-1, -1, "", _token_layer_df([]), _json({"error": _err("Selected token inspect failed", exc)}), ":", _json({"error": str(exc)}), _empty_df(), {}, _err("Selected token inspect failed", exc))


def send_token_to_kv_editor_cb(pos: int, layer: int, component: str, scope: str, head: int, dim_start: int, dim_count: int):
    try:
        h, count = _scope_to_head_dim(scope, head, dim_count)
        index_expr = token_kv_index(KV_STATE.cache, int(layer), component, int(pos), head=h, dim_start=int(dim_start), dim_count=count)
        data = inspect_cache_slice(KV_STATE.cache, int(layer), component, index_expr)
        df = pd.DataFrame(data["preview"])
        return int(layer), component, int(pos), index_expr, _json(data["stats"]), df, _kv_slice_state(int(layer), component, index_expr, df), f"✅ Sent token position {int(pos)} to KV editor slice `{index_expr}`."
    except Exception as exc:
        return int(layer or 0), component or "key", -1, ":", _json({"error": str(exc)}), _empty_df(), {}, _err("Send to KV editor failed", exc)


def inspector_apply_token_edit_cb(pos: int, layer: int, component: str, scope: str, head: int, dim_start: int, dim_count: int, mode: str, value: float, strength: float, temperature: float, top_k: int, top_p: float):
    try:
        h, count = _scope_to_head_dim(scope, head, dim_count)
        record = edit_cache_token_positions(KV_STATE.cache, [int(pos)], int(layer), component, mode, float(value), float(strength), head=h, dim_start=int(dim_start), dim_count=count)
        KV_EDIT_LOG.append(record)
        inspector = _inspect_token_position_outputs(int(pos), int(layer), component, scope, int(head), int(dim_start), int(dim_count), f"✅ Applied {mode} edit to")
        return (*inspector, *_kv_outputs(temperature, int(top_k), float(top_p)))
    except Exception as exc:
        err = _err("Inspector token edit failed", exc)
        return (-1, -1, "", _token_layer_df([]), _json({"error": err}), ":", _json({"error": str(exc)}), _empty_df(), {}, err, *_kv_error_outputs(err))


def select_inspector_preview_cell_cb(evt: gr.SelectData):
    return select_kv_preview_cell_cb(evt)


def edit_inspector_cell_cb(preview_table: Any, row: int, col: int, value: float, pos: int, layer: int, component: str, index_expr: str, preview_state: dict[str, Any], temperature: float, top_k: int, top_p: float):
    try:
        state = preview_state or {}
        old_preview = state.get("preview") or dataframe_to_matrix(preview_table)
        record = edit_cache_preview_cell(KV_STATE.cache, int(layer), component, index_expr or ":", old_preview, int(row), int(col), value)
        if record:
            record["token_pos"] = int(pos)
            record["source"] = "token_inspector_cell"
            KV_EDIT_LOG.append(record)
        # Refresh with the existing index expression so the edited cell stays visible.
        data = inspect_cache_slice(KV_STATE.cache, int(layer), component, index_expr or ":")
        df = pd.DataFrame(data["preview"])
        layer_rows = token_kv_layer_stats(KV_STATE.cache, int(pos))
        token_id = KV_STATE.input_ids[int(pos)] if 0 <= int(pos) < len(KV_STATE.input_ids) else -1
        token_text = _session().tokenizer.decode([int(token_id)], skip_special_tokens=False) if token_id >= 0 else ""
        info = {"position": int(pos), "token_id": int(token_id), "token_text": token_text, "layer": int(layer), "component": component, "slice": index_expr}
        status = f"✅ Wrote inspector preview cell [{int(row)}, {int(col)}] at token position {int(pos)}."
        return (int(pos), int(token_id), repr(token_text), _token_layer_df(layer_rows), _json(info), index_expr or ":", _json(data["stats"]), df, _kv_slice_state(int(layer), component, index_expr or ":", df), status, *_kv_outputs(temperature, int(top_k), float(top_p)))
    except Exception as exc:
        err = _err("Inspector cell write failed", exc)
        return (-1, -1, "", _token_layer_df([]), _json({"error": err}), index_expr or ":", _json({"error": str(exc)}), _empty_df(), {}, err, *_kv_error_outputs(err))


def export_kv_cb(export_root: str, export_dtype: str):
    try:
        zip_path = export_kv_state(KV_STATE, export_root=export_root or "exports/kv_cache", export_dtype=export_dtype)
        return f"✅ Exported KV state: {zip_path}", zip_path
    except Exception as exc:
        return _err("Export KV failed", exc), None


def import_kv_cb(file_obj: Any, dtype_mode: str, temperature: float, top_k: int, top_p: float):
    global KV_STATE
    try:
        if file_obj is None:
            raise ValueError("Select a .zip or cache.pt file first.")
        path = getattr(file_obj, "name", file_obj)
        KV_STATE = import_kv_state(_session(), path, dtype_mode=dtype_mode)
        return _kv_outputs(temperature, int(top_k), float(top_p))
    except Exception as exc:
        return _kv_error_outputs(_err("Import KV failed", exc))


def search_kv_text_tokens_cb(query: str, mode: str, max_matches: int):
    try:
        token_rows = tokenize_text_for_display(_session(), query or "")
        match_rows = current_context_token_rows(_session(), KV_STATE, query or "", mode or "auto", max_results=int(max_matches))
        visible = sum(1 for r in match_rows if bool(r.get("cache_visible")))
        status = f"✅ Found {len(match_rows)} token row(s); {visible} are currently visible in KV cache."
        if not match_rows:
            status = "No token match found in the current context. Try mode='all tokens', a leading-space word, or a token id."
        return status, _kv_text_tokenize_df(token_rows), _kv_text_match_df(match_rows)
    except Exception as exc:
        return _err("Token text search failed", exc), _kv_text_tokenize_df([]), _kv_text_match_df([])


def analyze_kv_text_tokens_cb(query: str, mode: str, layers: str, components: str, heads: str, dim_start: int, dim_count: int, max_matches: int):
    try:
        token_rows = tokenize_text_for_display(_session(), query or "")
        match_rows = current_context_token_rows(_session(), KV_STATE, query or "", mode or "auto", max_results=int(max_matches))
        layer_rows = token_kv_layer_details(
            _session(),
            KV_STATE,
            query or "",
            search_mode=mode or "auto",
            layers=layers or "all",
            component_scope=components or "both",
            head_spec=heads or "all",
            dim_start=int(dim_start or 0),
            dim_count=int(dim_count or 0),
            max_matches=int(max_matches),
        )
        status = f"✅ Token inspector: {len(match_rows)} matched token row(s), {len(layer_rows)} layer/detail row(s)."
        return status, _kv_text_tokenize_df(token_rows), _kv_text_match_df(match_rows), _kv_text_layer_df(layer_rows)
    except Exception as exc:
        return _err("Token/KV analysis failed", exc), _kv_text_tokenize_df([]), _kv_text_match_df([]), _kv_text_layer_df([])


def select_kv_text_match_cb(layer: int, component: str, heads: str, dim_start: int, dim_count: int, evt: gr.SelectData):
    try:
        row = _selected_row_dict(evt, KV_TEXT_MATCH_COLUMNS)
        kv_pos_raw = row.get("kv_pos")
        if kv_pos_raw in {None, ""}:
            raise ValueError("Selected token is not visible in the current KV cache window.")
        kv_pos = int(float(kv_pos_raw))
        head_text = (heads or "all").strip()
        # Selecting a multi-head expression with commas is not valid in the direct slice editor.
        if "," in head_text or "-" in head_text:
            head_text = "all"
        if head_text.lower() in {"", "all", "*", ":"}:
            head_for_slice: str | int | None = "all"
        else:
            head_for_slice = int(float(head_text))
        index_expr = token_kv_slice_expr(kv_pos, head_for_slice, int(dim_start or 0), int(dim_count or 0))
        data = inspect_cache_slice(KV_STATE.cache, int(layer), component or "key", index_expr)
        df = pd.DataFrame(data["preview"])
        status = f"✅ Bound token pos={row.get('pos')} kv_pos={kv_pos} token={row.get('token_repr')} to KV editor."
        return int(layer), component or "key", index_expr, kv_pos, _json(data["stats"]), df, _kv_slice_state(int(layer), component or "key", index_expr, df), status
    except Exception as exc:
        return int(layer or 0), component or "key", ":", -1, _json({"error": _err("Select token match failed", exc)}), _empty_df(), {}, _err("Select token match failed", exc)


def select_kv_text_layer_cb(evt: gr.SelectData):
    try:
        row = _selected_row_dict(evt, KV_TEXT_LAYER_COLUMNS)
        layer = int(float(row.get("layer") or 0))
        component = str(row.get("component") or "key")
        index_expr = str(row.get("slice") or ":")
        data = inspect_cache_slice(KV_STATE.cache, layer, component, index_expr)
        df = pd.DataFrame(data["preview"])
        status = f"✅ Loaded layer={layer} component={component} slice={index_expr} into KV editor."
        return layer, component, index_expr, _json(data["stats"]), df, _kv_slice_state(layer, component, index_expr, df), status
    except Exception as exc:
        return 0, "key", ":", _json({"error": _err("Select token layer failed", exc)}), _empty_df(), {}, _err("Select token layer failed", exc)


def apply_kv_text_edit_cb(
    query: str,
    mode: str,
    layers: str,
    components: str,
    heads: str,
    dim_start: int,
    dim_count: int,
    max_matches: int,
    edit_mode: str,
    value: float,
    strength: float,
    temperature: float,
    top_k: int,
    top_p: float,
):
    try:
        record = edit_cache_by_token_text(
            _session(),
            KV_STATE,
            query or "",
            mode or "auto",
            layers or "all",
            components or "both",
            heads or "all",
            int(dim_start or 0),
            int(dim_count or 0),
            edit_mode or "add",
            float(value),
            float(strength),
            max_matches=int(max_matches),
        )
        KV_EDIT_LOG.append(record)
        layer_rows = token_kv_layer_details(
            _session(),
            KV_STATE,
            query or "",
            search_mode=mode or "auto",
            layers=layers or "all",
            component_scope=components or "both",
            head_spec=heads or "all",
            dim_start=int(dim_start or 0),
            dim_count=int(dim_count or 0),
            max_matches=int(max_matches),
        )
        status = f"✅ Applied token-text KV edit. matched_positions={record.get('matched_positions')}, target_count={record.get('target_count')}"
        return (*_kv_outputs(temperature, int(top_k), float(top_p)), status, _kv_text_layer_df(layer_rows))
    except Exception as exc:
        return (*_kv_error_outputs(_err("Token-text KV edit failed", exc)), _err("Token-text KV edit failed", exc), _kv_text_layer_df([]))




def token_search_cb(query: str, mode: str, case_sensitive: bool, max_rows: int):
    try:
        rows, meta = search_context_tokens(_session(), KV_STATE, query, mode, bool(case_sensitive), int(max_rows))
        return _token_search_df(rows), _json(meta)
    except Exception as exc:
        return _token_search_df([]), _json({"error": _err("Token search failed", exc)})


def token_query_tokenize_cb(query: str):
    try:
        return _tokenize_df(query_tokenization_rows(_session(), query))
    except Exception as exc:
        return _tokenize_df([{"error": _err("Tokenize failed", exc)}])


def select_token_match_for_kv_cb(evt: gr.SelectData, layer: int, component: str, head_expr: str, dim_expr: str):
    try:
        row = _selected_row_dict(evt, TOKEN_SEARCH_COLUMNS)
        kv_pos = _kv_position_from_row(row)
        expr = build_token_kv_slice(kv_pos, head_expr or ":", dim_expr or ":")
        comp = component if component in {"key", "value"} else "key"
        data = inspect_cache_slice(KV_STATE.cache, int(layer), comp, expr)
        df = pd.DataFrame(data["preview"])
        status = f"✅ Selected token position={row.get('position')} kv_position={kv_pos}; slice={expr}"
        return int(row.get("position") or kv_pos), kv_pos, expr, _json(data["stats"]), df, _kv_slice_state(int(layer), comp, expr, df), status
    except Exception as exc:
        return 0, 0, ":", _json({"error": _err("Token match select failed", exc)}), _empty_df(), {}, _err("Token match select failed", exc)


def apply_token_kv_edit_cb(
    token_table: Any,
    layer: int,
    component_mode: str,
    all_layers: bool,
    head_expr: str,
    dim_expr: str,
    mode: str,
    value: float,
    strength: float,
    temperature: float,
    top_k: int,
    top_p: float,
):
    try:
        positions = _positions_from_table(token_table)
        if not positions:
            raise ValueError("No token rows selected/searched. Search token text first.")
        layer_expr = "all" if bool(all_layers) else str(int(layer))
        record = edit_token_positions(
            KV_STATE.cache,
            positions,
            layer_expr=layer_expr,
            component_mode=component_mode,
            head_expr=head_expr or ":",
            dim_expr=dim_expr or ":",
            mode=mode,
            value=float(value),
            strength=float(strength),
        )
        KV_EDIT_LOG.append(record)
        comp = "key" if component_mode == "both" else component_mode
        pos = positions[0]
        expr = build_token_kv_slice(pos, head_expr or ":", dim_expr or ":")
        inspect_layer = 0 if bool(all_layers) else int(layer)
        stats = inspect_cache_slice(KV_STATE.cache, inspect_layer, comp, expr)
        df = pd.DataFrame(stats["preview"])
        outputs = _kv_outputs(temperature, int(top_k), float(top_p))
        return (*outputs, _json(stats["stats"]), df, _kv_slice_state(inspect_layer, comp, expr, df))
    except Exception as exc:
        return (*_kv_error_outputs(_err("Token KV edit failed", exc)), _json({"error": str(exc)}), _empty_df(), {})


def explorer_analyze_cb(query: str, mode: str, case_sensitive: bool, max_rows: int, layer_expr: str, component_mode: str):
    try:
        tokenized = query_tokenization_rows(_session(), query)
    except Exception as exc:
        tokenized = [{"error": _err("Tokenize failed", exc)}]
    try:
        rows, meta = search_context_tokens(_session(), KV_STATE, query, mode, bool(case_sensitive), int(max_rows))
    except Exception as exc:
        rows, meta = [], {"error": _err("Context token search failed", exc)}
    try:
        positions = _positions_from_table(rows)
        layers = token_layer_info(_session(), KV_STATE, positions[:64], layer_expr or "all", component_mode or "both") if positions else []
    except Exception as exc:
        layers = [{"error": _err("Layer info failed", exc)}]
    return _tokenize_df(tokenized), _token_search_df(rows), _token_layer_df(layers), _json(meta)


def select_explorer_token_cb(evt: gr.SelectData, layer_expr: str, component_mode: str, head_expr: str, dim_expr: str):
    try:
        row = _selected_row_dict(evt, TOKEN_SEARCH_COLUMNS)
        kv_pos = _kv_position_from_row(row)
        expr = build_token_kv_slice(kv_pos, head_expr or ":", dim_expr or ":")
        rows = token_layer_info(_session(), KV_STATE, [kv_pos], layer_expr or "all", component_mode or "both")
        status = f"✅ Token selected: absolute position={row.get('position')}, kv_position={kv_pos}, token_id={row.get('token_id')}, slice={expr}"
        return int(row.get("position") or kv_pos), kv_pos, int(row.get("token_id") or 0), str(row.get("token_text") or ""), expr, _token_layer_df(rows), status
    except Exception as exc:
        return 0, 0, 0, "", ":", _token_layer_df([]), _err("Explorer token select failed", exc)


def inspect_explorer_layers_cb(kv_position: int, layer_expr: str, component_mode: str):
    try:
        rows = token_layer_info(_session(), KV_STATE, [int(kv_position)], layer_expr or "all", component_mode or "both")
        return _token_layer_df(rows)
    except Exception as exc:
        return _token_layer_df([{"error": _err("Inspect token layers failed", exc)}])


def send_explorer_token_to_kv_cb(kv_position: int, layer: int, component: str, head_expr: str, dim_expr: str):
    try:
        comp = component if component in {"key", "value"} else "key"
        expr = build_token_kv_slice(int(kv_position), head_expr or ":", dim_expr or ":")
        data = inspect_cache_slice(KV_STATE.cache, int(layer), comp, expr)
        df = pd.DataFrame(data["preview"])
        return int(layer), comp, expr, _json(data["stats"]), df, _kv_slice_state(int(layer), comp, expr, df)
    except Exception as exc:
        return int(layer or 0), "key", ":", _json({"error": _err("Send token to KV editor failed", exc)}), _empty_df(), {}


def explorer_edit_selected_cb(kv_position: int, layer_expr: str, component_mode: str, head_expr: str, dim_expr: str, mode: str, value: float, strength: float, temperature: float, top_k: int, top_p: float):
    try:
        record = edit_token_positions(
            KV_STATE.cache,
            [int(kv_position)],
            layer_expr=layer_expr or "all",
            component_mode=component_mode or "both",
            head_expr=head_expr or ":",
            dim_expr=dim_expr or ":",
            mode=mode,
            value=float(value),
            strength=float(strength),
        )
        KV_EDIT_LOG.append(record)
        return _kv_outputs(temperature, int(top_k), float(top_p))
    except Exception as exc:
        return _kv_error_outputs(_err("Explorer token edit failed", exc))

def disassemble_cb(query: str, regex: bool, leaf_only: bool, limit: int):
    try:
        return _df(disassemble_modules(_session(), query, regex, leaf_only, int(limit)))
    except Exception as exc:
        return _df([{"error": _err("Disassemble failed", exc)}])


def select_module_row_cb(evt: gr.SelectData):
    try:
        row = evt.row_value or []
        module_name = str(row[1]) if len(row) > 1 else str(evt.value)
        return module_name, module_name
    except Exception:
        return "", ""


def search_weights_cb(query: str, regex: bool, limit: int, include_stats: bool):
    try:
        return _df(search_parameters(_session(), query, regex, int(limit), include_stats))
    except Exception as exc:
        return _df([{"error": _err("Search weights failed", exc)}])


def inspect_weight_cb(name: str, index_expr: str, plot_kind: str):
    try:
        target = (name or "").strip()
        data = inspect_parameter(_session(), target, index_expr or ":")
        if plot_kind == "heatmap":
            fig = weight_heatmap(_session(), target, index_expr or ":")
        elif plot_kind == "histogram":
            fig = weight_histogram(_session(), target, index_expr or ":")
        else:
            fig = empty_figure("No plot selected")
        df = pd.DataFrame(data["preview"])
        return _json(data["stats"]), df, fig, _preview_state(target, index_expr or ":", df)
    except Exception as exc:
        return _json({"error": _err("Inspect weight failed", exc)}), _empty_df(), empty_figure("Inspect failed"), {}


def inspect_weight_btn_cb(name: str, index_expr: str, plot_kind: str):
    return inspect_weight_cb(name, index_expr, plot_kind)


def edit_weight_cb(name: str, index_expr: str, mode: str, value: float, strength: float, plot_kind: str):
    target = (name or "").strip()
    try:
        backup = apply_scalar_edit(_session().get_parameter(target), target, index_expr or ":", mode, float(value), float(strength))
        _session().append_edit(backup)
        stats, preview, fig, state = inspect_weight_cb(target, index_expr or ":", plot_kind)
        return f"✅ Applied {mode} to {target}[{index_expr or ':'}]", stats, preview, fig, _json(tail(_session().edit_log, 100)), state
    except Exception as exc:
        stats, preview, fig, state = inspect_weight_cb(target, index_expr or ":", plot_kind) if target else ("{}", _empty_df(), empty_figure("Edit failed"), {})
        return _err("Apply weight edit failed", exc), stats, preview, fig, _json(tail(_session().edit_log, 100)), state


def edit_weight_preview_cb(new_preview: Any, name: str, index_expr: str, plot_kind: str, preview_state: dict[str, Any]):
    target = (name or "").strip()
    try:
        state = preview_state or {}
        old_preview = state.get("preview")
        if not old_preview:
            stats, preview, fig, new_state = inspect_weight_cb(target, index_expr or ":", plot_kind)
            return "No active preview grid state; inspect the weight first.", stats, preview, fig, _json(tail(_session().edit_log, 100)), new_state
        if state.get("parameter") != target or state.get("index_expr") != (index_expr or ":"):
            stats, preview, fig, new_state = inspect_weight_cb(target, index_expr or ":", plot_kind)
            return "Preview state did not match the current parameter/slice; refreshed instead of editing.", stats, preview, fig, _json(tail(_session().edit_log, 100)), new_state
        backup = apply_preview_table_edits(_session().get_parameter(target), target, index_expr or ":", old_preview, new_preview)
        if backup is not None:
            _session().append_edit(backup)
            status = f"✅ Applied direct preview-grid edit to {target}[{index_expr or ':'}]"
        else:
            status = "No numeric cell change detected."
        stats, preview, fig, new_state = inspect_weight_cb(target, index_expr or ":", plot_kind)
        return status, stats, preview, fig, _json(tail(_session().edit_log, 100)), new_state
    except Exception as exc:
        stats, preview, fig, new_state = inspect_weight_cb(target, index_expr or ":", plot_kind) if target else ("{}", _empty_df(), empty_figure("Grid edit failed"), {})
        return _err("Preview grid edit failed", exc), stats, preview, fig, _json(tail(_session().edit_log, 100)), new_state


def edit_weight_cell_cb(preview_table: Any, row: int, col: int, value: float, name: str, index_expr: str, plot_kind: str, preview_state: dict[str, Any]):
    target = (name or "").strip()
    try:
        state = preview_state or {}
        old_preview = state.get("preview") or dataframe_to_matrix(preview_table)
        if state.get("parameter") and state.get("parameter") != target:
            raise ValueError("Preview state belongs to a different parameter. Inspect the target weight again.")
        backup = apply_preview_cell_edit(_session().get_parameter(target), target, index_expr or ":", old_preview, int(row), int(col), value)
        if backup is not None:
            _session().append_edit(backup)
            status = f"✅ Wrote preview cell [{int(row)}, {int(col)}] to {target}[{index_expr or ':'}]"
        else:
            status = "No numeric cell change detected."
        stats, preview, fig, new_state = inspect_weight_cb(target, index_expr or ":", plot_kind)
        return status, stats, preview, fig, _json(tail(_session().edit_log, 100)), new_state
    except Exception as exc:
        stats, preview, fig, new_state = inspect_weight_cb(target, index_expr or ":", plot_kind) if target else ("{}", _empty_df(), empty_figure("Cell write failed"), {})
        return _err("Weight cell write failed", exc), stats, preview, fig, _json(tail(_session().edit_log, 100)), new_state


def undo_weight_cb(name: str, index_expr: str, plot_kind: str):
    target = (name or "").strip()
    try:
        msg = _session().undo_last_edit()
        stats, preview, fig, state = inspect_weight_cb(target, index_expr or ":", plot_kind) if target else ("{}", _empty_df(), empty_figure("Undo completed"), {})
        return msg, stats, preview, fig, _json(tail(_session().edit_log, 100)), state
    except Exception as exc:
        stats, preview, fig, state = inspect_weight_cb(target, index_expr or ":", plot_kind) if target else ("{}", _empty_df(), empty_figure("Undo failed"), {})
        return _err("Undo failed", exc), stats, preview, fig, _json(tail(_session().edit_log, 100)), state


def set_alias_cb(name: str, alias: str):
    try:
        msg = _session().set_alias(name, alias)
        return f"✅ {msg}", _json(tail(_session().edit_log, 100)), _json(model_parameter_summary(_session()))
    except Exception as exc:
        return _err("Set alias failed", exc), _json(tail(_session().edit_log, 100)), "{}"


def select_weight_row_cb(evt: gr.SelectData):
    try:
        row = evt.row_value or []
        name = str(row[0]) if row else str(evt.value)
        alias = str(row[1]) if len(row) > 1 and row[1] is not None else _session().aliases.get(name, "")
        stats, preview, fig, state = inspect_weight_cb(name, ":", "histogram")
        return name, alias, ":", stats, preview, fig, f"✅ Selected parameter: {name}", state
    except Exception as exc:
        return "", "", ":", _json({"error": _err("Weight row select failed", exc)}), _empty_df(), empty_figure("Select failed"), _err("Weight row select failed", exc), {}


def export_model_cb(export_dir: str, safe_serialization: bool, make_zip: bool):
    try:
        out = _session().export(export_dir.strip() or default_export_dir(_session().model_id), safe_serialization=safe_serialization)
        manifest = Path(out, "nekoai_edit_manifest.json").read_text(encoding="utf-8")
        zip_path = zip_dir(out, out + ".zip") if make_zip else None
        return f"✅ Exported model: {out}", manifest, zip_path
    except Exception as exc:
        return _err("Export model failed", exc), "{}", None


def create_app() -> gr.Blocks:
    css_path = Path(__file__).resolve().parents[1] / "assets" / "theme.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    bg_path = Path(__file__).resolve().parents[1] / "assets" / "bg.png"
    if bg_path.exists():
        bg_data = base64.b64encode(bg_path.read_bytes()).decode("ascii")
        css += f"""
html,body,.gradio-container{{
  min-height:100vh;
}}
html::before{{
  content:"";
  position:fixed;
  inset:0;
  z-index:-1;
  background-image:url("data:image/png;base64,{bg_data}");
  background-size:cover;
  background-position:center center;
  background-repeat:no-repeat;
  opacity:.14;
  pointer-events:none;
}}
"""
    try:
        theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate", radius_size="lg")
    except Exception:
        theme = gr.themes.Base()
    block_kwargs = {"title": "NekoAIEditor v1.0"}
    block_params = inspect.signature(gr.Blocks).parameters
    if "css" in block_params:
        block_kwargs["css"] = css
    if "theme" in block_params:
        block_kwargs["theme"] = theme
    with gr.Blocks(**block_kwargs) as demo:
        demo._neko_launch_css = css
        demo._neko_launch_theme = theme
        gr.HTML(
            """
        <div class="neko-hero"><div class="neko-logo">NekoAIEditor <span>v1.0</span></div>
        <div class="neko-subtitle">AI debugger for LLMs</div></div>
        """
        )

        weight_preview_state = gr.State({})
        kv_slice_preview_state = gr.State({})
        inspector_kv_preview_state = gr.State({})
        top_table_state = gr.State({"rows": []})

        with gr.Tab("1. Model Loader"):
            with gr.Row():
                with gr.Column(scale=2, elem_classes=["neko-card"]):
                    model_id = gr.Textbox(label="Open model id or local path", value="sshleifer/tiny-gpt2")
                    with gr.Row():
                        device_mode = gr.Dropdown(["auto", "cpu", "cuda", "cuda:0", "cuda:1"], value="auto", label="Device")
                        dtype_name = gr.Dropdown(["auto", "float32", "float16", "bfloat16"], value="auto", label="Model dtype")
                    with gr.Row():
                        trust_remote_code = gr.Checkbox(False, label="trust_remote_code")
                        use_device_map = gr.Checkbox(True, label="device_map='auto' on CUDA")
                    revision = gr.Textbox(label="Revision", value="")
                    load_btn = gr.Button("Load model", variant="primary")
                    load_status = gr.Markdown("No model loaded.")
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    compat_btn = gr.Button("Refresh hardware report")
                    compat_report = gr.Textbox(label="Hardware / CUDA report", value=compatibility_report(), lines=14)
                    model_summary = gr.Code(label="Model summary", language="json")

        with gr.Tab("2. KV Cache Debugger"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    gr.Markdown("### Prompt / Prefill")
                    sys_prompt = gr.Textbox(label="System prompt helper", value=DEFAULT_SYSTEM_PROMPT, lines=2)
                    usr_prompt = gr.Textbox(label="User prompt helper", value="Explain KV cache debugging in one paragraph.", lines=3)
                    build_prompt_btn = gr.Button("Build chat prompt")
                    kv_prompt = gr.Textbox(label="Raw debugger prompt", value="Explain KV cache debugging in one paragraph.", lines=8)
                    kv_dtype = gr.Dropdown(["keep", "model", "float32", "float16", "bfloat16"], value="model", label="KV cache dtype/device policy")
                    prefill_btn = gr.Button("Prefill and pause at KV boundary", variant="primary")
                    gr.Markdown("Break at token boundaries, edit/export/import live KV state, then execute the queued token with the edited cache.")
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    gr.Markdown("### Decode Control")
                    temperature = gr.Slider(0, 2, value=0.0, step=0.01, label="Temperature")
                    top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
                    top_p = gr.Slider(0.01, 1.0, value=0.95, step=0.01, label="Top-p")
                    token_override = gr.Textbox(label="Manual token override (token id or one-token text)", value="")
                    logit_bias_json = gr.Code(label="Logit bias JSON", language="json", value="{}")
                    with gr.Row():
                        queue_btn = gr.Button("Queue next token")
                        execute_btn = gr.Button("Execute queued token")
                    auto_btn = gr.Button("Auto step: queue + execute", variant="primary")
                    run_steps = gr.Slider(1, 128, value=8, step=1, label="Run N auto steps")
                    with gr.Row():
                        run_btn = gr.Button("Run N steps")
                        stop_btn = gr.Button("Stop")
                with gr.Column(scale=2, elem_classes=["neko-card"]):
                    kv_status = gr.Markdown("No KV session.")
                    kv_manifest = gr.Code(label="KV runtime manifest", language="json")
                    queued_token = gr.Textbox(label="Queued token", interactive=False)
                    decoded = gr.Textbox(label="Decoded current context", lines=8, interactive=False)

            with gr.Row():
                with gr.Column(scale=2, elem_classes=["neko-card"]):
                    gr.Markdown("### Current logits top tokens")
                    gr.Markdown("Edit `edited_logit`, `logit_delta`, or `target_probability`, then use Apply. Row click also fills the quick editor below.")
                    top_table = gr.Dataframe(
                        label="Current logits top tokens - editable",
                        headers=TOP_TOKEN_COLUMNS,
                        datatype=TOP_TOKEN_TYPES,
                        interactive=True,
                        wrap=True,
                        show_row_numbers=True,
                        show_search="filter",
                        max_height=340,
                    )
                    with gr.Row():
                        apply_top_table_btn = gr.Button("Apply edited logits table", variant="primary")
                        queue_top_btn = gr.Button("Queue selected token")
                    with gr.Row():
                        top_token_id = gr.Number(label="Selected token id", value=0, precision=0)
                        top_token_text = gr.Textbox(label="Selected token text", value="", interactive=False)
                        top_selected_logit = gr.Number(label="Selected current logit", value=0.0)
                    with gr.Row():
                        top_logit_mode = gr.Dropdown(LOGIT_EDIT_MODES, value="add_delta", label="Quick logit edit mode")
                        top_logit_value = gr.Number(label="Value", value=1.0)
                        top_logit_strength = gr.Slider(0, 1, value=1.0, step=0.001, label="Strength")
                        apply_top_logit_btn = gr.Button("Apply quick logit edit")
                    top_edit_status = gr.Markdown("Tip: set `target_probability` between 0 and 1 to solve the token logit needed for that target probability.")
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    kv_table = gr.Dataframe(label="KV cache layer table - click row to inspect", interactive=False, wrap=True, show_search="filter")
                    kv_plot = gr.Plot(label="KV RMS")
                    kv_anomaly_table = gr.Dataframe(label="KV anomaly/watch table", interactive=False, wrap=True)

            with gr.Row():
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    gr.Markdown("### KV Inspect / Modify")
                    kv_layer = gr.Number(label="Layer", value=0, precision=0)
                    kv_component = gr.Dropdown(["key", "value"], value="key", label="Component")
                    kv_index = gr.Textbox(label="Tensor slice", value=":", placeholder="Example: :, 0, -1:, :")
                    inspect_kv_btn = gr.Button("Inspect KV slice")
                    kv_mode = gr.Dropdown(EDIT_MODES, value="add", label="Edit mode")
                    kv_value = gr.Number(label="Value / factor / target RMS / noise scale", value=0.0)
                    kv_strength = gr.Slider(0, 1, value=0.1, step=0.001, label="Edit strength")
                    edit_kv_btn = gr.Button("Apply KV edit", variant="primary")
                    with gr.Accordion("Head vector helper", open=False):
                        kv_head = gr.Number(label="Head", value=0, precision=0)
                        kv_token_pos = gr.Number(label="Token position (-1 = latest)", value=-1, precision=0)
                        kv_dim_start = gr.Number(label="Dim start", value=0, precision=0)
                        kv_dim_count = gr.Number(label="Dim count", value=128, precision=0)
                        inspect_head_btn = gr.Button("Inspect head vector")
                    with gr.Accordion("Token text -> KV locator/editor", open=True):
                        gr.Markdown("Search any token text or phrase in the current context. This is not limited to top logits; it maps text -> token position -> KV cache position.")
                        kv_text_query = gr.Textbox(label="Token text / phrase / token id", value="", placeholder="Examples: cat,  cat, 318, or a phrase from the prompt")
                        with gr.Row():
                            kv_text_mode = gr.Dropdown(KV_TEXT_SEARCH_MODES, value="auto", label="Search mode")
                            kv_text_max_matches = gr.Slider(1, 1000, value=200, step=1, label="Max matches")
                        with gr.Row():
                            kv_text_search_btn = gr.Button("Search token text")
                            kv_text_analyze_btn = gr.Button("Analyze matched tokens by layer")
                        gr.Markdown("Bulk-edit target. Layers support `all`, `0`, `0,2,4`, `0-5`; heads support `all`, `0`, `0,1,2`; dim count `0` means all remaining dims.")
                        with gr.Row():
                            kv_text_layers = gr.Textbox(label="Target layers", value="all")
                            kv_text_components = gr.Dropdown(KV_TEXT_COMPONENTS, value="both", label="Target component")
                        with gr.Row():
                            kv_text_heads = gr.Textbox(label="Target heads", value="all")
                            kv_text_dim_start = gr.Number(label="Dim start", value=0, precision=0)
                            kv_text_dim_count = gr.Number(label="Dim count", value=0, precision=0)
                        kv_text_edit_btn = gr.Button("Apply current KV edit mode/value to matched token KV", variant="primary")
                        kv_text_status = gr.Markdown("Click a match row to bind the exact KV slice into the main editor.")
                with gr.Column(scale=2, elem_classes=["neko-card"]):
                    kv_text_tokenize_table = gr.Dataframe(
                        label="Query tokenization",
                        headers=KV_TEXT_TOKENIZE_COLUMNS,
                        interactive=False,
                        wrap=True,
                        show_row_numbers=True,
                        max_height=150,
                    )
                    kv_text_match_table = gr.Dataframe(
                        label="Token text matches - click row to set KV slice",
                        headers=KV_TEXT_MATCH_COLUMNS,
                        interactive=False,
                        wrap=True,
                        show_search="filter",
                        show_row_numbers=True,
                        max_height=260,
                    )
                    kv_text_layer_table = gr.Dataframe(
                        label="Per-layer KV details - click row to inspect exact slice",
                        headers=KV_TEXT_LAYER_COLUMNS,
                        interactive=False,
                        wrap=True,
                        show_search="filter",
                        show_row_numbers=True,
                        max_height=260,
                    )
                    kv_slice_stats = gr.Code(label="KV slice stats", language="json")
                    kv_slice_preview = gr.Dataframe(label="KV slice preview - editable grid", interactive=True, datatype="number", show_row_numbers=True, max_height=360)
                    with gr.Row():
                        kv_cell_row = gr.Number(label="Cell row", value=0, precision=0)
                        kv_cell_col = gr.Number(label="Cell col", value=0, precision=0)
                        kv_cell_value = gr.Number(label="Cell value", value=0.0)
                        write_kv_cell_btn = gr.Button("Write selected KV cell", variant="primary")
                    gr.Markdown("If grid double-click edit does not work in your browser, click a cell, edit the value box, then press Write selected KV cell.")
                    kv_edit_log = gr.Code(label="KV edit log", language="json")
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    gr.Markdown("### KV State Export / Import")
                    kv_export_root = gr.Textbox(label="Export root", value="exports/kv_cache")
                    kv_export_dtype = gr.Dropdown(["keep", "float32", "float16", "bfloat16"], value="keep", label="Export dtype")
                    export_kv_btn = gr.Button("Export KV state", variant="primary")
                    kv_export_status = gr.Markdown()
                    kv_export_file = gr.File(label="Download KV zip")
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    kv_import_file = gr.File(label="Import KV zip or cache.pt")
                    kv_import_dtype = gr.Dropdown(["keep", "model", "float32", "float16", "bfloat16"], value="model", label="Import dtype")
                    import_kv_btn = gr.Button("Import KV state", variant="primary")

        with gr.Tab("3. Token / KV Inspector"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    gr.Markdown("### Text -> Token position -> KV layer inspector")
                    ti_query = gr.Textbox(label="Text / phrase / token id", value="", placeholder="Examples: hello,  hello, 318, or a phrase from the current prompt")
                    with gr.Row():
                        ti_mode = gr.Dropdown(KV_TEXT_SEARCH_MODES, value="auto", label="Search mode")
                        ti_max_matches = gr.Slider(1, 2000, value=300, step=1, label="Max matches")
                    gr.Markdown("Layer/detail controls. Click a match row to bind it to the main KV editor; click a layer row to inspect that exact layer/component/slice.")
                    with gr.Row():
                        ti_layers = gr.Textbox(label="Layers", value="all")
                        ti_components = gr.Dropdown(KV_TEXT_COMPONENTS, value="both", label="Components")
                    with gr.Row():
                        ti_heads = gr.Textbox(label="Heads", value="all")
                        ti_dim_start = gr.Number(label="Dim start", value=0, precision=0)
                        ti_dim_count = gr.Number(label="Dim count", value=0, precision=0)
                    with gr.Row():
                        ti_search_btn = gr.Button("Search token positions", variant="primary")
                        ti_analyze_btn = gr.Button("Search + analyze KV layers")
                    ti_status = gr.Markdown("This page searches the full active token context, not just the current logits top list.")
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    ti_tokenize_table = gr.Dataframe(
                        label="Query tokenization",
                        headers=KV_TEXT_TOKENIZE_COLUMNS,
                        interactive=False,
                        wrap=True,
                        show_row_numbers=True,
                        max_height=300,
                    )
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    gr.Markdown("### How to use")
                    gr.Markdown("1. Prefill in KV Cache Debugger.\n2. Search a word/phrase/token id here.\n3. Click a token match row to open its KV slice in the main editor.\n4. Click a layer-detail row to inspect that exact layer/component.\n5. Use the KV editor or bulk token-text edit to modify it.")

            with gr.Row():
                with gr.Column(scale=2, elem_classes=["neko-card"]):
                    ti_match_table = gr.Dataframe(
                        label="Matched token positions",
                        headers=KV_TEXT_MATCH_COLUMNS,
                        interactive=False,
                        wrap=True,
                        show_search="filter",
                        show_row_numbers=True,
                        max_height=430,
                    )
                with gr.Column(scale=2, elem_classes=["neko-card"]):
                    ti_layer_table = gr.Dataframe(
                        label="Layer/component KV details for matched tokens",
                        headers=KV_TEXT_LAYER_COLUMNS,
                        interactive=False,
                        wrap=True,
                        show_search="filter",
                        show_row_numbers=True,
                        max_height=430,
                    )

        with gr.Tab("4. Map & Weight Editor"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    dis_query = gr.Textbox(label="Module search", value="", placeholder="lm_head, attn, mlp, q_proj")
                    dis_regex = gr.Checkbox(False, label="Regex")
                    dis_leaf = gr.Checkbox(False, label="Leaf only")
                    dis_limit = gr.Slider(1, 5000, value=1000, step=1, label="Limit")
                    dis_btn = gr.Button("Disassemble model modules", variant="primary")
                with gr.Column(scale=2, elem_classes=["neko-card"]):
                    dis_table = gr.Dataframe(label="module map - click row to seed searches", interactive=False, wrap=True, show_search="filter")
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    weight_query = gr.Textbox(label="Search parameters", value="")
                    weight_regex = gr.Checkbox(False, label="Regex")
                    weight_stats_flag = gr.Checkbox(False, label="Include sampled stats")
                    weight_limit = gr.Slider(1, 1000, value=300, step=1, label="Limit")
                    weight_search_btn = gr.Button("Search weights", variant="primary")
                    gr.Markdown("Click a parameter row to bind the exact name. Use the explicit cell writer if browser table editing is unreliable.")
                with gr.Column(scale=2, elem_classes=["neko-card"]):
                    weight_table = gr.Dataframe(label="Parameters - click row to bind exact name", interactive=False, wrap=True, show_search="filter")
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    weight_name = gr.Textbox(label="Exact parameter name")
                    weight_alias = gr.Textbox(label="UI/export alias name (safe manifest rename)")
                    set_alias_btn = gr.Button("Set/Clear alias")
                    weight_index = gr.Textbox(label="Slice", value=":")
                    plot_kind = gr.Dropdown(["histogram", "heatmap", "none"], value="histogram", label="Plot")
                    inspect_weight_btn = gr.Button("Inspect weight")
                    weight_mode = gr.Dropdown(EDIT_MODES, value="add", label="Edit mode")
                    weight_value = gr.Number(label="Value / factor / target RMS / noise scale", value=0.0)
                    weight_strength = gr.Slider(0, 1, value=0.1, step=0.001, label="Edit strength")
                    edit_weight_btn = gr.Button("Apply weight edit", variant="primary")
                    undo_weight_btn = gr.Button("Undo last weight edit")
                    weight_edit_status = gr.Markdown()
                with gr.Column(scale=2, elem_classes=["neko-card"]):
                    weight_slice_stats = gr.Code(label="Weight slice stats", language="json")
                    weight_preview = gr.Dataframe(label="Weight preview - editable grid", interactive=True, datatype="number", show_row_numbers=True, max_height=360)
                    with gr.Row():
                        weight_cell_row = gr.Number(label="Cell row", value=0, precision=0)
                        weight_cell_col = gr.Number(label="Cell col", value=0, precision=0)
                        weight_cell_value = gr.Number(label="Cell value", value=0.0)
                        write_weight_cell_btn = gr.Button("Write selected weight cell", variant="primary")
                    gr.Markdown("If grid double-click edit does not work, click a visible preview cell, edit Cell value, then press Write selected weight cell.")
                    weight_plot = gr.Plot(label="Weight plot")
                    weight_edit_log = gr.Code(label="Weight edit log", language="json")

        with gr.Tab("5. Export Edited Model"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["neko-card"]):
                    export_dir = gr.Textbox(label="Export directory", value="exports/NekoAIEditor-export")
                    safe_serialization = gr.Checkbox(True, label="Use safetensors")
                    make_zip = gr.Checkbox(True, label="Create zip")
                    export_model_btn = gr.Button("Export edited model", variant="primary")
                    export_model_status = gr.Markdown()
                    export_model_file = gr.File(label="Download model zip")
                with gr.Column(scale=2, elem_classes=["neko-card"]):
                    export_manifest = gr.Code(label="Export manifest", language="json")

        with gr.Tab("6.Notes"):
            gr.Markdown(
                """
            ## NekoAIDbg Project v1.0

            NekoAIEditor v1.0 uses KV Cache breakpoints as the main runtime debugger. It pauses at token boundaries, exposes the live `past_key_values` state, lets you export/analyze/modify/import the state, then continues generation from the modified cache.

            Author: @NeuroSaki987
            GitHub: https://github.com/NeuroSaki987

            **Disclaimer:** The developer assumes no responsibility for any losses or damages arising from the use of this software.
            """
            )

        kv_outputs = [kv_status, kv_manifest, kv_table, kv_plot, top_table, decoded, queued_token, kv_edit_log, kv_anomaly_table, top_table_state]
        kv_edit_outputs = [kv_status, kv_manifest, kv_table, kv_plot, top_table, decoded, queued_token, kv_edit_log, kv_anomaly_table, top_table_state, kv_slice_stats, kv_slice_preview, kv_slice_preview_state]
        kv_text_edit_outputs = [*kv_outputs, kv_text_status, kv_text_layer_table]
        weight_inspect_outputs = [weight_slice_stats, weight_preview, weight_plot, weight_preview_state]
        weight_edit_outputs = [weight_edit_status, weight_slice_stats, weight_preview, weight_plot, weight_edit_log, weight_preview_state]

        load_btn.click(load_model_cb, [model_id, device_mode, dtype_name, trust_remote_code, use_device_map, revision], [load_status, model_summary, compat_report, export_dir])
        compat_btn.click(refresh_compat_cb, outputs=[compat_report])
        build_prompt_btn.click(build_chat_prompt_cb, [sys_prompt, usr_prompt], [kv_prompt])
        prefill_btn.click(prefill_kv_cb, [kv_prompt, kv_dtype, temperature, top_k, top_p], kv_outputs)
        queue_btn.click(queue_token_cb, [temperature, top_k, top_p, token_override, logit_bias_json], kv_outputs)
        execute_btn.click(execute_token_cb, [kv_dtype, temperature, top_k, top_p], kv_outputs)
        auto_btn.click(auto_step_cb, [temperature, top_k, top_p, kv_dtype, token_override, logit_bias_json], kv_outputs)
        run_event = run_btn.click(run_steps_cb, [run_steps, temperature, top_k, top_p, kv_dtype, logit_bias_json], kv_outputs)
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

        top_table.select(select_top_token_row_cb, outputs=[top_token_id, top_token_text, top_selected_logit, top_edit_status])
        top_table.input(apply_top_table_edits_cb, [top_table, top_table_state, temperature, top_k, top_p], kv_outputs)
        apply_top_table_btn.click(apply_top_table_edits_cb, [top_table, top_table_state, temperature, top_k, top_p], kv_outputs)
        apply_top_logit_btn.click(apply_top_logit_cb, [top_token_id, top_logit_mode, top_logit_value, top_logit_strength, temperature, top_k, top_p], kv_outputs)
        queue_top_btn.click(queue_selected_top_token_cb, [top_token_id, temperature, top_k, top_p], kv_outputs)

        inspect_kv_btn.click(inspect_kv_cb, [kv_layer, kv_component, kv_index], [kv_slice_stats, kv_slice_preview, kv_slice_preview_state])
        inspect_head_btn.click(inspect_kv_head_cb, [kv_layer, kv_component, kv_head, kv_token_pos, kv_dim_start, kv_dim_count], [kv_index, kv_slice_stats, kv_slice_preview, kv_slice_preview_state])
        edit_kv_btn.click(edit_kv_cb, [kv_layer, kv_component, kv_index, kv_mode, kv_value, kv_strength, temperature, top_k, top_p], kv_edit_outputs)
        kv_slice_preview.input(edit_kv_preview_cb, [kv_slice_preview, kv_layer, kv_component, kv_index, kv_slice_preview_state, temperature, top_k, top_p], kv_edit_outputs)
        kv_slice_preview.select(select_kv_preview_cell_cb, outputs=[kv_cell_row, kv_cell_col, kv_cell_value])
        write_kv_cell_btn.click(edit_kv_cell_cb, [kv_slice_preview, kv_cell_row, kv_cell_col, kv_cell_value, kv_layer, kv_component, kv_index, kv_slice_preview_state, temperature, top_k, top_p], kv_edit_outputs)
        kv_table.select(select_kv_row_cb, outputs=[kv_layer, kv_component, kv_index, kv_slice_stats, kv_slice_preview, kv_slice_preview_state])
        kv_text_search_btn.click(search_kv_text_tokens_cb, [kv_text_query, kv_text_mode, kv_text_max_matches], [kv_text_status, kv_text_tokenize_table, kv_text_match_table])
        kv_text_analyze_btn.click(analyze_kv_text_tokens_cb, [kv_text_query, kv_text_mode, kv_text_layers, kv_text_components, kv_text_heads, kv_text_dim_start, kv_text_dim_count, kv_text_max_matches], [kv_text_status, kv_text_tokenize_table, kv_text_match_table, kv_text_layer_table])
        kv_text_match_table.select(select_kv_text_match_cb, [kv_layer, kv_component, kv_text_heads, kv_text_dim_start, kv_text_dim_count], [kv_layer, kv_component, kv_index, kv_token_pos, kv_slice_stats, kv_slice_preview, kv_slice_preview_state, kv_text_status])
        kv_text_layer_table.select(select_kv_text_layer_cb, outputs=[kv_layer, kv_component, kv_index, kv_slice_stats, kv_slice_preview, kv_slice_preview_state, kv_text_status])
        kv_text_edit_btn.click(apply_kv_text_edit_cb, [kv_text_query, kv_text_mode, kv_text_layers, kv_text_components, kv_text_heads, kv_text_dim_start, kv_text_dim_count, kv_text_max_matches, kv_mode, kv_value, kv_strength, temperature, top_k, top_p], kv_text_edit_outputs)
        export_kv_btn.click(export_kv_cb, [kv_export_root, kv_export_dtype], [kv_export_status, kv_export_file])
        import_kv_btn.click(import_kv_cb, [kv_import_file, kv_import_dtype, temperature, top_k, top_p], kv_outputs)

        ti_search_btn.click(search_kv_text_tokens_cb, [ti_query, ti_mode, ti_max_matches], [ti_status, ti_tokenize_table, ti_match_table])
        ti_analyze_btn.click(analyze_kv_text_tokens_cb, [ti_query, ti_mode, ti_layers, ti_components, ti_heads, ti_dim_start, ti_dim_count, ti_max_matches], [ti_status, ti_tokenize_table, ti_match_table, ti_layer_table])
        ti_match_table.select(select_kv_text_match_cb, [kv_layer, kv_component, ti_heads, ti_dim_start, ti_dim_count], [kv_layer, kv_component, kv_index, kv_token_pos, kv_slice_stats, kv_slice_preview, kv_slice_preview_state, ti_status])
        ti_layer_table.select(select_kv_text_layer_cb, outputs=[kv_layer, kv_component, kv_index, kv_slice_stats, kv_slice_preview, kv_slice_preview_state, ti_status])

        dis_btn.click(disassemble_cb, [dis_query, dis_regex, dis_leaf, dis_limit], [dis_table])
        dis_table.select(select_module_row_cb, outputs=[dis_query, weight_query])
        weight_search_btn.click(search_weights_cb, [weight_query, weight_regex, weight_limit, weight_stats_flag], [weight_table])
        weight_table.select(select_weight_row_cb, outputs=[weight_name, weight_alias, weight_index, weight_slice_stats, weight_preview, weight_plot, weight_edit_status, weight_preview_state])
        inspect_weight_btn.click(inspect_weight_btn_cb, [weight_name, weight_index, plot_kind], weight_inspect_outputs)
        edit_weight_btn.click(edit_weight_cb, [weight_name, weight_index, weight_mode, weight_value, weight_strength, plot_kind], weight_edit_outputs)
        weight_preview.input(edit_weight_preview_cb, [weight_preview, weight_name, weight_index, plot_kind, weight_preview_state], weight_edit_outputs)
        weight_preview.select(select_weight_preview_cell_cb, outputs=[weight_cell_row, weight_cell_col, weight_cell_value])
        write_weight_cell_btn.click(edit_weight_cell_cb, [weight_preview, weight_cell_row, weight_cell_col, weight_cell_value, weight_name, weight_index, plot_kind, weight_preview_state], weight_edit_outputs)
        undo_weight_btn.click(undo_weight_cb, [weight_name, weight_index, plot_kind], weight_edit_outputs)
        set_alias_btn.click(set_alias_cb, [weight_name, weight_alias], [weight_edit_status, weight_edit_log, model_summary])
        export_model_btn.click(export_model_cb, [export_dir, safe_serialization, make_zip], [export_model_status, export_manifest, export_model_file])

    return demo
