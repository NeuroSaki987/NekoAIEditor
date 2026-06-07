from __future__ import annotations

import math
from typing import Any, Iterable

import torch


def _parse_int(text: str) -> int | None:
    text = text.strip()
    return None if text == "" else int(text)


def _parse_atom(atom: str) -> slice | int | type(Ellipsis):
    atom = atom.strip()
    if atom in {"...", "Ellipsis"}:
        return Ellipsis
    if ":" in atom:
        parts = atom.split(":")
        if len(parts) > 3:
            raise ValueError(f"Bad slice atom: {atom}")
        start = _parse_int(parts[0]) if len(parts) > 0 else None
        stop = _parse_int(parts[1]) if len(parts) > 1 else None
        step = _parse_int(parts[2]) if len(parts) > 2 else None
        return slice(start, stop, step)
    return int(atom)


def parse_index_expr(expr: str | None) -> tuple[Any, ...]:
    """Parse safe tensor slice text such as ':, 0, -1:, :'.

    This intentionally supports only integers, slices, and ellipsis. It does not
    eval Python code, so user input cannot execute arbitrary code.
    """
    text = (expr or "").strip()
    if text in {"", ":", "...", "Ellipsis"}:
        return tuple()
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        text = text[1:-1].strip()
    if text in {"", ":", "...", "Ellipsis"}:
        return tuple()
    return tuple(_parse_atom(a) for a in text.split(",") if a.strip())


def tensor_stats(t: torch.Tensor, sample: int = 200_000) -> dict[str, Any]:
    with torch.no_grad():
        x = t.detach()
        info: dict[str, Any] = {
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "device": str(x.device),
            "numel": int(x.numel()),
        }
        if x.numel() == 0:
            return info
        flat = x.reshape(-1)
        if flat.numel() > sample:
            step = max(1, int(flat.numel()) // int(sample))
            flat = flat[::step][:sample]
        if not torch.is_floating_point(flat):
            try:
                v_int = flat.cpu()
                info.update(sampled=int(v_int.numel()), min=int(v_int.min().item()), max=int(v_int.max().item()))
            except Exception:
                pass
            return info
        v = flat.float().cpu()
        finite = torch.isfinite(v)
        info.update(
            sampled=int(v.numel()),
            finite=int(finite.sum().item()),
            nan=int(torch.isnan(v).sum().item()),
            posinf=int(torch.isposinf(v).sum().item()),
            neginf=int(torch.isneginf(v).sum().item()),
        )
        if finite.any():
            vf = v[finite]
            info.update(
                mean=float(vf.mean().item()),
                std=float(vf.std(unbiased=False).item()) if vf.numel() > 1 else 0.0,
                min=float(vf.min().item()),
                max=float(vf.max().item()),
                rms=float(torch.sqrt(torch.mean(vf * vf)).item()),
                l2=float(torch.linalg.vector_norm(vf).item()),
            )
        return info


def tensor_preview(t: torch.Tensor, max_rows: int = 24, max_cols: int = 24) -> list[list[float]]:
    # Slice on the source device before copying to CPU. This keeps previews cheap
    # even for multi-GB matrices and avoids UI freezes.
    with torch.no_grad():
        x = t.detach()
        if x.ndim == 0:
            return [[float(x.float().cpu().item())]]
        if x.ndim == 1:
            small = x[: max_rows * max_cols].float().cpu()
            return [[float(v) for v in small.tolist()]]
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        small = x[:max_rows, :max_cols].float().cpu()
        return [[float(v) for v in row] for row in small.tolist()]


def _finite_view(view: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(view):
        raise TypeError("Only floating tensors can be edited with numeric transforms.")
    return view


def _safe_rms(view: torch.Tensor) -> torch.Tensor:
    x = view.float()
    finite = torch.isfinite(x)
    if not finite.any():
        return torch.tensor(0.0, device=view.device)
    return torch.sqrt(torch.mean(x[finite] * x[finite])).to(device=view.device)


def apply_tensor_edit_(view: torch.Tensor, mode: str, value: float, strength: float) -> None:
    """Apply in-place scalar edit with bounded strength and numerical safeguards."""
    if not torch.is_floating_point(view):
        raise TypeError("Only floating tensors can be edited.")
    s = max(0.0, min(1.0, float(strength)))
    v = float(value)
    mode = (mode or "add").lower().strip()
    eps = 1e-12
    with torch.no_grad():
        if mode in {"set", "lerp_to"}:
            view.mul_(1.0 - s).add_(v * s)
        elif mode == "add":
            view.add_(v * s)
        elif mode == "subtract":
            view.add_(-v * s)
        elif mode == "multiply":
            view.mul_(1.0 + (v - 1.0) * s)
        elif mode == "zero":
            view.mul_(1.0 - s)
        elif mode in {"clip_abs", "clamp_abs"}:
            clipped = view.clamp(-abs(v), abs(v))
            view.mul_(1.0 - s).add_(clipped * s)
        elif mode == "soft_clip":
            limit = max(abs(v), eps)
            clipped = torch.tanh(view / limit) * limit
            view.mul_(1.0 - s).add_(clipped * s)
        elif mode == "noise":
            view.add_(torch.randn_like(view) * v * s)
        elif mode == "center":
            x = view.float()
            finite = torch.isfinite(x)
            if finite.any():
                mean = x[finite].mean().to(dtype=view.dtype, device=view.device)
                view.add_(-mean * s)
        elif mode in {"normalize_rms", "scale_to_rms"}:
            target = abs(v)
            current = _safe_rms(view).clamp_min(eps)
            scaled = view * (target / current).to(dtype=view.dtype)
            view.mul_(1.0 - s).add_(scaled * s)
        elif mode == "standardize":
            target_std = abs(v) if abs(v) > eps else 1.0
            x = view.float()
            finite = torch.isfinite(x)
            if finite.any():
                mean = x[finite].mean().to(device=view.device)
                std = x[finite].std(unbiased=False).clamp_min(eps).to(device=view.device)
                standardized = ((view.float() - mean) / std * target_std).to(dtype=view.dtype)
                view.mul_(1.0 - s).add_(standardized * s)
        elif mode == "nan_to_num":
            repaired = torch.nan_to_num(view, nan=v, posinf=abs(v), neginf=-abs(v))
            view.mul_(1.0 - s).add_(repaired * s)
        else:
            raise ValueError(
                "mode must be one of set/add/subtract/multiply/lerp_to/zero/clip_abs/soft_clip/noise/center/normalize_rms/standardize/nan_to_num"
            )


def preview_cell_to_local_index(shape: Iterable[int] | torch.Size, row: int, col: int) -> tuple[int, ...]:
    """Map a displayed preview cell back into the inspected tensor view."""
    dims = tuple(int(x) for x in shape)
    r = int(row)
    c = int(col)
    if not dims:
        if r == 0 and c == 0:
            return tuple()
        raise IndexError("Scalar preview only has cell [0, 0].")
    if len(dims) == 1:
        if r != 0:
            raise IndexError("1D tensor preview uses row 0 only.")
        if c < 0 or c >= dims[0]:
            raise IndexError(f"Column {c} is outside 1D tensor length {dims[0]}.")
        return (c,)
    if r < 0 or r >= dims[0]:
        raise IndexError(f"Row {r} is outside first dimension {dims[0]}.")
    tail = dims[1:]
    width = math.prod(tail)
    if c < 0 or c >= width:
        raise IndexError(f"Column {c} is outside flattened tail width {width}.")
    coords: list[int] = []
    rem = c
    for dim in reversed(tail):
        coords.append(rem % dim)
        rem //= dim
    coords.reverse()
    return (r, *coords)


def dataframe_to_matrix(data: Any) -> list[list[Any]]:
    """Convert Gradio Dataframe payloads/pandas frames/lists into a plain matrix."""
    if data is None:
        return []
    if hasattr(data, "values"):
        return data.values.tolist()
    if isinstance(data, dict):
        if "data" in data:
            return [list(r) for r in data.get("data") or []]
        if "value" in data:
            return dataframe_to_matrix(data["value"])
    if isinstance(data, list):
        if not data:
            return []
        if all(not isinstance(x, (list, tuple)) for x in data):
            return [list(data)]
        return [list(r) for r in data]
    return []



def matrix_with_cell(data: Any, row: int, col: int, value: Any) -> list[list[Any]]:
    """Return a copy of a preview matrix with one visible cell replaced."""
    matrix = dataframe_to_matrix(data)
    if not matrix:
        raise ValueError("No active preview table. Inspect a slice first.")
    r = int(row)
    c = int(col)
    if r < 0 or r >= len(matrix):
        raise IndexError(f"Preview row {r} is outside 0..{len(matrix) - 1}.")
    if c < 0 or c >= len(matrix[r]):
        raise IndexError(f"Preview column {c} is outside 0..{len(matrix[r]) - 1}.")
    new_matrix = [list(x) for x in matrix]
    new_matrix[r][c] = value
    return new_matrix

def diff_matrices(old: list[list[Any]], new: list[list[Any]], max_changes: int = 256) -> list[tuple[int, int, Any, Any]]:
    changes: list[tuple[int, int, Any, Any]] = []
    rows = min(len(old), len(new))
    for r in range(rows):
        cols = min(len(old[r]), len(new[r]))
        for c in range(cols):
            a = old[r][c]
            b = new[r][c]
            if _cell_changed(a, b):
                changes.append((r, c, a, b))
                if len(changes) >= int(max_changes):
                    return changes
    return changes


def _as_float_or_none(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        return v
    except Exception:
        return None


def _cell_changed(a: Any, b: Any) -> bool:
    af = _as_float_or_none(a)
    bf = _as_float_or_none(b)
    if af is not None and bf is not None:
        if math.isnan(af) and math.isnan(bf):
            return False
        return abs(af - bf) > 1e-12
    return str(a) != str(b)
