from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

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
from .utils import EditRecord, now_ts


@dataclass
class SliceBackup:
    parameter: str
    index_expr: str
    before_cpu: Any
    record: EditRecord


def _view_for_index(param: torch.nn.Parameter, index_expr: str) -> torch.Tensor:
    idx = parse_index_expr(index_expr)
    return param.data[idx] if idx else param.data


def _clone_backup(view: torch.Tensor, max_backup_elements: int) -> Any:
    return None if int(view.numel()) > int(max_backup_elements) else view.detach().cpu().clone()


def apply_scalar_edit(
    param: torch.nn.Parameter,
    parameter_name: str,
    index_expr: str,
    mode: str,
    value: float,
    strength: float,
    max_backup_elements: int = 10_000_000,
) -> SliceBackup:
    with torch.no_grad():
        view = _view_for_index(param, index_expr)
        before = tensor_stats(view)
        before_cpu = _clone_backup(view, max_backup_elements)
        apply_tensor_edit_(view, mode, value, strength)
        after = tensor_stats(view)
    record = EditRecord(now_ts(), parameter_name, index_expr or ":", mode, float(value), float(strength), before, after)
    return SliceBackup(parameter_name, index_expr or ":", before_cpu, record)


def apply_preview_table_edits(
    param: torch.nn.Parameter,
    parameter_name: str,
    index_expr: str,
    old_preview: Any,
    new_preview: Any,
    max_changes: int = 256,
    max_backup_elements: int = 10_000_000,
) -> SliceBackup | None:
    """Apply direct cell edits made in the preview DataFrame to the real tensor slice.

    The preview table is a bounded 2D projection of the tensor slice. For tensors
    with more than two dimensions, columns map to the flattened tail dimensions,
    matching tensor_preview().
    """
    old = dataframe_to_matrix(old_preview)
    new = dataframe_to_matrix(new_preview)
    changes = diff_matrices(old, new, max_changes=max_changes)
    if not changes:
        return None
    with torch.no_grad():
        view = _view_for_index(param, index_expr)
        before = tensor_stats(view)
        before_cpu = _clone_backup(view, max_backup_elements)
        applied: list[dict[str, Any]] = []
        for row, col, old_val, new_val in changes:
            local_idx = preview_cell_to_local_index(view.shape, row, col)
            try:
                numeric = float(new_val)
            except Exception as exc:
                raise ValueError(f"Preview cell [{row}, {col}] must be numeric, got {new_val!r}.") from exc
            target = view[local_idx]
            if not torch.is_floating_point(target):
                raise TypeError("Only floating point parameters can be edited from the preview grid.")
            old_tensor_value = float(target.detach().float().cpu().item()) if target.numel() == 1 else None
            target.copy_(torch.as_tensor(numeric, device=target.device, dtype=target.dtype))
            applied.append({"row": row, "col": col, "local_index": local_idx, "old_preview": old_val, "old_tensor": old_tensor_value, "new": numeric})
        after = tensor_stats(view)
    note = f"preview_grid_cells={applied[:32]}" + (f" ... total={len(applied)}" if len(applied) > 32 else "")
    record = EditRecord(now_ts(), parameter_name, index_expr or ":", "preview_grid_set", 0.0, 1.0, before, after, note=note)
    return SliceBackup(parameter_name, index_expr or ":", before_cpu, record)



def apply_preview_cell_edit(
    param: torch.nn.Parameter,
    parameter_name: str,
    index_expr: str,
    old_preview: Any,
    row: int,
    col: int,
    value: Any,
    max_changes: int = 256,
    max_backup_elements: int = 10_000_000,
) -> SliceBackup | None:
    """Write one visible preview cell into the underlying tensor slice."""
    old = dataframe_to_matrix(old_preview)
    new = matrix_with_cell(old, int(row), int(col), value)
    return apply_preview_table_edits(
        param,
        parameter_name,
        index_expr,
        old,
        new,
        max_changes=max_changes,
        max_backup_elements=max_backup_elements,
    )

def restore_backup(param: torch.nn.Parameter, backup: SliceBackup) -> None:
    if backup.before_cpu is None:
        raise ValueError("The edited slice was too large for backup. Re-load the model or restore from an exported checkpoint.")
    idx = parse_index_expr(backup.index_expr)
    with torch.no_grad():
        view = param.data[idx] if idx else param.data
        view.copy_(backup.before_cpu.to(device=view.device, dtype=view.dtype))


def inspect_tensor_slice(t: torch.Tensor, index_expr: str) -> dict[str, Any]:
    idx = parse_index_expr(index_expr)
    with torch.no_grad():
        view = t.data[idx] if idx else t.data
        return {"stats": tensor_stats(view), "preview": tensor_preview(view)}
