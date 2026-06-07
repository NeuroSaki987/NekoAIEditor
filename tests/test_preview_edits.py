import torch

from nekoai.tensor_ops import dataframe_to_matrix, preview_cell_to_local_index, tensor_preview
from nekoai.weight_ops import apply_preview_table_edits, restore_backup


def test_preview_cell_mapping_for_flattened_tail():
    assert preview_cell_to_local_index((2, 3, 4), 1, 5) == (1, 1, 1)


def test_tensor_preview_slices_before_cpu_shape():
    x = torch.arange(10000, dtype=torch.float32).reshape(100, 100)
    preview = tensor_preview(x, max_rows=3, max_cols=4)
    assert len(preview) == 3
    assert len(preview[0]) == 4
    assert preview[2][3] == 203.0


def test_weight_preview_grid_edit_and_restore():
    p = torch.nn.Parameter(torch.zeros(2, 3))
    old = tensor_preview(p.data)
    new = [row[:] for row in old]
    new[1][2] = 7.0
    backup = apply_preview_table_edits(p, "w", ":", old, new)
    assert backup is not None
    assert p.data[1, 2].item() == 7.0
    restore_backup(p, backup)
    assert torch.allclose(p.data, torch.zeros(2, 3))
