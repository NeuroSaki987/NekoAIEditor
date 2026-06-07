import math

import torch

from nekoai.kv_cache_debugger import KVRuntimeState, edit_cache_preview_cell, edit_logits_from_table, edit_logits_value
from nekoai.tensor_ops import matrix_with_cell, tensor_preview
from nekoai.weight_ops import apply_preview_cell_edit


def test_matrix_with_cell_replaces_copy():
    original = [[1.0, 2.0], [3.0, 4.0]]
    new = matrix_with_cell(original, 1, 0, 9.0)
    assert original[1][0] == 3.0
    assert new[1][0] == 9.0


def test_weight_preview_single_cell_edit():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    preview = tensor_preview(p.data)
    backup = apply_preview_cell_edit(p, "w", ":", preview, 0, 1, 5.0)
    assert backup is not None
    assert p.data[0, 1].item() == 5.0


def test_kv_preview_single_cell_edit():
    cache = ((torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)),)
    preview = tensor_preview(cache[0][0])
    rec = edit_cache_preview_cell(cache, 0, "key", ":", preview, 0, 1, 3.0)
    assert rec is not None
    assert cache[0][0].reshape(1, -1)[0, 1].item() == 3.0


def test_edit_logits_value_and_table():
    state = KVRuntimeState(logits=torch.zeros(1, 5), supported=True)
    rec = edit_logits_value(state, 2, "add_delta", 4.0, 0.5)
    assert rec["after"] == 2.0
    old = {"rows": [{"rank": 1, "token_id": 2, "edited_logit": 2.0, "logit_delta": 0.0, "target_probability": ""}]}
    new = {"rows": [{"rank": 1, "token_id": 2, "edited_logit": 7.0, "logit_delta": 0.0, "target_probability": ""}]}
    rec2 = edit_logits_from_table(state, old, new)
    assert rec2 is not None
    assert math.isclose(float(state.logits[0, 2]), 7.0)


def test_target_probability_edit_is_finite():
    state = KVRuntimeState(logits=torch.zeros(1, 4), supported=True)
    edit_logits_value(state, 1, "target_probability", 0.5, 1.0)
    assert torch.isfinite(state.logits).all()
    prob = torch.softmax(state.logits.float(), dim=-1)[0, 1].item()
    assert 0.49 < prob < 0.51
