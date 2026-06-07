import torch

from nekoai.tensor_ops import apply_tensor_edit_, parse_index_expr


def test_parse_index_expr():
    idx = parse_index_expr("0, 1:3")
    assert idx[0] == 0
    assert idx[1].start == 1
    assert idx[1].stop == 3


def test_apply_tensor_edit():
    x = torch.zeros(2, 2)
    apply_tensor_edit_(x, "add", 2.0, 0.5)
    assert torch.allclose(x, torch.ones(2, 2))
