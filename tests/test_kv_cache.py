import torch

from nekoai.kv_cache_debugger import cache_summary, edit_cache_slice, inspect_cache_slice, iter_cache_layers


def fake_cache():
    return tuple((torch.zeros(1, 2, 3, 4), torch.ones(1, 2, 3, 4)) for _ in range(2))


def test_iter_legacy_cache():
    rows = iter_cache_layers(fake_cache())
    assert len(rows) == 2
    assert rows[0][1].shape == (1, 2, 3, 4)


def test_summary_and_edit():
    c = fake_cache()
    rows = cache_summary(c)
    assert rows[0]["k_heads"] == 2
    rec = edit_cache_slice(c, 0, "key", ":, 0, -1:, :", "add", 2.0, 0.5)
    assert rec["after"]["mean"] == 1.0
    data = inspect_cache_slice(c, 0, "key", ":, 0, -1:, :")
    assert data["stats"]["mean"] == 1.0
