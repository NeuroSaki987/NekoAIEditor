import torch

from nekoai.kv_cache_debugger import (
    KVRuntimeState,
    current_context_tokens,
    edit_cache_token_positions,
    inspect_token_kv_slice,
    search_tokens_in_state,
    tokenize_text,
    token_kv_index,
    token_kv_layer_stats,
)
from nekoai.model_manager import ModelSession


class FakeTokenizer:
    pieces = {1: "hello", 2: " world", 3: " cat", 4: "!", 198: "\\n"}
    token_pieces = {1: "hello", 2: "Ġworld", 3: "Ġcat", 4: "!", 198: "Ċ"}

    def encode(self, text, add_special_tokens=False):
        mapping = {
            "hello": [1],
            "world": [2],
            "hello world": [1, 2],
            "cat": [3],
            "!": [4],
            "\n": [198],
        }
        return list(mapping.get(text, []))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces.get(int(i), f"<{int(i)}>") for i in ids)

    def convert_ids_to_tokens(self, token_id):
        return self.token_pieces.get(int(token_id), f"<{int(token_id)}>")


def fake_session_and_state():
    session = ModelSession(model_id="fake", model=object(), tokenizer=FakeTokenizer())
    cache = tuple((torch.zeros(1, 2, 4, 3), torch.ones(1, 2, 4, 3)) for _ in range(2))
    state = KVRuntimeState(model_id="fake", input_ids=[1, 2, 3, 2], cache=cache, supported=True)
    return session, state


def test_tokenize_and_search_phrase_positions():
    session, state = fake_session_and_state()
    tokens = tokenize_text(session, "hello world")
    assert [row["token_id"] for row in tokens] == [1, 2]
    rows = search_tokens_in_state(session, state, "hello world", mode="encoded phrase")
    assert len(rows) == 1
    assert rows[0]["position"] == 0
    assert rows[0]["end_position"] == 1
    assert rows[0]["slice_all_heads"] == ":, :, 0, :"


def test_token_search_decoded_contains_and_id():
    session, state = fake_session_and_state()
    contains = search_tokens_in_state(session, state, "world", mode="decoded token contains")
    assert [row["position"] for row in contains] == [1, 3]
    by_id = search_tokens_in_state(session, state, "2", mode="token id")
    assert [row["position"] for row in by_id] == [1, 3]


def test_token_kv_index_stats_inspect_and_edit():
    _session, state = fake_session_and_state()
    assert token_kv_index(state.cache, 0, "key", 1) == ":, :, 1, :"
    assert token_kv_index(state.cache, 0, "key", 1, head=1, dim_start=1, dim_count=2) == ":, 1, 1, 1:3"
    layer_stats = token_kv_layer_stats(state.cache, 1)
    assert len(layer_stats) == 2
    inspected = inspect_token_kv_slice(state.cache, 1, 0, "key", head=1, dim_start=0, dim_count=2)
    assert inspected["slice"] == ":, 1, 1, 0:2"
    rec = edit_cache_token_positions(state.cache, [1, 3], 0, "key", "add", 2.0, 0.5)
    assert rec["position_count"] == 2
    assert torch.allclose(state.cache[0][0][:, :, 1, :], torch.ones_like(state.cache[0][0][:, :, 1, :]))
    assert torch.allclose(state.cache[0][0][:, :, 3, :], torch.ones_like(state.cache[0][0][:, :, 3, :]))


def test_current_context_tokens_has_slices():
    session, state = fake_session_and_state()
    rows = current_context_tokens(session, state, limit=2)
    assert len(rows) == 2
    assert rows[1]["slice_all_heads"] == ":, :, 1, :"
