import torch

from nekoai.kv_cache_debugger import (
    KVRuntimeState,
    current_context_token_rows,
    edit_cache_by_token_text,
    token_kv_layer_details,
    token_kv_slice_expr,
    tokenize_text_for_display,
)
from nekoai.model_manager import ModelSession


class TextTokenizer:
    pieces = {1: " hello", 2: " world", 3: " cat", 4: "!", 5: " hello"}

    def encode(self, text, add_special_tokens=False):
        mapping = {
            "hello": [1],
            " hello": [1],
            "world": [2],
            " world": [2],
            "hello world": [1, 2],
            " hello world": [1, 2],
            "cat": [3],
            "!": [4],
        }
        value = mapping.get(text)
        if value is not None:
            return list(value)
        try:
            return [int(text)]
        except Exception:
            return []

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces.get(int(i), f"<{int(i)}>") for i in ids)


def fake_session_and_state(seq_len=5, cache_len=5):
    session = ModelSession(model_id="fake", model=object(), tokenizer=TextTokenizer())
    cache = tuple((torch.zeros(1, 2, cache_len, 3), torch.ones(1, 2, cache_len, 3)) for _ in range(2))
    state = KVRuntimeState(model_id="fake", input_ids=[1, 2, 3, 4, 5][:seq_len], cache=cache, supported=True)
    return session, state


def test_text_query_finds_all_matching_context_positions():
    session, state = fake_session_and_state()
    rows = current_context_token_rows(session, state, "hello", "contains token text", max_results=10)
    assert [row["pos"] for row in rows] == [0, 4]
    assert [row["kv_pos"] for row in rows] == [0, 4]
    assert all(row["cache_visible"] for row in rows)


def test_tokenized_phrase_query_reports_span_rows():
    session, state = fake_session_and_state()
    tokenized = tokenize_text_for_display(session, "hello world")
    assert [row["token_id"] for row in tokenized] == [1, 2]
    rows = current_context_token_rows(session, state, "hello world", "tokenized text sequence", max_results=10)
    assert len(rows) == 2
    assert rows[0]["span"] == "0:2"
    assert rows[1]["query_token_index"] == 1


def test_layer_details_and_slice_expression_for_matched_token():
    session, state = fake_session_and_state()
    rows = token_kv_layer_details(session, state, "world", "contains token text", "all", "both", "0", 0, 2, max_matches=10)
    assert len(rows) == 4  # 2 layers x key/value for one matched token/head
    assert rows[0]["kv_pos"] == 1
    assert rows[0]["slice"] == ":, 0:1, 1:2, 0:2"
    assert token_kv_slice_expr(1, head=0, dim_start=0, dim_count=2) == ":, 0:1, 1:2, 0:2"


def test_text_edit_updates_only_matched_token_head_and_layer():
    session, state = fake_session_and_state()
    record = edit_cache_by_token_text(
        session,
        state,
        query="hello",
        search_mode="contains token text",
        layers="0",
        component_scope="key",
        head_spec="0",
        dim_start=0,
        dim_count=0,
        edit_mode="add",
        value=2.0,
        strength=0.5,
        max_matches=10,
    )
    assert record["matched_positions"] == 2
    assert record["target_count"] == 2
    key_layer0 = state.cache[0][0]
    assert torch.allclose(key_layer0[:, 0, 0, :], torch.ones(1, 3))
    assert torch.allclose(key_layer0[:, 0, 4, :], torch.ones(1, 3))
    assert torch.allclose(key_layer0[:, 1, 0, :], torch.zeros(1, 3))
    assert torch.allclose(state.cache[1][0][:, 0, 0, :], torch.zeros(1, 3))


def test_sliding_cache_maps_absolute_position_to_kv_position():
    session, state = fake_session_and_state(seq_len=5, cache_len=3)
    rows = current_context_token_rows(session, state, "cat", "contains token text", max_results=10)
    assert rows[0]["pos"] == 2
    assert rows[0]["kv_pos"] == 0
    assert rows[0]["cache_visible"] is True
    hidden = current_context_token_rows(session, state, "hello", "contains token text", max_results=10)
    assert hidden[0]["pos"] == 0
    assert hidden[0]["kv_pos"] == -1
    assert hidden[0]["cache_visible"] is False
