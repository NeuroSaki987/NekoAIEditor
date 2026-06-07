from __future__ import annotations


def test_gradio_bool_schema_patch_handles_additional_properties_false():
    from nekoai.gradio_compat import apply_gradio_schema_patch
    import gradio_client.utils as client_utils

    apply_gradio_schema_patch()
    schema = {
        "type": "object",
        "properties": {"payload": {"type": "object", "additionalProperties": False}},
    }
    assert "payload" in client_utils.json_schema_to_python_type(schema)
