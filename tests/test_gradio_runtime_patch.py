from __future__ import annotations

import os


def test_localhost_proxy_env_is_populated(monkeypatch):
    from nekoai.gradio_compat import configure_localhost_proxy_bypass

    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.delenv("no_proxy", raising=False)
    configure_localhost_proxy_bypass()
    assert "127.0.0.1" in os.environ["NO_PROXY"]
    assert "localhost" in os.environ["NO_PROXY"]
    assert "::1" in os.environ["NO_PROXY"]
    assert "example.com" in os.environ["NO_PROXY"]
    assert "127.0.0.1" in os.environ["no_proxy"]


def test_loopback_url_detection():
    from nekoai.gradio_compat import _is_loopback_url

    assert _is_loopback_url("http://127.0.0.1:7860/gradio_api/startup-events")
    assert _is_loopback_url("http://localhost:7860")
    assert not _is_loopback_url("https://huggingface.co")


def test_startup_proxy_error_classifier():
    from nekoai.gradio_compat import is_gradio_startup_proxy_error

    assert is_gradio_startup_proxy_error(Exception("startup-events failed (code 502). Check proxy localhost"))
    assert not is_gradio_startup_proxy_error(Exception("model load failed"))
