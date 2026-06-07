from __future__ import annotations

"""Compatibility helpers for Gradio runtime edge cases.

This module keeps NekoAIEditor usable across several Gradio / gradio_client /
Pydantic / Windows proxy combinations that otherwise fail before the UI opens.
"""

import asyncio
import os
from typing import Any
from urllib.parse import urlsplit


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
_NO_PROXY_ITEMS = ["127.0.0.1", "localhost", "::1", "0.0.0.0"]


def _append_env_list(name: str, items: list[str]) -> None:
    existing = os.environ.get(name, "")
    parts = [p.strip() for p in existing.replace(";", ",").split(",") if p.strip()]
    lowered = {p.lower() for p in parts}
    for item in items:
        if item.lower() not in lowered:
            parts.append(item)
    os.environ[name] = ",".join(parts)


def configure_localhost_proxy_bypass() -> None:
    """Make local Gradio self-check requests ignore system HTTP proxies.

    Gradio's launch path performs an internal HTTP request to
    ``/gradio_api/startup-events``. On Windows systems with Clash/V2Ray/other
    proxy environment variables, HTTPX can route that loopback request through a
    proxy, which often returns 502.  The UI server is actually running, but
    Gradio aborts because the self-check did not reach localhost.
    """

    _append_env_list("NO_PROXY", _NO_PROXY_ITEMS)
    _append_env_list("no_proxy", _NO_PROXY_ITEMS)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def _is_loopback_url(url: Any) -> bool:
    try:
        parsed = urlsplit(str(url))
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return host in _LOOPBACK_HOSTS


def apply_httpx_loopback_patch() -> bool:
    """Patch HTTPX so Gradio localhost startup calls never use proxies.

    Returns True when a new patch was applied.  The patch is intentionally narrow:
    only top-level ``httpx.get`` calls targeting loopback URLs are forced to use
    ``trust_env=False``.  All external Hugging Face / user network traffic keeps
    the user's normal proxy settings.
    """

    try:
        import httpx
    except Exception:
        return False

    if getattr(httpx, "_neko_loopback_patch", False):
        return False

    original_get = httpx.get

    def safe_get(url: Any, *args: Any, **kwargs: Any):
        if _is_loopback_url(url):
            # Make the Gradio /startup-events self-check ignore HTTP_PROXY and
            # HTTPS_PROXY.  Do not override explicit user arguments.
            kwargs.setdefault("trust_env", False)
        return original_get(url, *args, **kwargs)

    httpx.get = safe_get  # type: ignore[assignment]
    httpx._neko_loopback_patch = True  # type: ignore[attr-defined]
    httpx._neko_original_get = original_get  # type: ignore[attr-defined]
    return True


def apply_gradio_schema_patch() -> bool:
    """Patch gradio_client's JSON-schema renderer to tolerate boolean schemas.

    Some Gradio + gradio_client + Pydantic combinations generate JSON Schema
    nodes such as ``additionalProperties: false``. Older gradio_client versions
    assume every schema node is a dictionary and crash while serving /api/info
    with ``TypeError: argument of type 'bool' is not iterable``.
    """

    try:
        import gradio_client.utils as client_utils
    except Exception:
        return False

    if getattr(client_utils, "_neko_bool_schema_patch", False):
        return False

    original_get_type = getattr(client_utils, "get_type", None)
    original_json_schema_to_python_type = getattr(client_utils, "json_schema_to_python_type", None)
    original_inner = getattr(client_utils, "_json_schema_to_python_type", None)

    if original_inner is None:
        return False

    def safe_get_type(schema: Any):
        if not isinstance(schema, dict):
            return {}
        if original_get_type is None:
            return schema.get("type", {})
        return original_get_type(schema)

    def safe_inner(schema: Any, defs: Any = None) -> str:
        if isinstance(schema, bool) or schema is None:
            return "Any"
        try:
            return original_inner(schema, defs)
        except (TypeError, AttributeError) as exc:
            message = str(exc)
            if isinstance(schema, bool) or "argument of type 'bool' is not iterable" in message or "object has no attribute 'get'" in message:
                return "Any"
            raise

    def safe_json_schema_to_python_type(schema: Any) -> str:
        if not isinstance(schema, dict):
            return "Any"
        if original_json_schema_to_python_type is None:
            return safe_inner(schema, schema.get("$defs"))
        try:
            return original_json_schema_to_python_type(schema)
        except (TypeError, AttributeError) as exc:
            message = str(exc)
            if "argument of type 'bool' is not iterable" in message or "object has no attribute 'get'" in message:
                return safe_inner(schema, schema.get("$defs")).replace(
                    getattr(client_utils, "CURRENT_FILE_DATA_FORMAT", "FileData"),
                    "filepath",
                )
            raise

    client_utils.get_type = safe_get_type
    client_utils._json_schema_to_python_type = safe_inner
    client_utils.json_schema_to_python_type = safe_json_schema_to_python_type
    client_utils._neko_bool_schema_patch = True
    return True


def apply_all_runtime_patches() -> dict[str, bool]:
    """Apply all safe startup patches and return a small status report."""

    configure_localhost_proxy_bypass()
    return {
        "schema_patch": apply_gradio_schema_patch(),
        "httpx_loopback_patch": apply_httpx_loopback_patch(),
    }


def trigger_startup_events_if_needed(demo: Any) -> str:
    """Run Gradio startup events manually when launch self-check was blocked.

    This is a fallback for cases where a local proxy still interferes with the
    Gradio startup-events HTTP check.  It only runs if Gradio already created the
    FastAPI app and marked the UI server as running.
    """

    app = getattr(demo, "app", None)
    if app is None:
        return "no app object"
    if getattr(app, "startup_events_triggered", False):
        return "already triggered"

    demo.run_startup_events()
    extra = getattr(demo, "run_extra_startup_events", None)
    if callable(extra):
        try:
            asyncio.run(extra())
        except RuntimeError:
            # If an event loop is already active, schedule best-effort execution.
            loop = asyncio.get_event_loop()
            loop.run_until_complete(extra())
    app.startup_events_triggered = True
    return "triggered"


def is_gradio_startup_proxy_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "startup-events" in message and ("502" in message or "proxy" in message or "localhost" in message)
