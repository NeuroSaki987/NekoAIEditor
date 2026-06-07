from __future__ import annotations

# These patches must run before importing gradio.  In particular, HTTPX reads
# proxy-related environment variables during local Gradio startup checks.
from nekoai.gradio_compat import (  # noqa: E402
    apply_all_runtime_patches,
    is_gradio_startup_proxy_error,
    trigger_startup_events_if_needed,
)

PATCH_STATUS = apply_all_runtime_patches()

import argparse
import inspect
import os
import sys
from pathlib import Path

import gradio as gr

from nekoai.ui import create_app  # noqa: E402


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NekoAIEditor v1.0")
    parser.add_argument("--host", default=os.environ.get("NEKOAI_HOST", "127.0.0.1"), help="Bind host, default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=_int_env("NEKOAI_PORT", 7860), help="Bind port, default: 7860")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--inbrowser", action="store_true", help="Open browser automatically")
    parser.add_argument("--quiet", action="store_true", help="Reduce Gradio logs")
    return parser.parse_args()


def _launch_kwargs(demo, args: argparse.Namespace) -> dict:
    kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "show_error": True,
        "share": bool(args.share),
        "inbrowser": bool(args.inbrowser),
        "quiet": bool(args.quiet),
        # Gradio 6 can start an experimental SSR node path depending on install.
        # NekoAIEditor is a local debugger; disabling SSR makes startup simpler
        # and avoids another localhost proxy hop.
        "ssr_mode": False,
    }
    params = inspect.signature(demo.launch).parameters
    if "show_api" in params:
        kwargs["show_api"] = False
    if "strict_cors" in params:
        kwargs["strict_cors"] = False
    if "enable_monitoring" in params:
        kwargs["enable_monitoring"] = False
    css = getattr(demo, "_neko_launch_css", None)
    if css and "css" in params:
        kwargs["css"] = css
    theme = getattr(demo, "_neko_launch_theme", None)
    if theme is not None and "theme" in params:
        kwargs["theme"] = theme
    return {k: v for k, v in kwargs.items() if k in params}


def _print_startup_banner(args: argparse.Namespace) -> None:
    print(f"NekoAIEditor v1.0 starting at http://{args.host}:{args.port}", flush=True)
    print(f"Gradio {gr.__version__}; schema/proxy compatibility patches active: {PATCH_STATUS}", flush=True)
    print(f"NO_PROXY={os.environ.get('NO_PROXY', '')}", flush=True)


def _fallback_block_after_startup_error(demo, exc: BaseException) -> None:
    if not getattr(demo, "is_running", False):
        raise exc
    status = trigger_startup_events_if_needed(demo)
    local_url = getattr(demo, "local_url", "http://127.0.0.1:7860")
    print("", flush=True)
    print("Gradio startup self-check was blocked, but the local server is running.", flush=True)
    print(f"Manual startup-events fallback: {status}", flush=True)
    print(f"Open this URL manually: {local_url}", flush=True)
    print("Tip: keep localhost/127.0.0.1/::1 in NO_PROXY or proxy bypass list.", flush=True)
    demo.block_thread()


def main() -> None:
    args = _parse_args()
    demo = create_app()
    _print_startup_banner(args)
    try:
        demo.queue(default_concurrency_limit=1).launch(**_launch_kwargs(demo, args))
    except Exception as exc:
        if is_gradio_startup_proxy_error(exc):
            _fallback_block_after_startup_error(demo, exc)
            return
        print("NekoAIEditor failed during startup:", file=sys.stderr, flush=True)
        print(repr(exc), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
