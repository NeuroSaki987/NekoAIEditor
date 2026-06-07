#!/usr/bin/env bash
set -e
export NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,${no_proxy:-}"
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1
python app.py "$@"
