#!/usr/bin/env bash
set -e
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0"
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0"
export GRADIO_ANALYTICS_ENABLED=False
python app.py "$@"
