#!/usr/bin/env bash
set -euo pipefail
# Fix Gradio API schema error: TypeError: argument of type 'bool' is not iterable
python -m pip uninstall -y gradio gradio_client pydantic pydantic_core
python -m pip install -r requirements.txt
