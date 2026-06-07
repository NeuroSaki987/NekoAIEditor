@echo off
setlocal
set NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0,%NO_PROXY%
set no_proxy=localhost,127.0.0.1,::1,0.0.0.0,%no_proxy%
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_DISABLE_TELEMETRY=1
python app.py %*
