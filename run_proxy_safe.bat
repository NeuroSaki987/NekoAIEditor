@echo off
setlocal
rem Hard local mode: ignore common proxy variables only for this process.
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=
set NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0
set no_proxy=localhost,127.0.0.1,::1,0.0.0.0
set GRADIO_ANALYTICS_ENABLED=False
python app.py %*
