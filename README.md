
# NekoAIEditor

AI Debugger for LLMs — Inspect and Edit KV Cache & Weights in Real Time

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.0+-orange.svg)](https://gradio.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

## Description

NekoAIEditor is a local debugging tool for causal LLMs that lets you pause inference at token boundaries and directly inspect/modify KV Cache — the actual memory state attention uses, not just text prompts.

- **KV Cache Breakpoints** — Pause after prefill, edit any K/V tensor slice (e.g., `[:, 0, -1:, :]`), then resume decoding
- **Token → KV Locator** — Search any word/phrase/token ID, instantly locate its position in KV cache, edit that exact slice
- **Live Logits Editing** — Modify top token probabilities before sampling, or force specific tokens
- **Weight Editor** — Regex search parameters, visualize distributions, apply 10+ edit modes (add, clip, normalize, etc.)
- **Export** — Save edited models as safetensors or ZIP

Perfect for LLM interpretability, intervention experiments, and attention pattern debugging.



## Installing

CPU：
```bash
python -m venv .venv
source .venv/bin/activate
bash install_cpu.sh
python app.py
```

CUDA（N卡，包括RTX 50系）：
```bash
python -m venv .venv
source .venv/bin/activate
bash install_cuda_auto.sh
python app.py
```

Windows：
```bat
python -m venv .venv
.venv\Scripts\activate
install_cuda_auto.bat
run.bat
```

CUDA：
```bash
NEKOAI_PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu132 bash install_cuda_auto.sh
```

Open http://127.0.0.1:7860

## FIX Gradio 500

- Windows: `repair_gradio_stack.bat`
- Linux/macOS: `bash repair_gradio_stack.sh`

## Star History
![Star History Chart](https://api.star-history.com/svg?repos=NeuroSaki987/NekoAIEditor&type=Date)



