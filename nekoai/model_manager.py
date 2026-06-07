from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

from .utils import ensure_dir, safe_name, write_json


@dataclass
class ModelSession:
    model_id: str = ""
    model: Any = None
    tokenizer: Any = None
    device_mode: str = "auto"
    dtype_name: str = "auto"
    trust_remote_code: bool = False
    edit_backups: list[Any] = field(default_factory=list)
    edit_log: list[dict[str, Any]] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def require_loaded(self) -> None:
        if not self.loaded:
            raise RuntimeError("请先加载一个 causal LLM。")

    def input_device(self) -> torch.device:
        self.require_loaded()
        emb = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else None
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
        for p in self.model.parameters():
            return p.device
        return torch.device("cpu")

    def named_parameters(self):
        self.require_loaded()
        return self.model.named_parameters()

    def get_parameter(self, name: str) -> torch.nn.Parameter:
        self.require_loaded()
        needle = (name or "").strip()
        if not needle:
            raise KeyError("Parameter name is empty.")
        for n, p in self.model.named_parameters():
            if n == needle:
                return p
        # Allow selecting by alias when the alias is unique.
        matches = [real for real, alias in self.aliases.items() if alias == needle]
        if len(matches) == 1:
            return self.get_parameter(matches[0])
        raise KeyError(f"Parameter not found: {needle}")

    def append_edit(self, backup: Any) -> None:
        self.edit_backups.append(backup)
        self.edit_log.append(backup.record.__dict__)

    def undo_last_edit(self) -> str:
        from .weight_ops import restore_backup

        if not self.edit_backups:
            return "No edit to undo."
        backup = self.edit_backups.pop()
        restore_backup(self.get_parameter(backup.parameter), backup)
        self.edit_log.append({"timestamp": backup.record.timestamp, "target": backup.parameter, "mode": "undo", "index_expr": backup.index_expr})
        return f"Undid {backup.parameter}[{backup.index_expr}]"

    def set_alias(self, parameter: str, alias: str) -> str:
        self.require_loaded()
        name = (parameter or "").strip()
        self.get_parameter(name)
        clean = (alias or "").strip()
        if clean:
            self.aliases[name] = clean
            self.edit_log.append({"timestamp": "alias", "target": name, "mode": "set_alias", "alias": clean})
            return f"Alias set: {name} -> {clean}"
        self.aliases.pop(name, None)
        self.edit_log.append({"timestamp": "alias", "target": name, "mode": "clear_alias"})
        return f"Alias cleared: {name}"

    def export(self, export_dir: str, safe_serialization: bool = True) -> str:
        self.require_loaded()
        out = ensure_dir(export_dir)
        self.model.save_pretrained(out, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(out)
        write_json(
            out / "nekoai_edit_manifest.json",
            {
                "name": "NekoAIEditor v1.0 export",
                "source_model": self.model_id,
                "edit_log": self.edit_log,
                "aliases": self.aliases,
                "alias_note": "Aliases are UI/export manifest labels. Core PyTorch state_dict keys are preserved so the exported model remains loadable.",
            },
        )
        return str(out)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def choose_device(device_mode: str) -> str:
    mode = (device_mode or "auto").lower().strip()
    if mode == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if mode.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Run install_cuda_auto/install_cuda132 or choose CPU.")
        return mode
    return "cpu"


def _safe_bf16_supported() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def choose_dtype(dtype_name: str, device: str) -> torch.dtype:
    name = (dtype_name or "auto").lower().strip()
    if name == "auto":
        if not str(device).startswith("cuda"):
            return torch.float32
        return torch.bfloat16 if _safe_bf16_supported() else torch.float16
    mapping = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    if name not in mapping:
        raise ValueError("dtype must be auto/float32/float16/bfloat16")
    if not str(device).startswith("cuda") and mapping[name] in {torch.float16, torch.bfloat16}:
        # CPU kernels for fp16/bf16 are uneven across platforms; keep the model usable.
        return torch.float32
    return mapping[name]


def load_model_session(
    model_id_or_path: str,
    device_mode: str = "auto",
    dtype_name: str = "auto",
    trust_remote_code: bool = False,
    use_device_map: bool = True,
    revision: str | None = None,
) -> ModelSession:
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError("transformers is not installed. Run install_cpu/install_cuda_auto first.")
    if not (model_id_or_path or "").strip():
        raise ValueError("Model id/local path is empty.")
    clear_memory()
    device = choose_device(device_mode)
    dtype = choose_dtype(dtype_name, device)
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, **kwargs)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "torch_dtype": dtype, "low_cpu_mem_usage": True}
    if revision:
        model_kwargs["revision"] = revision
    if str(device).startswith("cuda") and use_device_map:
        model_kwargs["device_map"] = "auto"
    elif str(device).startswith("cuda"):
        # Supports cuda, cuda:0, cuda:1.
        model_kwargs["device_map"] = {"": device}
    else:
        model_kwargs["device_map"] = {"": "cpu"}
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **model_kwargs)
    model.eval()
    session = ModelSession(model_id_or_path, model, tokenizer, device, str(model_kwargs.get("torch_dtype", dtype)).replace("torch.", ""), trust_remote_code)
    return session


def recommended_cuda_index() -> str:
    """Return a practical PyTorch wheel index hint for this host."""
    env = os.environ.get("NEKOAI_PYTORCH_CUDA_INDEX", "").strip()
    if env:
        return env
    if not torch.cuda.is_available():
        return "CPU build active. For NVIDIA GPUs install a CUDA PyTorch wheel such as cu132/cu130 from the PyTorch selector."
    caps = []
    for i in range(torch.cuda.device_count()):
        try:
            caps.append(torch.cuda.get_device_capability(i))
        except Exception:
            pass
    if any(major >= 12 for major, _minor in caps):
        return "RTX 50/Blackwell class detected: prefer PyTorch CUDA 13.2 or 13.0 wheels; CUDA 12.8+ may work if your PyTorch build provides sm_120 support."
    return "CUDA GPU detected: use the newest PyTorch CUDA wheel supported by your driver; cu132/cu130/cu126 are tried by install_cuda_auto."


def compatibility_report() -> str:
    lines = [
        f"PyTorch: {torch.__version__}",
        f"CUDA runtime compiled in torch: {torch.version.cuda or 'not installed'}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"Recommended install path: {recommended_cuda_index()}",
    ]
    if torch.cuda.is_available():
        try:
            lines.append(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                props = torch.cuda.get_device_properties(i)
                free = total = None
                try:
                    free, total = torch.cuda.mem_get_info(i)
                except Exception:
                    pass
                mem = f"VRAM {props.total_memory/(1024**3):.2f} GiB"
                if free is not None and total is not None:
                    mem += f", free {free/(1024**3):.2f} GiB"
                lines.append(f"GPU {i}: {torch.cuda.get_device_name(i)}, CC {cap[0]}.{cap[1]}, {mem}")
                if cap[0] >= 12:
                    lines.append("  Blackwell/RTX 50 compute capability class detected. Use recent CUDA wheels and recent NVIDIA drivers.")
        except Exception as exc:
            lines.append(f"CUDA probe failed: {exc}")
    return "\n".join(lines)


def default_export_dir(model_id: str, root: str = "exports") -> str:
    return str(Path(root) / f"{safe_name(model_id)}-nekoai-edited")
