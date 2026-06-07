from __future__ import annotations

import json
from typing import Any

import torch

DEFAULT_SYSTEM_PROMPT = "You are a helpful local open-source language model."


def build_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt or ""},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"System: {system_prompt or DEFAULT_SYSTEM_PROMPT}\n\nUser: {user_prompt or ''}\n\nAssistant:\n"


def filter_logits(logits: torch.Tensor, temperature: float = 0.0, top_k: int = 50, top_p: float = 0.95) -> torch.Tensor:
    out = logits.float().clone()
    if float(temperature) > 0:
        out = out / max(float(temperature), 1e-6)
    if top_k and int(top_k) > 0:
        k = min(int(top_k), out.shape[-1])
        kth = torch.topk(out, k=k, dim=-1).values[..., -1, None]
        out = torch.where(out < kth, torch.full_like(out, -float("inf")), out)
    if top_p and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(out, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        out = torch.full_like(out, -float("inf"))
        out.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return out


def sample_token(logits: torch.Tensor, temperature: float = 0.0, top_k: int = 50, top_p: float = 0.95) -> int:
    filtered = filter_logits(logits, temperature, top_k, top_p)
    if float(temperature) <= 0:
        return int(torch.argmax(filtered, dim=-1).item())
    probs = torch.softmax(filtered, dim=-1)
    if torch.isnan(probs).any() or float(probs.sum().item()) <= 0:
        return int(torch.argmax(filtered, dim=-1).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def top_tokens(tokenizer: Any, logits: torch.Tensor | None, k: int = 10, temperature: float = 0.0, top_k: int = 50, top_p: float = 0.95) -> list[dict[str, Any]]:
    if logits is None:
        return []
    raw = logits.float()
    filtered = filter_logits(logits, temperature, top_k, top_p)
    probs = torch.softmax(filtered.float(), dim=-1)
    values, indices = torch.topk(probs, k=min(int(k), probs.shape[-1]), dim=-1)
    rows: list[dict[str, Any]] = []
    for rank, (prob, token_id) in enumerate(zip(values[0].detach().cpu().tolist(), indices[0].detach().cpu().tolist()), start=1):
        tid = int(token_id)
        raw_logit = float(raw[0, tid].detach().cpu().item())
        filtered_logit = float(filtered.float()[0, tid].detach().cpu().item())
        rows.append({
            "rank": rank,
            "token_id": tid,
            "token": repr(tokenizer.decode([tid], skip_special_tokens=False)),
            "logit": raw_logit,
            "filtered_logit": filtered_logit,
            "probability": float(prob),
            "edited_logit": raw_logit,
            "logit_delta": 0.0,
            "target_probability": "",
        })
    return rows


def parse_token_override(tokenizer: Any, text: str | None) -> int | None:
    value = (text or "").strip()
    if not value:
        return None
    if value.lstrip("-").isdigit():
        return int(value)
    ids = tokenizer.encode(value, add_special_tokens=False)
    if not ids:
        raise ValueError("Token override text did not tokenize to anything.")
    if len(ids) > 1:
        raise ValueError(f"Token override must be exactly one token, got {len(ids)} ids: {ids}")
    return int(ids[0])


def parse_logit_bias(tokenizer: Any, text: str | None) -> dict[int, float]:
    if not text or not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Logit bias must be a JSON object.")
    result: dict[int, float] = {}
    for key, value in data.items():
        if str(key).strip().lstrip("-").isdigit():
            result[int(key)] = float(value)
        else:
            for token_id in tokenizer.encode(str(key), add_special_tokens=False):
                result[int(token_id)] = float(value)
    return result


def apply_logit_bias(logits: torch.Tensor, bias: dict[int, float]) -> torch.Tensor:
    if not bias:
        return logits
    out = logits.clone()
    vocab = out.shape[-1]
    for token_id, value in bias.items():
        if 0 <= int(token_id) < vocab:
            out[..., int(token_id)] += float(value)
    return out
