from __future__ import annotations

import plotly.graph_objects as go
import torch

from .model_manager import ModelSession
from .tensor_ops import parse_index_expr


def empty_figure(title: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def weight_histogram(session: ModelSession, name: str, index_expr: str = ":") -> go.Figure:
    p = session.get_parameter(name)
    idx = parse_index_expr(index_expr)
    with torch.no_grad():
        view = p.data[idx] if idx else p.data
        x = view.detach().reshape(-1)
        if x.numel() > 200_000:
            step = max(1, int(x.numel()) // 200_000)
            x = x[::step][:200_000]
        values = x.float().cpu().numpy()
    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=96)])
    fig.update_layout(template="plotly_dark", title=f"Weight histogram: {name}", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def weight_heatmap(session: ModelSession, name: str, index_expr: str = ":", max_rows: int = 96, max_cols: int = 96) -> go.Figure:
    p = session.get_parameter(name)
    idx = parse_index_expr(index_expr)
    with torch.no_grad():
        x = p.data[idx] if idx else p.data
        x = x.detach().float().cpu()
        if x.ndim == 0:
            z = [[float(x.item())]]
        elif x.ndim == 1:
            z = [x[:max_cols].tolist()]
        else:
            if x.ndim > 2:
                x = x.reshape(x.shape[0], -1)
            z = x[:max_rows, :max_cols].tolist()
    fig = go.Figure(data=[go.Heatmap(z=z, colorbar={"title": "value"})])
    fig.update_layout(template="plotly_dark", title=f"Weight heatmap: {name}", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def kv_rms_figure(rows: list[dict]) -> go.Figure:
    if not rows:
        return empty_figure("No KV cache")
    x = [r.get("layer") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[r.get("k_rms") for r in rows], mode="lines+markers", name="K RMS"))
    fig.add_trace(go.Scatter(x=x, y=[r.get("v_rms") for r in rows], mode="lines+markers", name="V RMS"))
    fig.update_layout(template="plotly_dark", title="KV cache RMS by layer", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig
