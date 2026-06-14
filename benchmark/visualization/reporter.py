from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.models import BenchmarkRun, MemoryArchitecture, MetricsSnapshot

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None


ARCH_LABELS = {
    MemoryArchitecture.TRANSCRIPT_REPLAY: "Transcript Replay",
    MemoryArchitecture.HYBRID_RETRIEVAL: "Hybrid Retrieval",
    MemoryArchitecture.RECONSTRUCTED_STATE: "Reconstructed State",
}

ARCH_COLORS = {
    MemoryArchitecture.TRANSCRIPT_REPLAY: "#636efa",
    MemoryArchitecture.HYBRID_RETRIEVAL: "#ef553b",
    MemoryArchitecture.RECONSTRUCTED_STATE: "#00cc96",
}

PLOTLY_AVAILABLE = go is not None


def _metrics(arch_name: str, results: dict) -> list[MetricsSnapshot]:
    run: BenchmarkRun = results[arch_name]["run"]
    return run.metrics


def _arch_key(results: dict) -> list[str]:
    return list(results.keys())


def _known_arch(key: str) -> MemoryArchitecture | None:
    for arch in MemoryArchitecture:
        if arch.value == key:
            return arch
    return None


def _label(key: str) -> str:
    arch = _known_arch(key)
    return ARCH_LABELS.get(arch, key) if arch else key


def _color(key: str) -> str:
    arch = _known_arch(key)
    return ARCH_COLORS.get(arch, "#636efa") if arch else "#636efa"


def plot_context_growth(results: dict) -> go.Figure | None:
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    for key in _arch_key(results):
        ms = _metrics(key, results)
        turns = [m.turn_id for m in ms]
        sizes = [m.context_size_tokens for m in ms]
        fig.add_trace(go.Scatter(
            x=turns, y=sizes, mode="lines+markers",
            name=_label(key), line=dict(color=_color(key)),
        ))
    fig.update_layout(
        title="Context Size Growth Over Session",
        xaxis_title="Turn",
        yaxis_title="Context Size (tokens)",
        legend=dict(x=0, y=1),
        template="plotly_white",
    )
    return fig


def plot_latency_stability(results: dict) -> go.Figure | None:
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    for key in _arch_key(results):
        ms = _metrics(key, results)
        turns = [m.turn_id for m in ms]
        latencies = [m.total_latency_ms for m in ms]
        fig.add_trace(go.Scatter(
            x=turns, y=latencies, mode="lines+markers",
            name=_label(key), line=dict(color=_color(key)),
        ))
    fig.update_layout(
        title="Latency Stability Over Session",
        xaxis_title="Turn",
        yaxis_title="Total Latency (ms)",
        legend=dict(x=0, y=1),
        template="plotly_white",
    )
    return fig


def plot_token_cost(results: dict) -> go.Figure | None:
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    for key in _arch_key(results):
        ms = _metrics(key, results)
        turns = [m.turn_id for m in ms]
        cumulative = []
        total = 0
        for m in ms:
            total += m.input_tokens + m.output_tokens
            cumulative.append(total)
        fig.add_trace(go.Scatter(
            x=turns, y=cumulative, mode="lines+markers",
            name=_label(key), line=dict(color=_color(key)),
        ))
    fig.update_layout(
        title="Cumulative Token Cost Over Session",
        xaxis_title="Turn",
        yaxis_title="Total Tokens Consumed",
        legend=dict(x=0, y=1),
        template="plotly_white",
    )
    return fig


def plot_recall_accuracy(results: dict) -> go.Figure | None:
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    for key in _arch_key(results):
        ms = _metrics(key, results)
        turns = [m.turn_id for m in ms]
        cumulative = []
        total = 0
        for m in ms:
            total += m.facts_recalled_this_turn
            cumulative.append(total)
        fig.add_trace(go.Scatter(
            x=turns, y=cumulative, mode="lines+markers",
            name=_label(key), line=dict(color=_color(key)),
        ))
    fig.update_layout(
        title="Cumulative Facts Recalled Over Session",
        xaxis_title="Turn",
        yaxis_title="Total Facts Recalled",
        legend=dict(x=0, y=1),
        template="plotly_white",
    )
    return fig


def plot_hallucination_rate(results: dict) -> go.Figure | None:
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    for key in _arch_key(results):
        ms = _metrics(key, results)
        turns = [m.turn_id for m in ms]
        counts = [len(m.hallucination_indicators) for m in ms]
        fig.add_trace(go.Bar(
            x=turns, y=counts, name=_label(key),
            marker_color=_color(key), opacity=0.7,
        ))
    fig.update_layout(
        title="Hallucination Events Per Turn",
        xaxis_title="Turn",
        yaxis_title="Hallucination Count",
        barmode="group",
        template="plotly_white",
    )
    return fig


def plot_memory_growth(results: dict) -> go.Figure | None:
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    for key in _arch_key(results):
        ms = _metrics(key, results)
        turns = [m.turn_id for m in ms]
        counts = [m.active_memory_count for m in ms]
        fig.add_trace(go.Scatter(
            x=turns, y=counts, mode="lines+markers",
            name=_label(key), line=dict(color=_color(key)),
        ))
    fig.update_layout(
        title="Memory / Checkpoint Growth Over Session",
        xaxis_title="Turn",
        yaxis_title="Active Memory Count",
        legend=dict(x=0, y=1),
        template="plotly_white",
    )
    return fig


def generate_report(
    results: dict[str, dict[str, Any]],
    output_dir: str = "benchmark_report",
    scenario_name: str = "comparison",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, str] = {}

    plot_fns = {
        "context_growth": (plot_context_growth, "Context Size Growth"),
        "latency_stability": (plot_latency_stability, "Latency Stability"),
        "token_cost": (plot_token_cost, "Cumulative Token Cost"),
        "recall_accuracy": (plot_recall_accuracy, "Facts Recalled"),
        "hallucination_rate": (plot_hallucination_rate, "Hallucination Events"),
        "memory_growth": (plot_memory_growth, "Memory Growth"),
    }

    hmtl_parts = [f"<h1>Benchmark Report — {scenario_name}</h1>"]

    for plot_key, (fn, title) in plot_fns.items():
        fig = fn(results)
        if fig is None:
            continue

        html_file = output_path / f"{plot_key}.html"
        fig.write_html(str(html_file), include_plotlyjs="cdn")
        saved_files[f"{plot_key}.html"] = str(html_file)

        hmtl_parts.append(f"<h2>{title}</h2>")
        hmtl_parts.append(
            f'<iframe src="{plot_key}.html"'
            ' width="100%" height="500" frameborder="0"></iframe>'
        )

    hmtl_parts.append("<h2>Score Comparison</h2>")
    score_table = "<table border='1' cellpadding='6'><tr><th>Metric</th>"
    for key in _arch_key(results):
        score_table += f"<th>{_label(key)}</th>"
    score_table += "</tr>"

    all_score_names: set[str] = set()
    for key in _arch_key(results):
        all_score_names.update(results[key]["scores"].keys())

    sorted_names = sorted(all_score_names)
    for name in sorted_names:
        score_table += f"<tr><td>{name}</td>"
        for key in _arch_key(results):
            val = results[key]["scores"].get(name, "—")
            score_table += f"<td>{val}</td>"
        score_table += "</tr>"
    score_table += "</table>"
    hmtl_parts.append(score_table)

    hmtl_parts.append("<h2>Run Summary</h2>")
    for key in _arch_key(results):
        run: BenchmarkRun = results[key]["run"]
        hmtl_parts.append(f"<h3>{_label(key)}</h3>")
        hmtl_parts.append(
            "<ul>"
            f"<li>Total turns: {len(run.metrics)}</li>"
            f"<li>Input tokens: {run.total_input_tokens}</li>"
            f"<li>Output tokens: {run.total_output_tokens}</li>"
            f"<li>Total latency: {run.total_latency_ms:.1f}ms</li>"
            f"<li>Decisions matched: {run.decisions_matched}/{run.decisions_total}</li>"
            f"<li>Facts recalled: {run.facts_recalled}/{run.facts_total}</li>"
            f"<li>Hallucination events: {run.hallucination_events}</li>"
            f"<li>Contradictions: {run.contradictions_detected}</li>"
            "</ul>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <title>Benchmark Report — {scenario_name}</title>
    <style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 2rem;
    background: #f8f9fa;
}}
h1, h2, h3 {{
    color: #1a1a1a;
}}
img {{
    border: 1px solid #ddd;
    border-radius: 4px;
    margin: 1rem 0;
}}
table {{
    border-collapse: collapse;
    margin: 1rem 0;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 8px 12px;
}}
th {{
    background: #4a6cf7;
    color: white;
}}
tr:nth-child(even) {{
    background: #f2f2f2;
}}
    </style>
</head>
<body>
{"".join(hmtl_parts)}
</body>
</html>"""

    html_path = output_path / "report.html"
    html_path.write_text(html_content, encoding="utf-8")
    saved_files["report.html"] = str(html_path)

    return saved_files
