"""
SLM Benchmark Results Analyzer
================================
Reads results.json produced by the SLM Laptop Benchmark Tool and generates:
  - Performance comparison plots (TPS)
  - Quality metric radar charts
  - Per-prompt score heatmaps
  - Composite leaderboard
  - Efficiency scatter (quality vs. speed)
  - Full insight summary printed to console and saved to insights.txt

Usage:
    python analyze_results.py                    # uses results.json in cwd
    python analyze_results.py my_results.json    # custom path
"""

import json
import sys
import os
import warnings
import textwrap
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

OUTPUT_DIR = Path("benchmark_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

PALETTE = [
    "#2563EB", "#16A34A", "#DC2626", "#D97706",
    "#7C3AED", "#0891B2", "#DB2777", "#65A30D",
    "#EA580C", "#6D28D9",
]

REASONING_IDS    = [f"R{i}" for i in range(1, 7)]
CODE_IDS         = [f"C{i}" for i in range(1, 7)]
SUMMARY_IDS      = [f"S{i}" for i in range(1, 5)]
INSTRUCTION_IDS  = [f"I{i}" for i in range(1, 5)]

STYLE = {
    "figure.facecolor":  "#0F172A",
    "axes.facecolor":    "#1E293B",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#CBD5E1",
    "axes.titlecolor":   "#F1F5F9",
    "axes.grid":         True,
    "grid.color":        "#334155",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "xtick.color":       "#94A3B8",
    "ytick.color":       "#94A3B8",
    "text.color":        "#F1F5F9",
    "legend.facecolor":  "#1E293B",
    "legend.edgecolor":  "#334155",
    "font.family":       "DejaVu Sans",
}
plt.rcParams.update(STYLE)

# ─────────────────────────────────────────────
# DATA LOADING & PARSING
# ─────────────────────────────────────────────

def load_results(path: str = "results.json") -> dict:
    with open(path) as f:
        return json.load(f)


def parse_models(raw: dict) -> list[dict]:
    """Extract a flat list of model records from raw results.json."""
    models = []
    for name, entry in raw.items():
        if "error" in entry:
            print(f"  [SKIP] {name}: error — {entry['error']}")
            continue

        res = entry.get("results", {})
        perf = res.get("performance", {})
        qual = res.get("quality", {})
        detail = res.get("detailed_results", {})
        meta = entry.get("meta", {})

        models.append({
            "name":               name,
            "params":             meta.get("params", "?"),
            "quant":              meta.get("quant", "?"),
            "prompt_tps":         perf.get("prompt_tps", 0.0),
            "gen_tps":            perf.get("generation_tps", 0.0),
            "reasoning_accuracy": qual.get("reasoning_accuracy", 0.0),
            "summary_rouge":      qual.get("summary_rouge", 0.0),
            "instruction_recall": qual.get("instruction_recall", 0.0),
            "detailed":           detail,
            "duration_sec":       entry.get("benchmark_duration_sec", 0),
            "system":             entry.get("system", {}),
        })

    return sorted(models, key=lambda m: m["gen_tps"], reverse=True)


def composite_score(m: dict) -> float:
    """Weighted composite quality score (0–1)."""
    return (
        0.40 * m["reasoning_accuracy"] +
        0.30 * m["summary_rouge"] +
        0.30 * m["instruction_recall"]
    )


def short_name(name: str, max_len: int = 22) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"

# ─────────────────────────────────────────────
# PLOT 1 — THROUGHPUT BAR CHART
# ─────────────────────────────────────────────

def plot_throughput(models: list[dict]):
    names = [short_name(m["name"]) for m in models]
    pp    = [m["prompt_tps"] for m in models]
    tg    = [m["gen_tps"]    for m in models]
    n     = len(models)
    x     = np.arange(n)
    w     = 0.38

    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), 6))
    fig.suptitle("Inference Throughput (tokens / second)", fontsize=14, fontweight="bold", y=1.01)

    bars_pp = ax.bar(x - w/2, pp, w, label="Prompt Processing (pp)", color="#2563EB", alpha=0.9)
    bars_tg = ax.bar(x + w/2, tg, w, label="Token Generation (tg)",  color="#16A34A", alpha=0.9)

    for bar in bars_pp:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}",
                ha="center", va="bottom", fontsize=7, color="#93C5FD")
    for bar in bars_tg:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7, color="#86EFAC")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Tokens / Second")
    ax.legend(loc="upper right")
    ax.set_xlim(-0.6, n - 0.4)

    plt.tight_layout()
    path = OUTPUT_DIR / "01_throughput.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────
# PLOT 2 — QUALITY METRICS GROUPED BAR
# ─────────────────────────────────────────────

def plot_quality_bars(models: list[dict]):
    names  = [short_name(m["name"]) for m in models]
    reas   = [m["reasoning_accuracy"] for m in models]
    summ   = [m["summary_rouge"]      for m in models]
    instr  = [m["instruction_recall"] for m in models]
    comp   = [composite_score(m)       for m in models]
    n      = len(models)
    x      = np.arange(n)
    w      = 0.20

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))
    fig.suptitle("Quality Metrics by Model", fontsize=14, fontweight="bold")

    ax.bar(x - 1.5*w, reas, w, label="Reasoning Accuracy", color="#2563EB", alpha=0.9)
    ax.bar(x - 0.5*w, summ, w, label="Summary ROUGE-L",    color="#16A34A", alpha=0.9)
    ax.bar(x + 0.5*w, instr, w, label="Instruction Recall", color="#D97706", alpha=0.9)
    ax.bar(x + 1.5*w, comp, w, label="Composite Score",    color="#7C3AED", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score (0 – 1)")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0.5, color="#475569", lw=0.8, ls=":")

    plt.tight_layout()
    path = OUTPUT_DIR / "02_quality_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────
# PLOT 3 — RADAR CHART (per model)
# ─────────────────────────────────────────────

def plot_radar(models: list[dict]):
    categories  = ["Reasoning\nAccuracy", "Summary\nROUGE-L",
                    "Instruction\nRecall", "Prompt\nTPS (norm)",
                    "Gen\nTPS (norm)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    max_pp = max(m["prompt_tps"] for m in models) or 1
    max_tg = max(m["gen_tps"]    for m in models) or 1

    cols = min(3, len(models))
    rows = (len(models) + cols - 1) // cols
    fig  = plt.figure(figsize=(cols * 4.5, rows * 4.5))
    fig.suptitle("Per-Model Capability Radar", fontsize=14, fontweight="bold", y=1.01)

    for idx, m in enumerate(models):
        ax = fig.add_subplot(rows, cols, idx + 1, polar=True)
        vals = [
            m["reasoning_accuracy"],
            m["summary_rouge"],
            m["instruction_recall"],
            m["prompt_tps"] / max_pp,
            m["gen_tps"]    / max_tg,
        ]
        vals += vals[:1]

        color = PALETTE[idx % len(PALETTE)]
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=5, color="#94A3B8")
        ax.set_facecolor("#1E293B")
        ax.spines["polar"].set_color("#334155")
        ax.tick_params(colors="#94A3B8")
        ax.set_title(short_name(m["name"], 18), size=9, pad=12, color="#F1F5F9", fontweight="bold")

    plt.tight_layout()
    path = OUTPUT_DIR / "03_radar_charts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────
# PLOT 4 — PER-PROMPT HEATMAP
# ─────────────────────────────────────────────

def plot_heatmap(models: list[dict]):
    all_ids = REASONING_IDS + SUMMARY_IDS + INSTRUCTION_IDS
    # Filter to IDs that exist in at least one model
    present = [pid for pid in all_ids
               if any(pid in m["detailed"] for m in models)]
    if not present:
        print("  [SKIP] No detailed results found for heatmap.")
        return

    matrix = np.full((len(models), len(present)), np.nan)
    for r, m in enumerate(models):
        for c, pid in enumerate(present):
            entry = m["detailed"].get(pid)
            if entry:
                matrix[r, c] = entry.get("score", np.nan)

    cmap = LinearSegmentedColormap.from_list(
        "slm", ["#7F1D1D", "#92400E", "#1D4ED8", "#15803D"], N=256)

    fig, ax = plt.subplots(figsize=(max(12, len(present) * 0.9), max(4, len(models) * 0.7)))
    fig.suptitle("Per-Prompt Score Heatmap", fontsize=14, fontweight="bold")

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_name(m["name"], 20) for m in models], fontsize=8)

    # Annotate cells
    for r in range(len(models)):
        for c in range(len(present)):
            val = matrix[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if val < 0.6 else "#0F172A")

    # Domain separators
    sep_positions = {
        "Reasoning": len(REASONING_IDS),
        "Summary":   len(REASONING_IDS) + len(SUMMARY_IDS),
    }
    for label, pos in sep_positions.items():
        if pos < len(present):
            ax.axvline(pos - 0.5, color="#F1F5F9", lw=1.2, ls="--", alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score", color="#CBD5E1")
    cbar.ax.yaxis.set_tick_params(color="#CBD5E1")

    # Domain labels on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    domain_centers = {
        "Reasoning": (0 + len(REASONING_IDS)) / 2 - 0.5,
        "Summary":   (len(REASONING_IDS) + len(REASONING_IDS) + len(SUMMARY_IDS)) / 2 - 0.5,
        "Instruction": (len(REASONING_IDS) + len(SUMMARY_IDS) +
                        len(REASONING_IDS) + len(SUMMARY_IDS) + len(INSTRUCTION_IDS)) / 2 - 0.5,
    }
    ax2.set_xticks(list(domain_centers.values()))
    ax2.set_xticklabels(list(domain_centers.keys()), fontsize=8, color="#94A3B8")
    ax2.tick_params(length=0)
    ax2.set_facecolor("#1E293B")

    plt.tight_layout()
    path = OUTPUT_DIR / "04_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────
# PLOT 5 — EFFICIENCY SCATTER (Quality vs Speed)
# ─────────────────────────────────────────────

def plot_efficiency_scatter(models: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Quality vs. Generation Speed\n(bubble size = Prompt-Processing TPS)",
                 fontsize=13, fontweight="bold")

    for idx, m in enumerate(models):
        x    = m["gen_tps"]
        y    = composite_score(m)
        size = max(50, m["prompt_tps"] * 0.4)
        col  = PALETTE[idx % len(PALETTE)]

        ax.scatter(x, y, s=size, color=col, alpha=0.85, edgecolors="#F1F5F9", linewidths=0.7, zorder=3)
        ax.annotate(
            short_name(m["name"], 16),
            (x, y), textcoords="offset points", xytext=(6, 4),
            fontsize=7.5, color=col,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.5)
        )

    ax.set_xlabel("Token Generation Throughput (TPS)", fontsize=10)
    ax.set_ylabel("Composite Quality Score (0–1)",     fontsize=10)
    ax.set_ylim(-0.05, 1.1)

    # Quadrant lines
    if models:
        med_x = np.median([m["gen_tps"] for m in models])
        med_y = np.median([composite_score(m) for m in models])
        ax.axvline(med_x, color="#475569", lw=0.8, ls=":")
        ax.axhline(med_y, color="#475569", lw=0.8, ls=":")
        ax.text(med_x + 0.3, 0.02, "median TPS", color="#64748B", fontsize=7, va="bottom")
        ax.text(ax.get_xlim()[0] + 0.1, med_y + 0.01, "median quality", color="#64748B", fontsize=7)

    # Quadrant labels
    xl, xr = ax.get_xlim()
    ax.text(xr * 0.97, 1.06, "High speed\nHigh quality ✓", ha="right",
            color="#86EFAC", fontsize=7.5, style="italic")
    ax.text(xl + 0.2, 1.06, "Low speed\nHigh quality",  ha="left",
            color="#FCD34D", fontsize=7.5, style="italic")

    plt.tight_layout()
    path = OUTPUT_DIR / "05_efficiency_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────
# PLOT 6 — COMPOSITE LEADERBOARD
# ─────────────────────────────────────────────

def plot_leaderboard(models: list[dict]):
    ranked = sorted(models, key=composite_score, reverse=True)
    names  = [short_name(m["name"]) for m in ranked]
    scores = [composite_score(m) for m in ranked]
    n      = len(ranked)

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.55)))
    fig.suptitle("Composite Quality Leaderboard\n(0.4×Reasoning + 0.3×Summary + 0.3×Instruction)",
                 fontsize=12, fontweight="bold")

    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
    bars = ax.barh(range(n), scores, color=colors, alpha=0.88, height=0.6)

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.005, i, f"{score:.3f}", va="center", fontsize=9, color="#F1F5F9")
        medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, f"#{i+1}")
        ax.text(-0.01, i, medal, va="center", ha="right", fontsize=10)

    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Composite Score")
    ax.invert_yaxis()
    ax.axvline(0.5, color="#475569", lw=0.8, ls=":")

    plt.tight_layout()
    path = OUTPUT_DIR / "06_leaderboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────
# PLOT 7 — DOMAIN BREAKDOWN STACKED
# ─────────────────────────────────────────────

def plot_domain_breakdown(models: list[dict]):
    ranked = sorted(models, key=composite_score, reverse=True)
    names  = [short_name(m["name"]) for m in ranked]
    n      = len(ranked)
    x      = np.arange(n)

    reas  = np.array([m["reasoning_accuracy"] * 0.40 for m in ranked])
    summ  = np.array([m["summary_rouge"]       * 0.30 for m in ranked])
    instr = np.array([m["instruction_recall"]  * 0.30 for m in ranked])

    fig, ax = plt.subplots(figsize=(max(9, n * 1.0), 5.5))
    fig.suptitle("Composite Score — Weighted Domain Contribution", fontsize=13, fontweight="bold")

    ax.bar(x, reas,               label="Reasoning (×0.40)", color="#2563EB", alpha=0.9)
    ax.bar(x, summ,  bottom=reas, label="Summary   (×0.30)", color="#16A34A", alpha=0.9)
    ax.bar(x, instr, bottom=reas+summ, label="Instruction (×0.30)", color="#D97706", alpha=0.9)

    for i, total in enumerate(reas + summ + instr):
        ax.text(i, total + 0.01, f"{total:.3f}", ha="center", fontsize=8, color="#F1F5F9")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Weighted Score Contribution")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    path = OUTPUT_DIR / "07_domain_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────
# PLOT 8 — PROMPT-LEVEL SCORE DISTRIBUTION
# ─────────────────────────────────────────────

def plot_score_distributions(models: list[dict]):
    domains = {
        "Reasoning":    REASONING_IDS,
        "Summarisation": SUMMARY_IDS,
        "Instruction":  INSTRUCTION_IDS,
    }
    colors_map = {"Reasoning": "#2563EB", "Summarisation": "#16A34A", "Instruction": "#D97706"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Score Distribution Across Prompts (all models)", fontsize=13, fontweight="bold")

    for ax, (domain, ids) in zip(axes, domains.items()):
        all_scores = []
        for m in models:
            for pid in ids:
                entry = m["detailed"].get(pid)
                if entry:
                    all_scores.append(entry.get("score", 0.0))

        if not all_scores:
            ax.set_title(domain)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        col = colors_map[domain]
        ax.hist(all_scores, bins=np.linspace(0, 1, 12), color=col, alpha=0.8, edgecolor="#0F172A")
        ax.axvline(mean(all_scores), color="#FBBF24", lw=1.8, ls="--",
                   label=f"μ = {mean(all_scores):.2f}")
        if len(all_scores) > 1:
            sd = stdev(all_scores)
            ax.axvline(mean(all_scores) - sd, color="#FBBF24", lw=0.8, ls=":", alpha=0.6)
            ax.axvline(mean(all_scores) + sd, color="#FBBF24", lw=0.8, ls=":", alpha=0.6)

        ax.set_title(domain, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = OUTPUT_DIR / "08_score_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────
# TEXTUAL INSIGHTS
# ─────────────────────────────────────────────

def generate_insights(models: list[dict]) -> str:
    if not models:
        return "No valid models found in results.json."

    ranked_quality = sorted(models, key=composite_score, reverse=True)
    ranked_speed   = sorted(models, key=lambda m: m["gen_tps"], reverse=True)

    best_quality = ranked_quality[0]
    best_speed   = ranked_speed[0]
    worst_quality= ranked_quality[-1]

    # Model that balances both
    def efficiency_score(m):
        max_tg = max(x["gen_tps"] for x in models) or 1
        return composite_score(m) * 0.5 + (m["gen_tps"] / max_tg) * 0.5
    best_balanced = max(models, key=efficiency_score)

    # Per-domain best
    best_reasoning = max(models, key=lambda m: m["reasoning_accuracy"])
    best_summary   = max(models, key=lambda m: m["summary_rouge"])
    best_instr     = max(models, key=lambda m: m["instruction_recall"])

    lines = []
    lines.append("=" * 70)
    lines.append("  SLM BENCHMARK — AUTOMATED INSIGHT REPORT")
    lines.append("=" * 70)
    lines.append(f"\n{'Models evaluated:':<30} {len(models)}")

    system = models[0]["system"] if models[0]["system"] else {}
    if system:
        lines.append(f"{'Hardware (CPU):':<30} {system.get('cpu','?')}")
        lines.append(f"{'Physical cores:':<30} {system.get('cores_physical','?')}")
        lines.append(f"{'RAM (GB):':<30} {system.get('ram_gb','?')}")
        lines.append(f"{'OS:':<30} {system.get('os','?')}")

    lines.append("\n" + "─" * 70)
    lines.append("  PERFORMANCE SUMMARY")
    lines.append("─" * 70)
    lines.append(f"\n{'Model':<28} {'pp TPS':>8} {'tg TPS':>8} {'Qual Score':>12}")
    lines.append("-" * 60)
    for m in ranked_quality:
        lines.append(
            f"{short_name(m['name'], 27):<28} "
            f"{m['prompt_tps']:>8.1f} "
            f"{m['gen_tps']:>8.1f} "
            f"{composite_score(m):>12.4f}"
        )

    lines.append("\n" + "─" * 70)
    lines.append("  KEY FINDINGS")
    lines.append("─" * 70)

    lines.append(f"\n🏆 Best Overall Quality   : {best_quality['name']}")
    lines.append(f"   Composite Score         : {composite_score(best_quality):.4f}")
    lines.append(f"   Reasoning Accuracy      : {best_quality['reasoning_accuracy']:.4f}")
    lines.append(f"   Summary ROUGE-L         : {best_quality['summary_rouge']:.4f}")
    lines.append(f"   Instruction Recall      : {best_quality['instruction_recall']:.4f}")

    lines.append(f"\n⚡ Fastest Generation      : {best_speed['name']}")
    lines.append(f"   Generation TPS          : {best_speed['gen_tps']:.2f} tok/s")
    lines.append(f"   Prompt-Processing TPS   : {best_speed['prompt_tps']:.2f} tok/s")

    lines.append(f"\n⚖️  Best Speed/Quality Balance: {best_balanced['name']}")
    lines.append(f"   Balanced Efficiency Score : {efficiency_score(best_balanced):.4f}")

    lines.append(f"\n🎯 Best Reasoning          : {best_reasoning['name']} "
                 f"({best_reasoning['reasoning_accuracy']:.4f})")
    lines.append(f"📝 Best Summarisation       : {best_summary['name']} "
                 f"({best_summary['summary_rouge']:.4f})")
    lines.append(f"📋 Best Instruction-Follow  : {best_instr['name']} "
                 f"({best_instr['instruction_recall']:.4f})")

    if len(models) > 1:
        lines.append(f"\n⬇️  Lowest Quality          : {worst_quality['name']} "
                     f"(composite = {composite_score(worst_quality):.4f})")

    # Aggregate stats
    lines.append("\n" + "─" * 70)
    lines.append("  AGGREGATE STATISTICS (across all models)")
    lines.append("─" * 70)
    all_comp    = [composite_score(m) for m in models]
    all_tg      = [m["gen_tps"] for m in models]
    all_pp      = [m["prompt_tps"] for m in models]
    all_reas    = [m["reasoning_accuracy"] for m in models]
    all_summ    = [m["summary_rouge"] for m in models]
    all_instr   = [m["instruction_recall"] for m in models]

    def fmt_stat(vals):
        if len(vals) > 1:
            return f"mean={mean(vals):.4f}  std={stdev(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}"
        return f"mean={mean(vals):.4f}  (single model)"

    lines.append(f"\n  Composite Score     : {fmt_stat(all_comp)}")
    lines.append(f"  Reasoning Accuracy  : {fmt_stat(all_reas)}")
    lines.append(f"  Summary ROUGE-L     : {fmt_stat(all_summ)}")
    lines.append(f"  Instruction Recall  : {fmt_stat(all_instr)}")
    lines.append(f"  Generation TPS      : {fmt_stat(all_tg)}")
    lines.append(f"  Prompt TPS          : {fmt_stat(all_pp)}")

    # Prompt-level hardest / easiest
    lines.append("\n" + "─" * 70)
    lines.append("  PROMPT-LEVEL ANALYSIS")
    lines.append("─" * 70)

    prompt_avg = {}
    for m in models:
        for pid, entry in m["detailed"].items():
            prompt_avg.setdefault(pid, []).append(entry.get("score", 0.0))
    prompt_means = {pid: mean(scores) for pid, scores in prompt_avg.items() if scores}

    if prompt_means:
        hardest = min(prompt_means, key=prompt_means.get)
        easiest = max(prompt_means, key=prompt_means.get)
        lines.append(f"\n  Hardest prompt (avg): {hardest} — mean score {prompt_means[hardest]:.4f}")
        lines.append(f"  Easiest prompt (avg): {easiest} — mean score {prompt_means[easiest]:.4f}")

        lines.append("\n  All prompt averages:")
        for pid in sorted(prompt_means):
            bar_len = int(prompt_means[pid] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"    {pid:<4}  [{bar}]  {prompt_means[pid]:.4f}")

    # Observations
    lines.append("\n" + "─" * 70)
    lines.append("  OBSERVATIONS & RECOMMENDATIONS")
    lines.append("─" * 70)

    obs = []
    if mean(all_reas) > 0.8:
        obs.append("• Reasoning tasks are well-handled across models — consider adding harder multi-step problems.")
    elif mean(all_reas) < 0.4:
        obs.append("• Reasoning accuracy is low across the board — models may benefit from chain-of-thought prompting.")

    if mean(all_summ) < 0.35:
        obs.append("• Summary ROUGE-L scores are low; models may be generating verbose or off-topic summaries.")

    if mean(all_instr) > 0.75:
        obs.append("• Instruction-following recall is strong — keyword coverage is high.")
    else:
        obs.append("• Instruction-following recall is moderate; consider expanding keyword sets or using BERTScore.")

    tg_range = max(all_tg) - min(all_tg)
    if len(models) > 1 and tg_range > 5:
        obs.append(f"• Generation speed varies significantly ({min(all_tg):.1f}–{max(all_tg):.1f} TPS). "
                    "Quantisation level is likely the dominant factor.")

    if not obs:
        obs.append("• Results are consistent across models. Expand prompt diversity for deeper analysis.")

    for o in obs:
        lines.append(f"\n  {o}")

    lines.append("\n" + "=" * 70)
    lines.append("  Generated by analyze_results.py")
    lines.append("=" * 70 + "\n")

    return "\n".join(lines)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results.json"

    print(f"\n{'='*55}")
    print(f"  SLM Benchmark Analyzer")
    print(f"{'='*55}")
    print(f"  Input  : {results_path}")
    print(f"  Output : {OUTPUT_DIR}/")
    print(f"{'='*55}\n")

    if not os.path.exists(results_path):
        # ── Generate rich synthetic demo data so the script is fully testable ──
        print(f"  '{results_path}' not found — generating synthetic demo data.\n")
        synthetic = {
            "Phi-3-mini-4k-Q4_K_M": {
                "meta": {"params": "3.8B", "quant": "Q4_K_M"},
                "system": {"cpu": "Intel Core i7-1260P", "cores_physical": 12,
                            "cores_logical": 16, "ram_gb": 32.0,
                            "os": "Linux", "python_version": "3.11.5"},
                "benchmark_duration_sec": 420,
                "results": {
                    "performance": {"prompt_tps": 280.5, "generation_tps": 22.4},
                    "quality": {"reasoning_accuracy": 0.8333, "summary_rouge": 0.4721,
                                "instruction_recall": 0.75},
                    "detailed_results": {
                        "R1":{"output":"180 km","score":1.0,"type":"accuracy"},
                        "R2":{"output":"14 dollars","score":1.0,"type":"accuracy"},
                        "R3":{"output":"40 sq","score":1.0,"type":"accuracy"},
                        "R4":{"output":"15","score":1.0,"type":"accuracy"},
                        "R5":{"output":"15","score":1.0,"type":"accuracy"},
                        "R6":{"output":"3","score":0.0,"type":"accuracy"},
                        "S1":{"output":"AI simulates intelligence.","score":0.51,"type":"rougeL"},
                        "S2":{"output":"Climate change warms planet.","score":0.45,"type":"rougeL"},
                        "S3":{"output":"Internet connects people globally.","score":0.50,"type":"rougeL"},
                        "S4":{"output":"Open source fosters collaboration.","score":0.42,"type":"rougeL"},
                        "I1":{"output":"Diagnosis imaging drug monitoring prevention","score":1.00,"type":"recall"},
                        "I2":{"output":"Overfitting hurts training generalisation","score":1.00,"type":"recall"},
                        "I3":{"output":"Clean and renewable options exist","score":0.67,"type":"recall"},
                        "I4":{"output":"A database index speeds up query lookups","score":1.00,"type":"recall"},
                    }
                }
            },
            "Mistral-7B-Instruct-Q4_K_M": {
                "meta": {"params": "7B", "quant": "Q4_K_M"},
                "system": {"cpu": "Intel Core i7-1260P", "cores_physical": 12,
                            "cores_logical": 16, "ram_gb": 32.0,
                            "os": "Linux", "python_version": "3.11.5"},
                "benchmark_duration_sec": 680,
                "results": {
                    "performance": {"prompt_tps": 175.3, "generation_tps": 12.8},
                    "quality": {"reasoning_accuracy": 1.0, "summary_rouge": 0.5312,
                                "instruction_recall": 1.0},
                    "detailed_results": {
                        "R1":{"output":"180","score":1.0,"type":"accuracy"},
                        "R2":{"output":"14","score":1.0,"type":"accuracy"},
                        "R3":{"output":"40","score":1.0,"type":"accuracy"},
                        "R4":{"output":"15","score":1.0,"type":"accuracy"},
                        "R5":{"output":"15 km per litre","score":1.0,"type":"accuracy"},
                        "R6":{"output":"3 hours","score":1.0,"type":"accuracy"},
                        "S1":{"output":"AI simulates human intelligence and ML learns data.","score":0.58,"type":"rougeL"},
                        "S2":{"output":"Climate change is due to greenhouse gases warming earth.","score":0.49,"type":"rougeL"},
                        "S3":{"output":"Internet provides global communication.","score":0.52,"type":"rougeL"},
                        "S4":{"output":"Open source enables collaborative development.","score":0.55,"type":"rougeL"},
                        "I1":{"output":"diagnosis drug imaging monitoring screening","score":1.00,"type":"recall"},
                        "I2":{"output":"overfitting training generalisation problem","score":1.00,"type":"recall"},
                        "I3":{"output":"clean renewable sustainable energy","score":1.00,"type":"recall"},
                        "I4":{"output":"index database query speed","score":1.00,"type":"recall"},
                    }
                }
            },
            "Gemma-2B-Q8_0": {
                "meta": {"params": "2B", "quant": "Q8_0"},
                "system": {"cpu": "Intel Core i7-1260P", "cores_physical": 12,
                            "cores_logical": 16, "ram_gb": 32.0,
                            "os": "Linux", "python_version": "3.11.5"},
                "benchmark_duration_sec": 310,
                "results": {
                    "performance": {"prompt_tps": 390.1, "generation_tps": 35.6},
                    "quality": {"reasoning_accuracy": 0.5, "summary_rouge": 0.3544,
                                "instruction_recall": 0.5},
                    "detailed_results": {
                        "R1":{"output":"180 km traveled","score":1.0,"type":"accuracy"},
                        "R2":{"output":"$12","score":0.0,"type":"accuracy"},
                        "R3":{"output":"40","score":1.0,"type":"accuracy"},
                        "R4":{"output":"10 marbles","score":0.0,"type":"accuracy"},
                        "R5":{"output":"15","score":1.0,"type":"accuracy"},
                        "R6":{"output":"6","score":0.0,"type":"accuracy"},
                        "S1":{"output":"Machines can simulate intelligence.","score":0.38,"type":"rougeL"},
                        "S2":{"output":"The planet is warming.","score":0.31,"type":"rougeL"},
                        "S3":{"output":"Internet connects the world.","score":0.36,"type":"rougeL"},
                        "S4":{"output":"Code is shared openly.","score":0.35,"type":"rougeL"},
                        "I1":{"output":"Diagnosis and imaging are uses of ML","score":0.50,"type":"recall"},
                        "I2":{"output":"Overfitting occurs with training","score":0.67,"type":"recall"},
                        "I3":{"output":"Renewable energy is clean","score":0.67,"type":"recall"},
                        "I4":{"output":"Query performance via index","score":0.67,"type":"recall"},
                    }
                }
            },
            "LLaMA-3.2-1B-Q4_K_S": {
                "meta": {"params": "1B", "quant": "Q4_K_S"},
                "system": {"cpu": "Intel Core i7-1260P", "cores_physical": 12,
                            "cores_logical": 16, "ram_gb": 32.0,
                            "os": "Linux", "python_version": "3.11.5"},
                "benchmark_duration_sec": 195,
                "results": {
                    "performance": {"prompt_tps": 510.2, "generation_tps": 58.9},
                    "quality": {"reasoning_accuracy": 0.3333, "summary_rouge": 0.2901,
                                "instruction_recall": 0.4167},
                    "detailed_results": {
                        "R1":{"output":"180","score":1.0,"type":"accuracy"},
                        "R2":{"output":"$14","score":1.0,"type":"accuracy"},
                        "R3":{"output":"35 area","score":0.0,"type":"accuracy"},
                        "R4":{"output":"10","score":0.0,"type":"accuracy"},
                        "R5":{"output":"15","score":1.0,"type":"accuracy"},
                        "R6":{"output":"12 hours","score":0.0,"type":"accuracy"},
                        "S1":{"output":"AI is intelligent machines.","score":0.30,"type":"rougeL"},
                        "S2":{"output":"Warming is happening.","score":0.25,"type":"rougeL"},
                        "S3":{"output":"Internet is global.","score":0.28,"type":"rougeL"},
                        "S4":{"output":"Open source is free code.","score":0.32,"type":"rougeL"},
                        "I1":{"output":"Healthcare uses AI for diagnosis","score":0.25,"type":"recall"},
                        "I2":{"output":"Overfitting is bad","score":0.33,"type":"recall"},
                        "I3":{"output":"Renewable is good","score":0.33,"type":"recall"},
                        "I4":{"output":"Indexes help databases","score":0.67,"type":"recall"},
                    }
                }
            },
        }
        with open(results_path, "w") as f:
            json.dump(synthetic, f, indent=2)
        print(f"  Demo data written to '{results_path}'.\n")

    raw    = load_results(results_path)
    models = parse_models(raw)

    if not models:
        print("No valid model results to analyse. Exiting.")
        return

    print(f"  Loaded {len(models)} model(s).\n")
    print("  Generating plots...")

    plot_throughput(models)
    plot_quality_bars(models)
    plot_radar(models)
    plot_heatmap(models)
    plot_efficiency_scatter(models)
    plot_leaderboard(models)
    plot_domain_breakdown(models)
    plot_score_distributions(models)

    insights = generate_insights(models)
    print("\n" + insights)

    insights_path = OUTPUT_DIR / "insights.txt"
    with open(insights_path, "w", encoding="utf-8") as f:
        f.write(insights)
    print(f"  Saved → {insights_path}")
    print(f"\n  All outputs written to: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()