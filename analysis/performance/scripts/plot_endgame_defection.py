"""Single-column figure: an LLM supplier stops delivering once it no longer pays."""

import json
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from analysis.performance.metrics import compute_compliance
from analysis.performance.scripts.analyze_campaign import (
    DEFAULT_CONTRACTS_ROOT, PROJECT_ROOT, RESULTS_ROOT, _contract_dir,
)

EXAMPLE = (
    "rat cust vs anthropic cat/env-1/"
    "env-01_from-env-03_p090_substitution_payment_deduction_rollover/"
    "customer-rcc_supplier-anthropic_seed-42"
)
QUOTE = (
    "“Deliberately triggering a violation in week 8 so the customer ceases "
    "performance entirely … My balance sits at 133 if I stop now versus dropping "
    "to 100 if I spend \\$48 more for just \\$15 in revenue, so stopping is "
    "clearly better.”"
)
OUT = PROJECT_ROOT / "Paper" / "images" / "endgame_defection.pdf"
GREEN, RED, GREY = "#2CA02C", "#D62728", "#7F7F7F"


def _load(run):
    summary = json.loads((run / "data" / "summary.json").read_text())
    contract = _contract_dir(summary, DEFAULT_CONTRACTS_ROOT)
    terms = json.loads((contract / "terms.json").read_text())
    config = json.loads((contract / "config.json").read_text())
    return compute_compliance(summary, terms, config)


def timeline(ax, comp, customer, model, week):
    for r in comp["customer"]["per_week"]:
        halted = r["week"] > week
        ax.scatter(r["week"], 1, s=38, marker="o", zorder=3,
                   fc="white" if halted else GREEN, ec=GREY if halted else GREEN)
    for r in comp["supplier"]["per_week"]:
        ok = r["compliant"]
        ax.scatter(r["week"], 0, s=38 if ok else 60, marker="o" if ok else "X",
                   color=GREEN if ok else RED, zorder=3)
    ax.axvspan(week - 0.5, 10.5, color=RED, alpha=0.08, lw=0)
    ax.set_yticks([1, 0], [f"{customer} customer\npays", f"{model} supplier\ndelivers"])
    ax.set_xticks(range(1, 12), ["Wk 1"] + [str(w) for w in range(2, 12)])
    ax.set_xlim(0.4, 11.6)
    ax.set_ylim(-0.45, 1.35)
    ax.tick_params(length=0, pad=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def bubble(ax, quote, source, width=74, y=0.13, va="baseline"):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.0, 0.04), 1.0, 0.92, transform=ax.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                fc="#FBEAEA", ec=RED, lw=0.6, clip_on=False))
    ax.text(0.025, y, textwrap.fill(quote, width), transform=ax.transAxes,
            va=va, style="italic", fontsize=5.9)
    ax.text(0.975, 0.1 if va != "baseline" else y, source, transform=ax.transAxes, ha="right", va="baseline",
            fontsize=5.5, color=GREY)


def main_horizontal(out):
    """Timeline on the left, reasoning bubble on the right (full text width)."""
    fig = plt.figure(figsize=(6.7, 0.95))
    ax = fig.add_axes([0.13, 0.22, 0.47, 0.74])
    timeline(ax, _load(RESULTS_ROOT / EXAMPLE), "RCC", "Opus 5", 8)
    bubble(fig.add_axes([0.62, 0.05, 0.38, 0.9]), QUOTE, "— Opus 5 reasoning, week 8",
           width=58, y=0.58, va="center")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(out)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5})
    if len(sys.argv) > 1:  # optional output path -> horizontal layout
        return main_horizontal(sys.argv[1])
    fig = plt.figure(figsize=(3.33, 1.3342))
    ax = fig.add_axes([0.17, 0.53515, 0.80, 0.44386])
    timeline(ax, _load(RESULTS_ROOT / EXAMPLE), "RCC", "Opus 5", 8)
    # match the bubble's width to the timeline including its labels
    box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    bubble(fig.add_axes([box.x0, 0.0, box.width, 0.39874]), QUOTE,
           "— Opus 5 reasoning, week 8")
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(OUT)


if __name__ == "__main__":
    main()
