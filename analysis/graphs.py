

from __future__ import annotations
import os
import matplotlib.pyplot as plt 
import pandas as pd
import csv


import os
import re
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

# --------------------------------------------------------------------------- #
# Styling setup: dark background with pastel outlined bars
# --------------------------------------------------------------------------- #
plt.style.use("default") 
BASE_BG   = "#ffffff"
TICK_COLOR = "#1a1a1a"
GRID_COLOR = "#d0d0d0"
TITLE_COLOR = "#111111"
LABEL_COLOR = "#222222"

plt.rcParams.update(
    {
        "figure.facecolor": BASE_BG,
        "axes.facecolor": BASE_BG,
        "axes.edgecolor": "#b0b0b0",
        "axes.labelcolor": LABEL_COLOR,
        "text.color": LABEL_COLOR,
        "xtick.color": TICK_COLOR,
        "ytick.color": TICK_COLOR,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "axes.titleweight": "semibold",
        "axes.labelweight": "semibold",
        "legend.facecolor": BASE_BG,
        "legend.edgecolor": "#a0a0a0",
    }
)

# New palette: pastel fills with stronger outlines + different hatch set
MODEL_COLORS = [
    "#87bff2",
    "#f4a7b0",
    "#9ed7a5",
    "#f8d996",
    "#c5b7f5",
    "#ffcab1",
    "#a9d4ff",
    "#ffddf1",
]
MODEL_OUTLINES = [
    "#2a5d90",
    "#a22e50",
    "#2c7c53",
    "#a26a18",
    "#4d3b99",
    "#a14c3d",
    "#2f5f84",
    "#9a5686",
]
MODEL_HATCHES = ["\\\\", "//", "oo", "++", "..", "xx", "||", "**"]

# Where to save the output figures.
OUTPUT_DIR = Path("/work/pi_mrobson_smith_edu/ParEval_amt/analysis/visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Metric configurations
# --------------------------------------------------------------------------- #

METRICS = [
    # {
    #     "column": "strict_pass@1",
    #     "ylabel": "strict_pass@1",
    #     "title": "strict_pass@1 by Prompt Category and Model",
    #     "filename": "strict_pass1.png",
    #     "formatter": lambda v: f"{v:.2f}",
    #     "broken_axis": False,
    # },
    {
        "column": "speedup@1",
        "ylabel": "speedup@1",
        "title": "speedup@1 by Prompt Category and Model",
        "filename": "speedup1.png",
        "formatter": lambda v: f"{v:.2f}",
        "broken_axis": True,
    },
    # {
    #     "column": "efficiency@1",
    #     "ylabel": "efficiency@1",
    #     "title": "efficiency@1 by Prompt Category and Model",
    #     "filename": "efficiency1.png",
    #     "formatter": lambda v: f"{v:.2f}",
    #     "broken_axis": True,
    # },
]

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_prompt_name(raw_problem_type: str | float) -> str:
    """Infer a prompt category name from the raw 'problem type' column."""
    if pd.isna(raw_problem_type):
        return "Unknown"

    text = str(raw_problem_type).strip()
    for sep in (":", "|", "=>"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()

    if "/" in text or "\\" in text:
        text = re.split(r"[\\/]", text)[-1].strip()

    if "-" in text:
        dash_tokens = [token.strip() for token in text.split("-") if token.strip()]
        if dash_tokens:
            text = dash_tokens[0]

    return text or "Unknown"


def load_scores(csv_path: Path | str, metric_columns: Iterable[str]) -> pd.DataFrame:
    """Read a CSV file and return the relevant columns."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_columns = {"model", "problem type", *metric_columns}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
        )

    df = df[["model", "problem type", *metric_columns]].copy()
    df["prompt_category"] = df["problem type"].apply(parse_prompt_name)

    for metric in metric_columns:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    cleaned = df.dropna(subset=metric_columns)
    cleaned["source_csv"] = csv_path.name
    columns_to_keep = ["prompt_category", "model", "source_csv", *metric_columns]
    print(cleaned[columns_to_keep])
    return cleaned[columns_to_keep]


def determine_break_parameters(values: np.ndarray, low_threshold: float = 1.5):
    """
    Determine whether a y-axis break is useful and return the break parameters.

    Returns:
        (break_threshold, lower_ylim, upper_ylim_low, upper_ylim_high)
        or None if a break is not helpful.
    """
    arr = np.asarray(values, dtype=float).flatten()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    low_vals = arr[arr <= low_threshold]
    high_vals = arr[arr > low_threshold]

    if low_vals.size == 0 or high_vals.size == 0:
        return None
    if low_vals.size / arr.size < 0.5:
        return None

    lower_ylim = max(1.05, low_vals.max() * 1.1)
    break_threshold = lower_ylim

    high_vals = arr[arr > break_threshold]
    if high_vals.size == 0:
        high_vals = arr[arr > low_threshold]
        if high_vals.size == 0:
            return None
        break_threshold = max(low_threshold * 1.05, lower_ylim)
        lower_ylim = break_threshold

    upper_ylim_low = min(high_vals) * 0.95
    if upper_ylim_low <= lower_ylim:
        upper_ylim_low = lower_ylim * 1.05

    upper_ylim_high = max(high_vals) * 1.2
    if upper_ylim_high <= upper_ylim_low:
        upper_ylim_high = upper_ylim_low + abs(upper_ylim_low) * 0.5

    return break_threshold, lower_ylim, upper_ylim_low, upper_ylim_high


def _prepare_bar_style(index: int) -> dict:
    face = mcolors.to_rgba(MODEL_COLORS[index % len(MODEL_COLORS)], alpha=0.7)
    edge = MODEL_OUTLINES[index % len(MODEL_OUTLINES)]
    hatch = MODEL_HATCHES[index % len(MODEL_HATCHES)]
    return {
        "color": face,
        "edgecolor": edge,
        "linewidth": 1.8,
        "hatch": hatch,
        "alpha": 0.85,
        "path_effects": [
            patheffects.withStroke(linewidth=2.4, foreground="#00000088")
        ],
    }


def _build_custom_legend_handles(model_names):
    handles = [
        Patch(
            facecolor=mcolors.to_rgba(MODEL_COLORS[idx % len(MODEL_COLORS)], alpha=0.7),
            edgecolor=MODEL_OUTLINES[idx % len(MODEL_OUTLINES)],
            linewidth=1.8,
            hatch=MODEL_HATCHES[idx % len(MODEL_HATCHES)],
            label=model,
        )
        for idx, model in enumerate(model_names)
    ]
    return handles


def _add_bottom_legend(fig, handles, title="Model"):
    legend = fig.legend(
        handles=handles,
        title=title,
        loc="lower center",           # anchor point on the legend box
        bbox_to_anchor=(0.5, 1.0),    
        ncol=len(handles),
        handlelength=1.6,
        handleheight=1.2,
        frameon=True,
        framealpha=0.95,
        fontsize=11,
    )
    legend.get_title().set_color(LABEL_COLOR)
    legend.get_frame().set_facecolor("#f9f9f9")
    legend.get_frame().set_edgecolor("#b7b7b7")
    legend.get_frame().set_linewidth(1.2)
    fig.subplots_adjust(bottom=0.5)

def _style_axes(ax):
    ax.set_facecolor(BASE_BG)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color("#6b6f7e")
        spine.set_linewidth(0.9)


def _format_labels(ax, title, ylabel):
    ax.set_ylabel(ylabel, color=LABEL_COLOR, fontsize=13)
    #ax.set_title(title, fontsize=18, color=TITLE_COLOR, pad=16)
    ax.tick_params(axis="x", labelsize=12, colors=TICK_COLOR)
    ax.tick_params(axis="y", labelsize=11, colors=TICK_COLOR)

# Modify _plot_single_axis to always use 0-1 range:
def _plot_single_axis(
    prompt_categories,
    model_names,
    pivot,
    output_path,
    ylabel,
    title,
    value_formatter,
    show_plot,
    show_reference_line=False
) -> None:
    x_positions = np.arange(len(prompt_categories))
    num_models = len(model_names)
    bar_width = 0.8 / max(num_models, 1)

    # Increased padding for more space
    label_padding = 0.04  # Space between bar top and label start

    fig, ax = plt.subplots(
        figsize=(max(10, len(prompt_categories) * 1.6), 4.4), constrained_layout=True
    )
    fig.set_facecolor(BASE_BG)
    _style_axes(ax)

    for idx, model in enumerate(model_names):
        offsets = x_positions + (idx - (num_models - 1) / 2) * bar_width
        heights = pivot[model].to_numpy()
        
        # Clip all bars at 1.0
        display_heights = np.clip(heights, 0, 1.0)

        bars = ax.bar(
            offsets,
            display_heights,
            width=bar_width,
            label=model,
            **_prepare_bar_style(idx),
        )

        # Add a lighter highlight strip at the top of each bar for sheen
        highlight_height = display_heights * 0.15
        ax.bar(
            offsets,
            highlight_height,
            width=bar_width,
            bottom=display_heights - highlight_height,
            color="#ffffff",
            alpha=0.15,
            edgecolor="none",
        )

        # Labels: place at top of (clipped) bar, show true value
        for bar, true_value in zip(bars, heights):
            bar_top = bar.get_height()
            label_y = bar_top + label_padding
            ax.annotate(
                value_formatter(true_value),
                xy=(bar.get_x() + bar.get_width() / 2, label_y),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90,
                color="#000000",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(prompt_categories, rotation=20, ha="right")
    _format_labels(ax, title, ylabel)
    
    # Increased space above: use larger multiplier
    ax.set_ylim(0, 1.0 + label_padding * 8)
    
    # Set explicit y-ticks to only show 0.0 to 1.0
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    
    if show_reference_line:
        # Optional: reference line at 1.0
        ax.axhline(y=1.0, color='#ff6b6b', linestyle='--', linewidth=1.2, 
                alpha=0.6, zorder=0)

    handles = _build_custom_legend_handles(model_names)
    _add_bottom_legend(fig, handles, "Model")

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())

    if show_plot:
        plt.show()
    plt.close(fig)


# Update make_grouped_bar_chart to ONLY use single axis (no broken axis):
def make_grouped_bar_chart(
    data: pd.DataFrame,
    metric: str,
    output_path: Path | str,
    ylabel: str,
    title: str,
    value_formatter,
    broken_axis: bool = False,  # Keep parameter but ignore it
    show_plot: bool = False,
) -> None:
    """Generate a grouped bar chart with bars clipped at 1.0."""
    if data.empty:
        raise ValueError("No rows available to plot.")

    agg = (
        data.groupby(["prompt_category", "model"], as_index=False)[metric]
        .mean()
        .sort_values(["prompt_category", "model"])
    )

    pivot = agg.pivot(index="prompt_category", columns="model", values=metric).fillna(0.0)

    prompt_categories = pivot.index.tolist()
    model_names = pivot.columns.tolist()

    show_reference_line = "speedup" in metric.lower() or "efficiency" in metric.lower()
    # Always use single axis with clipping
    _plot_single_axis(
        prompt_categories,
        model_names,
        pivot,
        output_path,
        ylabel,
        title,
        value_formatter,
        show_plot,
        show_reference_line=show_reference_line
    )

def _plot_with_broken_axis(
    prompt_categories,
    model_names,
    pivot,
    output_path,
    ylabel,
    title,
    value_formatter,
    break_params,
    show_plot,
) -> None:
    break_threshold, lower_ylim, upper_ylim_low, upper_ylim_high = break_params

    x_positions = np.arange(len(prompt_categories))
    num_models = len(model_names)
    bar_width = 0.8 / max(num_models, 1)

    visual_floor = max(lower_ylim * 0.02, 0.02)
    lower_label_padding = lower_ylim * 0.06
    upper_label_padding = max((upper_ylim_high - upper_ylim_low) * 0.05, 0.1)

    fig, (ax_upper, ax_lower) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(max(10, len(prompt_categories) * 1.6), 5.6),
        constrained_layout=False,
    )
    fig.subplots_adjust(hspace=0.07)
    fig.patch.set_facecolor(BASE_BG)
    ax_upper.set_facecolor(BASE_BG)
    ax_lower.set_facecolor(BASE_BG)
    # fig.suptitle(
    #     title,
    #     y=0.99,
    #     fontsize=18,
    #     color=TITLE_COLOR,
    #     fontweight="semibold",
    # )

    for ax in (ax_upper, ax_lower):
        _style_axes(ax)

    ax_upper.spines["bottom"].set_visible(False)
    ax_lower.spines["top"].set_visible(False)
    ax_upper.tick_params(labelbottom=False)
    ax_lower.xaxis.tick_bottom()

    legend_handles = []

    for idx, model in enumerate(model_names):
        style_kwargs = _prepare_bar_style(idx)
        offsets = x_positions + (idx - (num_models - 1) / 2) * bar_width
        heights = pivot[model].to_numpy()
        display_heights = np.where(heights > 0, heights, visual_floor)

        bars_low = ax_lower.bar(
            offsets,
            display_heights,
            width=bar_width,
            label=model if idx == 0 else "_nolegend_",
            **style_kwargs,
        )
        ax_lower.bar(
            offsets,
            display_heights * 0.15,
            width=bar_width,
            bottom=display_heights * 0.85,
            color="#ffffff",
            alpha=0.12,
            edgecolor="none",
        )

        high_mask = heights > break_threshold
        if high_mask.any():
            ax_upper.bar(
                offsets[high_mask],
                display_heights[high_mask],
                width=bar_width,
                label="_nolegend_",
                **style_kwargs,
            )
            ax_upper.bar(
                offsets[high_mask],
                display_heights[high_mask] * 0.15,
                width=bar_width,
                bottom=display_heights[high_mask] * 0.85,
                color="#ffffff",
                alpha=0.12,
                edgecolor="none",
            )

        for bar, true_value in zip(bars_low, heights):
            if true_value <= break_threshold:
                base_height = bar.get_height()
                label_y = base_height + lower_label_padding
                ax_lower.annotate(
                    value_formatter(true_value),
                    xy=(bar.get_x() + bar.get_width() / 2, label_y),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=90,
                    color="#000000",
                )

        for offset, true_value in zip(offsets[high_mask], heights[high_mask]):
            label_y = true_value + upper_label_padding
            ax_upper.annotate(
                value_formatter(true_value),
                xy=(offset, label_y),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90,
                color="#000000",
            )

        legend_handles.append(
            Patch(
                facecolor=mcolors.to_rgba(MODEL_COLORS[idx % len(MODEL_COLORS)], alpha=0.7),
                edgecolor=MODEL_OUTLINES[idx % len(MODEL_OUTLINES)],
                linewidth=1.8,
                hatch=MODEL_HATCHES[idx % len(MODEL_HATCHES)],
                label=model,
            )
        )
        
    extra_space_factor = 2.5
    # ax_lower.set_ylim(0, lower_ylim)
    # ax_upper.set_ylim(upper_ylim_low, upper_ylim_high)
    ax_lower.set_ylim(0, lower_ylim + lower_label_padding * 8)

    upper_limit = upper_ylim_high * extra_space_factor
    ax_upper.set_ylim(upper_ylim_low, upper_limit)

    # optionally, recalc the annotation padding so labels still sit nicely
    upper_label_padding = max((upper_limit - upper_ylim_low) * 0.05, 0.15)

    ax_lower.set_xticks(x_positions)
    ax_lower.set_xticklabels(prompt_categories, rotation=20, ha="right")
    ax_lower.set_ylabel(ylabel, color=LABEL_COLOR, fontsize=13)
    ax_upper.tick_params(labelcolor=TICK_COLOR)
    ax_lower.tick_params(labelcolor=TICK_COLOR)

    _add_bottom_legend(fig, legend_handles, "Model")

    d = 0.007
    kwargs = dict(transform=ax_upper.transAxes, color="#f3f3f3", clip_on=False, linewidth=1.1)
    ax_upper.plot((-d, +d), (-d, +d), **kwargs)
    ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_lower.transAxes)
    ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())

    if show_plot:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def main() -> None:
    #FULL AND HOPEFULLY ALL CORRECT DATA
    csv_files = [
     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/futures_promises/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/geometry/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/histogram/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/la/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/scan/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/search/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/sort/driver/tcmalloc/data.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/stencil/driver/tcmalloc/data.csv",
     "/work/pi_mrobson_smith_edu/scratch/generation_hpx/transform/driver/tcmalloc/data.csv"
    ]

    metric_columns = [config["column"] for config in METRICS]
    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            print("FILE NOT FOUND")
            raise FileNotFoundError(csv_path)
        frames.append(load_scores(csv_path, metric_columns))

    combined = pd.concat(frames, ignore_index=True)

    for config in METRICS:
        make_grouped_bar_chart(
            combined,
            metric=config["column"],
            output_path=OUTPUT_DIR / config["filename"],
            ylabel=config["ylabel"],
            title=config["title"],
            value_formatter=config["formatter"],
            broken_axis=config["broken_axis"],
            show_plot=False,
        )

    print("All histograms generated.")


if __name__ == "__main__":
    main()