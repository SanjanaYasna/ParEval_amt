from __future__ import annotations
import os
import matplotlib.pyplot as plt 
import pandas as pd
import csv
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
# Styling setup
# --------------------------------------------------------------------------- #
plt.style.use("default") 
BASE_BG   = "#ffffff"
TICK_COLOR = "#1a1a1a"
GRID_COLOR = "#d0d0d0"
TITLE_COLOR = "#111111"
LABEL_COLOR = "#222222"
MIN_VISUAL_WIDTH = 0.02


MODEL_DISPLAY_NAMES = {
    "gpt5": "GPT-5",
    "glm": "GLM 4.7 Flash",
    "hpccoder": "HPCCoder 6.7B",
    "magicoder": "MagiCoder",
    "oss": "GPT OSS 120B",
    "minimax": "MiniMax M2.5",
    "sonnet": "Claude Sonnet 4.5",
}


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

OUTPUT_DIR = Path("/work/pi_mrobson_smith_edu/ParEval_amt/analysis/visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    {
        "column": "strict_pass@1",
        "xlabel": "strict_pass@1",
        "title": "strict_pass@1 by Prompt Category and Model",
        "filename": "strict_pass1_horizontal.png",
        "formatter": lambda v: f"{v:.2f}",
        "broken_axis": False,
    },
    {
        "column": "speedup@1",
        "xlabel": "speedup@1",
        "title": "speedup@1 by Prompt Category and Model",
        "filename": "speedup1_horizontal.png",
        "formatter": lambda v: f"{v:.2f}",
        "broken_axis": True,
    },
    {
        "column": "efficiency@1",
        "xlabel": "efficiency@1",
        "title": "efficiency@1 by Prompt Category and Model",
        "filename": "efficiency1_horizontal.png",
        "formatter": lambda v: f"{v:.2f}",
        "broken_axis": True,
    },
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
    return cleaned[columns_to_keep]


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
            label=MODEL_DISPLAY_NAMES.get(model, model),
        )
        for idx, model in enumerate(model_names)
    ]
    return handles


def _add_top_legend(fig, handles, title="Model"):
    """Place legend at top of figure."""
    legend = fig.legend(
        handles=handles,
        title=title,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=min(len(handles), 4),
        handlelength=1.6,
        handleheight=1.2,
        frameon=True,
        framealpha=0.95,
        fontsize=10,
        columnspacing=1.0,
    )
    legend.get_title().set_color(LABEL_COLOR)
    legend.get_frame().set_facecolor("#f9f9f9")
    legend.get_frame().set_edgecolor("#b7b7b7")
    legend.get_frame().set_linewidth(1.2)


def _style_axes(ax):
    ax.set_facecolor(BASE_BG)
    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color("#6b6f7e")
        spine.set_linewidth(0.9)


def _format_labels(ax, xlabel):
    ax.set_xlabel(xlabel, color=LABEL_COLOR, fontsize=13)
    ax.tick_params(axis="both", labelsize=11, colors=TICK_COLOR)


def _plot_split_horizontal(
    prompt_categories,
    model_names,
    pivot,
    output_path,
    xlabel,
    title,
    value_formatter,
    show_plot,
    show_reference_line=False
) -> None:
    """Horizontal bar chart split into two side-by-side panels."""
    num_categories = len(prompt_categories)
    split_idx = (num_categories + 1) // 2
    
    categories_left = prompt_categories[:split_idx]
    categories_right = prompt_categories[split_idx:]
    
    pivot_left = pivot.iloc[:split_idx]
    pivot_right = pivot.iloc[split_idx:]
    
    num_models = len(model_names)
    bar_height = 0.75 / max(num_models, 1)
    label_padding = 0.04

    # Wide figure with two columns
    #first num for height, second num for width
    fig_height = max(20, max(len(categories_left), len(categories_right)) * 0.7)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, fig_height), sharey=False, constrained_layout=False
    )
    fig.set_facecolor(BASE_BG)
    fig.subplots_adjust(top=0.94, bottom=0.05, left=0.12, right=0.98, wspace=0.2)
    
    for ax in (ax_left, ax_right):
        _style_axes(ax)

    # LEFT PANEL
    y_positions_left = np.arange(len(categories_left))
    for idx, model in enumerate(model_names):
        offsets = y_positions_left + (idx - (num_models - 1) / 2) * bar_height
        widths = pivot_left[model].to_numpy()
        display_widths = np.clip(widths, MIN_VISUAL_WIDTH, 1.0) #np.clip(widths, 0, 1.0)

        bars = ax_left.barh(
            offsets,
            display_widths,
            height=bar_height,
            **_prepare_bar_style(idx),
        )

        highlight_width = display_widths * 0.15
        ax_left.barh(
            offsets,
            highlight_width,
            height=bar_height,
            left=display_widths - highlight_width,
            color="#ffffff",
            alpha=0.15,
            edgecolor="none",
        )

        for bar, true_value in zip(bars, widths):
            bar_right = bar.get_width()
            label_x = bar_right + label_padding
            ax_left.annotate(
                value_formatter(true_value),
                xy=(label_x, bar.get_y() + bar.get_height() / 2),
                xytext=(0, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                color="#000000",
            )

    ax_left.set_yticks(y_positions_left)
    ax_left.set_yticklabels(categories_left, fontsize=11)
    ax_left.set_xlim(0, 1.0 + label_padding * 3.5)
    ax_left.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _format_labels(ax_left, xlabel)
    
    if show_reference_line:
        ax_left.axvline(x=1.0, color='#ff6b6b', linestyle='--', linewidth=1.2, 
                        alpha=0.6, zorder=0)

    # RIGHT PANEL
    y_positions_right = np.arange(len(categories_right))
    for idx, model in enumerate(model_names):
        offsets = y_positions_right + (idx - (num_models - 1) / 2) * bar_height
        widths = pivot_right[model].to_numpy()
        display_widths = np.clip(widths, MIN_VISUAL_WIDTH, 1.0)

        bars = ax_right.barh(
            offsets,
            display_widths,
            height=bar_height,
            **_prepare_bar_style(idx),
        )

        highlight_width = display_widths * 0.15
        ax_right.barh(
            offsets,
            highlight_width,
            height=bar_height,
            left=display_widths - highlight_width,
            color="#ffffff",
            alpha=0.15,
            edgecolor="none",
        )

        for bar, true_value in zip(bars, widths):
            bar_right = bar.get_width()
            label_x = bar_right + label_padding
            ax_right.annotate(
                value_formatter(true_value),
                xy=(label_x, bar.get_y() + bar.get_height() / 2),
                xytext=(0, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                color="#000000",
            )

    ax_right.set_yticks(y_positions_right)
    ax_right.set_yticklabels(categories_right, fontsize=11)
    ax_right.set_xlim(0, 1.0 + label_padding * 3.5)
    ax_right.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _format_labels(ax_right, xlabel)
    
    if show_reference_line:
        ax_right.axvline(x=1.0, color='#ff6b6b', linestyle='--', linewidth=1.2, 
                         alpha=0.6, zorder=0)

    handles = _build_custom_legend_handles(model_names)
    _add_top_legend(fig, handles, "Model")

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')

    if show_plot:
        plt.show()
    plt.close(fig)


def make_grouped_bar_chart(
    data: pd.DataFrame,
    metric: str,
    output_path: Path | str,
    xlabel: str,
    title: str,
    value_formatter,
    broken_axis: bool = False,
    show_plot: bool = False,
) -> None:
    """Generate a grouped horizontal bar chart split into two panels."""
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
    
    _plot_split_horizontal(
        prompt_categories,
        model_names,
        pivot,
        output_path,
        xlabel,
        title,
        value_formatter,
        show_plot,
        show_reference_line=show_reference_line
    )


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #
def main() -> None:
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
            print(f"FILE NOT FOUND: {csv_path}")
            raise FileNotFoundError(csv_path)
        frames.append(load_scores(csv_path, metric_columns))

    combined = pd.concat(frames, ignore_index=True)

    for config in METRICS:
        make_grouped_bar_chart(
            combined,
            metric=config["column"],
            output_path=OUTPUT_DIR / config["filename"],
            xlabel=config["xlabel"],
            title=config["title"],
            value_formatter=config["formatter"],
            broken_axis=config["broken_axis"],
            show_plot=False,
        )

    print("All histograms generated.")


if __name__ == "__main__":
    main()