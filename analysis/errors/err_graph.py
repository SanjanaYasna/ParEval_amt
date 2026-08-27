#!/usr/bin/env python3
"""
Aggregate HPX build/runtime error CSVs and generate per-category heatmaps.

Expected directory structure:

    <root>/<prompt_type>/<model>_errors.csv

For example:

    build_errors/
        fft/
            gpt5_errors.csv
            hpc-coder_errors.csv
            magicoder_errors.csv
            oss_errors.csv
            sonnet_errors.csv
        transform/
            gpt5_errors.csv
            hpc-coder_errors.csv
            magicoder_errors.csv
            oss_errors.csv
            sonnet_errors.csv

Example usage:

    python net_error_csv.py \
        /work/pi_mrobson_smith_edu/scratch/generation_hpx/build_errors/ \
        -o error_summary.csv \
        --fig-out error_heatmap.png
"""

import os
import sys
import argparse
import csv
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROMPT_TYPES = [
    "fft",
    "futures_promises",
    "geometry",
    "graph",
    "histogram",
    "dense_la",
    "sparse_la",
    "locking_contention",
    "reduce",
    "scan",
    "search",
    "sort",
    "stencil",
    "transform",
]


# Internal model name -> CSV filename.
#
# This is explicit because gpt-5 is stored as gpt5_errors.csv.
MODEL_FILES = {
    "gpt5": "gpt5_errors.csv",
    "hpc-coder": "hpc-coder_errors.csv",
    "magicoder": "magicoder_errors.csv",
    "oss": "oss_errors.csv",
    "sonnet": "sonnet_errors.csv",
}


MODEL_NAMES = list(MODEL_FILES.keys())


MODEL_DISPLAY_NAMES = {
    "gpt5": "GPT-5",
    "hpc-coder": "HPCCoder 16B",
    "magicoder": "MagiCoder",
    "oss": "GPT OSS 120B",
    "sonnet": "Claude Sonnet",
}


ERROR_CATEGORIES = [
   # "Boost_Error",
    "Syntax_Error",
    "Type_Error",
    "Declaration_Or_Function_Call_Error",
    "Linker_Error",
    "Runtime_Error",
]


# ============================================================================
# ARGUMENTS
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate HPX *_errors.csv files and produce "
            "a heatmap plus summary CSV."
        )
    )

    parser.add_argument(
        "root_dir",
        type=str,
        help=(
            "Root error directory containing one directory "
            "per prompt type."
        ),
    )

    parser.add_argument(
        "-o",
        "--output-csv",
        type=str,
        default="error_summary.csv",
    )

    parser.add_argument(
        "--fig-out",
        type=str,
        default="error_heatmap.png",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    return parser.parse_args()


# ============================================================================
# GATHER ERROR COUNTS
# ============================================================================

def gather_counts(
    root_dir: str
) -> Dict[Tuple[str, str, str], int]:
    """
    Read:

        <root>/<prompt_type>/<model>_errors.csv

    and aggregate counts by:

        (prompt_type, model_name, error_type)
    """

    counts: Dict[
        Tuple[str, str, str],
        int
    ] = defaultdict(int)

    error_set = {
        category.lower(): category
        for category in ERROR_CATEGORIES
    }

    files_found = 0

    for prompt_type in PROMPT_TYPES:

        for model_name, filename in MODEL_FILES.items():

            csv_path = os.path.join(
                root_dir,
                prompt_type,
                filename,
            )

            if not os.path.isfile(csv_path):
                continue

            files_found += 1

            print(
                f"Reading: {csv_path}"
            )

            with open(
                csv_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as f:

                reader = csv.DictReader(f)

                for row in reader:

                    error_type = (
                        row.get(
                            "error_type",
                            "",
                        )
                        .strip()
                        .lower()
                    )

                    if error_type not in error_set:
                        continue

                    canonical_error_type = (
                        error_set[error_type]
                    )

                    counts[
                        (
                            prompt_type,
                            model_name,
                            canonical_error_type,
                        )
                    ] += 1

    print()
    print(
        f"Processed {files_found} error CSV files."
    )

    return counts


# ============================================================================
# BUILD SUMMARY DATAFRAME
# ============================================================================

def build_matrix(
    counts: Dict[Tuple[str, str, str], int]
) -> pd.DataFrame:
    """
    Build one row per:

        prompt_type x model

    with one column per error category.
    """

    records = []

    for prompt_type in PROMPT_TYPES:

        for model_name in MODEL_NAMES:

            row = {
                "prompt_type": prompt_type,
                "model_name": model_name,
            }

            for error_type in ERROR_CATEGORIES:

                row[error_type] = counts.get(
                    (
                        prompt_type,
                        model_name,
                        error_type,
                    ),
                    0,
                )

            records.append(row)

    return pd.DataFrame(records)


# ============================================================================
# HEATMAP
# ============================================================================

def plot_heatmap(
    df: pd.DataFrame,
    fig_out: str,
    dpi: int,
):
    """
    Generate one heatmap panel per error category.

    Rows:
        prompt types

    Columns:
        models

    Cells:
        number of extracted error instances
    """

    n_categories = len(
        ERROR_CATEGORIES
    )

    ncols = 2

    nrows = (
        n_categories
        + ncols
        - 1
    ) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            ncols * 6.0,
            nrows * 5.5,
        ),
        gridspec_kw={
            "hspace": 0.65,
            "wspace": 0.45,
        },
    )

    axes = np.atleast_2d(
        axes
    )

    # ------------------------------------------------------------------
    # Global colour range
    # ------------------------------------------------------------------

    all_values = [
        value
        for error_type in ERROR_CATEGORIES
        for value in df[error_type]
    ]

    global_max = max(
        max(all_values),
        1,
    )

    cmap = (
        mcolors
        .LinearSegmentedColormap
        .from_list(
            "white_red",
            [
                "#fff5f0",
                "#fee0d2",
                "#fcbba1",
                "#fc9272",
                "#fb6a4a",
                "#ef3b2c",
                "#cb181d",
                "#a50f15",
                "#67000d",
            ],
        )
    )

    # Makes lower counts remain visually distinguishable.
    norm = mcolors.PowerNorm(
        gamma=0.45,
        vmin=0,
        vmax=global_max,
    )

    # ------------------------------------------------------------------
    # One panel per error category
    # ------------------------------------------------------------------

    for idx, error_type in enumerate(
        ERROR_CATEGORIES
    ):

        row_idx, col_idx = divmod(
            idx,
            ncols,
        )

        ax = axes[
            row_idx
        ][
            col_idx
        ]

        pivot = (
            df.pivot(
                index="prompt_type",
                columns="model_name",
                values=error_type,
            )
            .reindex(
                index=PROMPT_TYPES,
                columns=MODEL_NAMES,
            )
            .fillna(0)
            .astype(int)
        )

        data = pivot.values

        n_rows_data, n_cols_data = (
            data.shape
        )

        # --------------------------------------------------------------
        # Heatmap
        # --------------------------------------------------------------

        ax.imshow(
            data,
            cmap=cmap,
            norm=norm,
            aspect="auto",
        )

        # White grid lines between cells.
        ax.set_xticks(
            np.arange(
                -0.5,
                n_cols_data,
                1,
            ),
            minor=True,
        )

        ax.set_yticks(
            np.arange(
                -0.5,
                n_rows_data,
                1,
            ),
            minor=True,
        )

        ax.grid(
            which="minor",
            color="white",
            linewidth=2.5,
        )

        ax.tick_params(
            which="minor",
            length=0,
        )

        # --------------------------------------------------------------
        # Cell annotations
        # --------------------------------------------------------------

        for i in range(
            n_rows_data
        ):
            for j in range(
                n_cols_data
            ):

                value = data[i, j]

                rgba = cmap(
                    norm(value)
                )

                brightness = (
                    0.299 * rgba[0]
                    + 0.587 * rgba[1]
                    + 0.114 * rgba[2]
                )

                label = str(
                    value
                )

                fontsize = (
                    9
                    if len(label) <= 2
                    else 7
                )

                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight="bold",
                    color=(
                        "white"
                        if brightness < 0.52
                        else "black"
                    ),
                )

        # --------------------------------------------------------------
        # Axes styling
        # --------------------------------------------------------------

        for spine in (
            ax.spines.values()
        ):
            spine.set_visible(
                False
            )

        ax.tick_params(
            axis="both",
            which="major",
            length=0,
        )

        # --------------------------------------------------------------
        # Panel title
        # --------------------------------------------------------------

        title = error_type.replace(
            "_",
            " ",
        )

        if (
            error_type
            == "Declaration_Or_Function_Call_Error"
        ):
            title = (
                "Declaration Or\n"
                "Function Call Error"
            )

        ax.set_title(
            title,
            fontsize=10,
            fontweight="bold",
            pad=8,
        )

        # --------------------------------------------------------------
        # Y axis: prompt types
        # --------------------------------------------------------------

        ax.set_yticks(
            range(
                len(PROMPT_TYPES)
            )
        )

        ax.set_yticklabels(
            PROMPT_TYPES,
            fontsize=8,
        )

        # --------------------------------------------------------------
        # X axis: models
        # --------------------------------------------------------------

        ax.set_xticks(
            range(
                len(MODEL_NAMES)
            )
        )

        ax.set_xticklabels(
            [
                MODEL_DISPLAY_NAMES.get(
                    model,
                    model,
                )
                for model in MODEL_NAMES
            ],
            fontsize=7.5,
            rotation=45,
            ha="right",
        )

    # ------------------------------------------------------------------
    # Hide unused subplot slots
    # ------------------------------------------------------------------

    for idx in range(
        n_categories,
        nrows * ncols,
    ):
        row_idx, col_idx = divmod(
            idx,
            ncols,
        )

        axes[
            row_idx
        ][
            col_idx
        ].set_visible(
            False
        )

    # ------------------------------------------------------------------
    # Colour bar
    # ------------------------------------------------------------------

    sm = plt.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        shrink=0.75,
        pad=0.03,
    )

    cbar.set_label(
        "Error count",
        fontsize=10,
        fontweight="bold",
    )

    cbar.ax.tick_params(
        labelsize=8
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    fig.savefig(
        fig_out,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    print(
        f"Figure saved to {fig_out}"
    )

    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():

    args = get_args()

    root_dir = os.path.abspath(
        args.root_dir
    )

    if not os.path.isdir(
        root_dir
    ):
        print(
            f"ERROR: {root_dir} "
            "is not a directory.",
            file=sys.stderr,
        )

        sys.exit(1)

    counts = gather_counts(
        root_dir
    )

    if not counts:
        print(
            "No recognized errors found.",
            file=sys.stderr,
        )

        print(
            "Expected layout:",
            file=sys.stderr,
        )

        print(
            "  <root>/<prompt_type>/<model>_errors.csv",
            file=sys.stderr,
        )

        sys.exit(1)

    df = build_matrix(
        counts
    )

    df.to_csv(
        args.output_csv,
        index=False,
    )

    print(
        f"Summary CSV saved to "
        f"{args.output_csv}"
    )

    print()

    print(
        df.to_string(
            index=False
        )
    )

    plot_heatmap(
        df,
        args.fig_out,
        args.dpi,
    )


if __name__ == "__main__":
    main()