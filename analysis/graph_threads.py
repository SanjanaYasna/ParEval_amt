import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# configuration                                                               #
# --------------------------------------------------------------------------- #
CSV_PATHS = [
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/fft/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/graph/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/transform/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/scan/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/search/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/sort/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/stencil/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/locking_contention/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/la/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/reduce/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/histogram/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/geometry/driver/tcmalloc/json_runtime_summary.csv",
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/futures_promises/driver/tcmalloc/json_runtime_summary.csv"
]

THREADS = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
RUNTIME_COLS = [f"runtime_{t}" for t in THREADS.astype(int)]
OUTPUT_ROOT = "/work/pi_mrobson_smith_edu/ParEval_amt/analysis/visuals_specific"

PASTEL_COLORS = [
    "#6BAED6",  # blue
    "#FD8D3C",  # orange
    "#74C476",  # green
    "#E377C2",  # pink
    "#9E9AC8",  # purple
    "#FFC658",  # yellow
    "#8DD3C7",  # teal
    "#BC80BD",  # lavender
]

MODEL_STYLES = {
    "gpt5_combined": {"label": "GPT5", "color": PASTEL_COLORS[0]},
    "hpc-coder_combined": {"label": "HPC-Coder", "color": PASTEL_COLORS[1]},
    "starcoder_combined": {"label": "Starcoder", "color": PASTEL_COLORS[4]},
    "magicoder_combined": {"label": "Magicoder", "color": PASTEL_COLORS[2]},
    "sonnet_combined": {"label": "Sonnet", "color": PASTEL_COLORS[3]},
    "glm_combined": {"label": "GLM", "color": PASTEL_COLORS[5]},
    "oss_combined": {"label": "OSS", "color": PASTEL_COLORS[6]},
    "minimax_combined": {"label": "MiniMax", "color": PASTEL_COLORS[7]},
}
MODEL_ORDER = ["GPT5", "HPC-Coder", "Starcoder", "Magicoder", "Sonnet", "GLM", "OSS", "MiniMax"]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def safe_filename(name: str) -> str:
    """Turn an arbitrary string into something safe for filenames."""
    name = name.strip()
    name = re.sub(r"[^\w\-\. ]+", "_", name)
    return name.replace(" ", "_")


def get_prompt_category(csv_path: str) -> str:
    parts = csv_path.split(os.sep)
    try:
        idx = parts.index("generation_hpx")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


# --------------------------------------------------------------------------- #
# main plotting                                                               #
# --------------------------------------------------------------------------- #
for csv_path in CSV_PATHS:
    if not os.path.isfile(csv_path):
        print(f"[skip] Missing CSV: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    if "name" not in df.columns:
        print(f"[warn] 'name' column missing in {csv_path}; skipping.")
        continue

    available_cols = [col for col in RUNTIME_COLS if col in df.columns]
    available_threads = THREADS[[RUNTIME_COLS.index(col) for col in available_cols]]

    if not available_cols:
        print(f"[warn] No runtime_* columns found in {csv_path}; skipping.")
        continue

    prompt_category = get_prompt_category(csv_path)
    category_dir = os.path.join(OUTPUT_ROOT, prompt_category)
    os.makedirs(category_dir, exist_ok=True)

    for prompt_name, group in df.groupby("name", sort=False):
        fig, ax = plt.subplots(figsize=(7.5, 4.75))
        max_runtime = 0.0
        plotted_any = False
        for _, row in group.iterrows():
            model_key = row.get("model", "(unknown model)")
            style = MODEL_STYLES.get(model_key, None)

            if style is None:
                display_label = model_key
                color = "#333333"
            else:
                display_label = style["label"]
                color = style["color"]

            runtimes = row[available_cols].to_numpy(dtype=float, copy=True)
            mask = np.isfinite(runtimes)
            if mask.sum() == 0:
                continue
            max_runtime = max(max_runtime, np.max(runtimes[mask]))
            ax.plot(
                available_threads[mask],
                runtimes[mask],
                marker="o",
                linewidth=2,
                markersize=6,
                color=color,
                label=display_label,
            )
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue
        ax.set_ylim(0, max_runtime * 1.2)

        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("Runtime (s)")

        prompt_category_title = prompt_category.replace("_", " ").title()
        if prompt_category and prompt_category.lower() in prompt_name.lower():
            pattern = re.compile(re.escape(prompt_category), flags=re.IGNORECASE)
            prompt_title = pattern.sub(prompt_category_title, prompt_name)
        else:
            prompt_title = prompt_name
        ax.set_title(prompt_title)

        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = {}
        for handle, label in zip(handles, labels):
            label_to_handle[label] = handle
        ordered_labels = [label for label in MODEL_ORDER if label in label_to_handle]
        ordered_handles = [label_to_handle[label] for label in ordered_labels]
        if ordered_handles:
            ax.legend(ordered_handles, ordered_labels, frameon=False)
        else:
            ax.legend(frameon=False)

        fig.tight_layout()

        file_safe_prompt = safe_filename(prompt_name)
        output_path = os.path.join(category_dir, f"{file_safe_prompt}_runtime_vs_threads.png")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        print(f"[ok] Saved {prompt_category} → {prompt_name}: {output_path}")