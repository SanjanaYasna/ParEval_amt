import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
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
    "/work/pi_mrobson_smith_edu/scratch/generation_hpx/futures_promises/driver/tcmalloc/json_runtime_summary.csv",
]

THREADS = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
RUNTIME_COLS = [f"runtime_{t}" for t in THREADS.astype(int)]
OUTPUT_ROOT = "/work/pi_mrobson_smith_edu/ParEval_amt/analysis/visuals_specific_strict_pass"

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
    "gpt5_combined":         {"label": "GPT5",       "color": PASTEL_COLORS[0]},
    "hpc-coder_combined":    {"label": "HPC-Coder",  "color": PASTEL_COLORS[1]},
    "starcoder_combined":    {"label": "Starcoder",  "color": PASTEL_COLORS[4]},
    "magicoder_combined":    {"label": "Magicoder",  "color": PASTEL_COLORS[2]},
    "sonnet_combined":       {"label": "Sonnet",     "color": PASTEL_COLORS[3]},
    "glm_combined":          {"label": "GLM",        "color": PASTEL_COLORS[5]},
    "oss_combined":          {"label": "OSS",        "color": PASTEL_COLORS[6]},
    "minimax_combined":      {"label": "MiniMax",    "color": PASTEL_COLORS[7]},
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

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def safe_filename(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip()
    name = re.sub(r"[^\w.\- ]+", "", name)
    return name.replace(" ", "")

def get_prompt_category(csv_path: str) -> str:
    parts = csv_path.split(os.sep)
    try:
        idx = parts.index("generation_hpx")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"

def find_model_json_files(csv_path: str) -> dict[str, str]:
    """
    Given a CSV path, find all model-specific JSON files in the same directory.
    Returns a dict mapping model_key -> json_path
    """
    csv_dir = os.path.dirname(csv_path)
    model_jsons = {}
    
    for model_key in MODEL_STYLES.keys():
        json_filename = f"{model_key}.json"
        json_path = os.path.join(csv_dir, json_filename)
        if os.path.isfile(json_path):
            model_jsons[model_key] = json_path
    
    return model_jsons

def build_validity_index_for_model(json_path: str, model_key: str, available_threads: np.ndarray) -> tuple[dict, dict]:
    """
    Returns
    -------
    validity : dict
        problem_name -> {num_threads (float) -> bool}
        True only when is_source_valid AND is_valid are both true.

    valid_output_counts : dict
        problem_name -> int
        Number of outputs where:
        - is_source_valid is True, AND
        - at least one run has is_valid=True with non-null runtime
    """
    print(f"DEBUG: Loading {model_key} from {json_path}", flush=True)
    
    with open(json_path, "r") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        problems = data.get("results", data.get("problems", list(data.values())))
    else:
        problems = data

    validity: dict[str, dict[float, bool]] = {}
    valid_output_counts: dict[str, int] = {}

    for problem in problems:
        name = problem.get("name", "")
        outputs = problem.get("outputs", [])

        if name not in validity:
            validity[name] = {}
            valid_output_counts[name] = 0

        for output in outputs:
            source_valid = output.get("is_source_valid", False)
            are_all_valid = output.get("are_all_valid", False)

            # # Count this output only if it has source_valid AND valid runtimes for ALL threads (are_all_valid True)
            if source_valid and are_all_valid:
                valid_output_counts[name] += 1
            if not source_valid:
                # Mark all threads as invalid for this output
                runs = output.get("runs") or []
                for run in runs:
                    t = float(run.get("num_threads", -1))
                    if t < 0:
                        continue
                    validity[name].setdefault(t, False)
                continue
            
            runs = output.get("runs") or []
            for run in runs:
                t = float(run.get("num_threads", -1))
                if t < 0:
                    continue
                
                run_valid = run.get("is_valid", False)
                runtime = run.get("runtime", None)
                
                # Check if this run is valid AND has a non-null runtime
                if run_valid and runtime is not None:
                    validity[name][t] = validity[name].get(t, False) or True
                else:
                    # Keep existing value or set to False if not present
                    validity[name].setdefault(t, False)

    print(f"  {model_key}: Found {len(problems)} problems")
    for pname in list(validity.keys())[:3]:
        print(f"    {pname}: valid_output_count={valid_output_counts[pname]}, "
              f"thread_validity={validity[pname]}")
    print(flush=True)

    return validity, valid_output_counts

# ---------------------------------------------------------------------------
# main plotting
# ---------------------------------------------------------------------------
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

    # Find all model-specific JSON files
    model_json_files = find_model_json_files(csv_path)
    
    if not model_json_files:
        print(f"[warn] No model JSON files found for {csv_path}; skipping.")
        continue
    
    print(f"[info] Found {len(model_json_files)} model JSON files for {csv_path}")

    # Load validity information for each model
    model_validity = {}
    model_valid_counts = {}
    
    for model_key, json_path in model_json_files.items():
        validity, valid_counts = build_validity_index_for_model(json_path, model_key, available_threads)
        model_validity[model_key] = validity
        model_valid_counts[model_key] = valid_counts

    prompt_category = get_prompt_category(csv_path)
    category_dir = os.path.join(OUTPUT_ROOT, prompt_category)
    os.makedirs(category_dir, exist_ok=True)

    for prompt_name, group in df.groupby("name", sort=False):
        fig, ax = plt.subplots(figsize=(7.5, 4.75))
        max_runtime = 0.0
        plotted_any = False

        end_annotations: list[tuple[float, float, str, int]] = []

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

            # Apply validity mask for this specific model
            if model_key in model_validity:
                problem_validity = model_validity[model_key].get(prompt_name, {})
                for i, t in enumerate(available_threads):
                    if not problem_validity.get(t, False):
                        runtimes[i] = np.nan

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

            # Record where to place the count annotation
            rightmost_idx = np.where(mask)[0][-1]
            x_end = available_threads[rightmost_idx]
            y_end = runtimes[rightmost_idx]

            # Get count of outputs with source_valid=True AND valid runtimes
            count = model_valid_counts.get(model_key, {}).get(prompt_name, 0)
            end_annotations.append((x_end, y_end, color, count))

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

        # Legend (ordered)
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = {label: handle for handle, label in zip(handles, labels)}
        ordered_labels = [l for l in MODEL_ORDER if l in label_to_handle]
        ordered_handles = [label_to_handle[l] for l in ordered_labels]
        if ordered_handles:
            ax.legend(ordered_handles, ordered_labels, frameon=False)
        else:
            ax.legend(frameon=False)

        # Annotate valid output counts
        fig.tight_layout()
        fig.canvas.draw()

        end_annotations.sort(key=lambda a: a[1], reverse=True)

        MIN_Y_GAP_FRAC = 0.04
        y_min, y_max = ax.get_ylim()
        min_gap = (y_max - y_min) * MIN_Y_GAP_FRAC

        placed_y: list[float] = []

        for x_end, y_end, color, count in end_annotations:
            y_label = y_end
            for py in placed_y:
                if abs(y_label - py) < min_gap:
                    y_label = py - min_gap

            placed_y.append(y_label)

            ax.annotate(
                f"n={count}",
                xy=(x_end, y_end),
                xytext=(x_end * 1.08, y_label),
                fontsize=8,
                color=color,
                va="center",
                ha="left",
                annotation_clip=False,
            )

        file_safe_prompt = safe_filename(prompt_name)
        output_path = os.path.join(
            category_dir, f"{file_safe_prompt}_runtime_vs_threads.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"[ok] Saved {prompt_category} → {prompt_name}: {output_path}")