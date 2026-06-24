# ai generated slop and super tired so double check later 
#for a given model and category, what is the rate in which code winds up being executed with one or more null runtimes 

import os
import json
import numpy as np
import csv
from collections import defaultdict

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

MODEL_KEYS = [
    "gpt5_combined",
    "hpc-coder_combined",
    "starcoder_combined",
    "magicoder_combined",
    "sonnet_combined",
    "glm_combined",
    "oss_combined",
    "minimax_combined",
]

MODEL_LABELS = {
    "gpt5_combined":      "GPT5",
    "hpc-coder_combined": "HPC-Coder",
    "starcoder_combined": "Starcoder",
    "magicoder_combined": "Magicoder",
    "sonnet_combined":    "Sonnet",
    "glm_combined":       "GLM",
    "oss_combined":       "OSS",
    "minimax_combined":   "MiniMax",
}

THREADS = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)

CATEGORY_CSV = "category_null_runtime_rates.csv"
MODEL_CATEGORY_CSV = "model_category_null_runtime_rates.csv"
GLOBAL_MODEL_CSV = "global_model_null_runtime_rates.csv"

# ---------------------------------------------------------------------------
# helpers (UNCHANGED)
# ---------------------------------------------------------------------------

def get_prompt_category(csv_path: str) -> str:
    parts = csv_path.split(os.sep)
    try:
        idx = parts.index("generation_hpx")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def get_json_path(csv_path: str, model_key: str) -> str:
    return os.path.join(os.path.dirname(csv_path), f"{model_key}.json")


def load_problems(json_path: str):
    with open(json_path, "r") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return data.get("results", data.get("problems", list(data.values())))
    return data


def count_null_runtime_outputs(json_path: str) -> tuple[int, int]:

    problems = load_problems(json_path)

    total_eligible = 0
    outputs_with_null = 0

    for problem in problems:
        for output in problem.get("outputs", []):

            if not output.get("did_any_run", False):
                continue

            total_eligible += 1
            runs = output.get("runs") or []

            has_null = any(run.get("runtime") is None for run in runs)

            if not has_null:
                thread_counts_seen = {float(r.get("num_threads", -1)) for r in runs}
                missing_threads = THREADS[~np.isin(THREADS, list(thread_counts_seen))]
                if missing_threads.size > 0:
                    has_null = True

            if has_null:
                outputs_with_null += 1

    return outputs_with_null, total_eligible


def pct_value(num, den):
    if den == 0:
        return None
    return 100.0 * num / den


def pct(num, den):
    if num is None or den is None or den == 0:
        return "   N/A  "
    return f"{100.0 * num / den:6.2f}%"


# ---------------------------------------------------------------------------
# accumulation (UNCHANGED)
# ---------------------------------------------------------------------------
category_model_stats = defaultdict(lambda: {mk: [0, 0] for mk in MODEL_KEYS})
global_model_stats = {mk: [0, 0] for mk in MODEL_KEYS}
category_overall_stats = defaultdict(lambda: [0, 0])

grand_null = 0
grand_total = 0
missing = defaultdict(set)

for csv_path in CSV_PATHS:
    category = get_prompt_category(csv_path)

    for model_key in MODEL_KEYS:
        json_path = get_json_path(csv_path, model_key)

        if not os.path.isfile(json_path):
            missing[category].add(model_key)
            continue

        with_null, total = count_null_runtime_outputs(json_path)

        category_model_stats[category][model_key][0] += with_null
        category_model_stats[category][model_key][1] += total

        global_model_stats[model_key][0] += with_null
        global_model_stats[model_key][1] += total

        category_overall_stats[category][0] += with_null
        category_overall_stats[category][1] += total

        grand_null += with_null
        grand_total += total


# ---------------------------------------------------------------------------
# WRITE CSV FILES (NEW SECTION)
# ---------------------------------------------------------------------------

# 1️⃣ Category-level CSV
with open(CATEGORY_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "null_rate_percent"])
    for category in sorted(category_overall_stats.keys()):
        nulls, total = category_overall_stats[category]
        writer.writerow([category, pct_value(nulls, total)])

# 2️⃣ Per-model-per-category CSV
with open(MODEL_CATEGORY_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "model", "null_rate_percent"])
    for category in sorted(category_model_stats.keys()):
        for model_key in MODEL_KEYS:
            if model_key in missing[category]:
                writer.writerow([category, model_key, None])
            else:
                nulls, total = category_model_stats[category][model_key]
                writer.writerow([category, model_key, pct_value(nulls, total)])

# 3️⃣ Global per-model CSV
with open(GLOBAL_MODEL_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "null_rate_percent"])
    for model_key in MODEL_KEYS:
        nulls, total = global_model_stats[model_key]
        writer.writerow([model_key, pct_value(nulls, total)])

# ---------------------------------------------------------------------------
# existing reporting (UNCHANGED)
# ---------------------------------------------------------------------------
SEP_MAJOR = "=" * 58
SEP_MINOR = "-" * 58
col_w = max(len(MODEL_LABELS[mk]) for mk in MODEL_KEYS) + 2

print(SEP_MAJOR)
print("NULL-RUNTIME RATE REPORT")
print("(did_all_run=True outputs with ≥1 null/missing runtime)")
print(SEP_MAJOR)

for category in sorted(category_model_stats.keys()):
    cat_null, cat_total = category_overall_stats[category]

    print(f"\nCategory: {category.upper()}")
    print(SEP_MINOR)
    print(f"  {'Model':<{col_w}}  {'Null Rate':>10}")
    print("  " + "-" * (col_w + 13))

    for model_key in MODEL_KEYS:
        label = MODEL_LABELS[model_key]

        if model_key in missing[category]:
            print(f"  {label:<{col_w}}  {'(no file)':>10}")
            continue

        with_null, total = category_model_stats[category][model_key]
        print(f"  {label:<{col_w}}  {pct(with_null, total):>10}")

    print("  " + "-" * (col_w + 13))
    print(f"  {'[Category Total]':<{col_w}}  {pct(cat_null, cat_total):>10}")

print(f"\n{SEP_MAJOR}")
print("GLOBAL SUMMARY — All Categories Combined")
print(SEP_MAJOR)
print(f"  {'Model':<{col_w}}  {'Null Rate':>10}")
print("  " + "-" * (col_w + 13))

for model_key in MODEL_KEYS:
    label = MODEL_LABELS[model_key]
    with_null, total = global_model_stats[model_key]
    print(f"  {label:<{col_w}}  {pct(with_null, total):>10}")

print("  " + "-" * (col_w + 13))
print(f"  {'[Grand Total]':<{col_w}}  {pct(grand_null, grand_total):>10}")
print(SEP_MAJOR)

print("\nCSV files written:")
print(" -", CATEGORY_CSV)
print(" -", MODEL_CATEGORY_CSV)
print(" -", GLOBAL_MODEL_CSV)


# ---------------------------------------------------------------------------
# WRITE LaTeX TABLE (GPT-5, Sonnet, OSS only)
# Sorted by descending combined null rate across the three models
# ---------------------------------------------------------------------------

selected_models = [
    "gpt5_combined",
    "sonnet_combined",
    "oss_combined",
]

latex_rows = []

for category in category_model_stats.keys():
    rates = []
    model_rates = {}

    for model_key in selected_models:
        nulls, total = category_model_stats[category][model_key]
        rate = pct_value(nulls, total)
        model_rates[model_key] = rate

        if rate is not None:
            rates.append(rate)

    if len(rates) == 0:
        combined_rate = -1  # push empty categories to bottom
    else:
        combined_rate = sum(rates) / len(rates)

    latex_rows.append((category, combined_rate, model_rates))

# Sort descending by combined null rate
latex_rows.sort(key=lambda x: x[1], reverse=True)

with open("desc_null_rate.tex", "w") as f:
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\toprule\n")
    f.write("Category & GPT-5 & Sonnet & OSS \\\\\n")
    f.write("\\midrule\n")

    for category, _, model_rates in latex_rows:
        gpt5 = model_rates["gpt5_combined"]
        sonnet = model_rates["sonnet_combined"]
        oss = model_rates["oss_combined"]

        def fmt(x):
            return "-" if x is None else f"{x:.1f}"

        f.write(
            f"{category} & "
            f"{fmt(gpt5)} & "
            f"{fmt(sonnet)} & "
            f"{fmt(oss)} \\\\\n"
        )

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print(" -", "desc_null_rate.tex")