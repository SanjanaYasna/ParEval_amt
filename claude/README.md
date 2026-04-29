# Claude Batch API Pipeline for HPX Code Generation

This directory contains the end-to-end pipeline for generating high-performance
C++ HPX code using Anthropic's Claude models (Sonnet 4.5 and Opus 4.5) via
both the **Batch API** and **direct (non-batch) API** queries, then compiling
and evaluating the generated code on the Unity HPC cluster via SLURM.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Directory Overview](#directory-overview)
3. [Pipeline Overview](#pipeline-overview)
4. [Stage 1: Generate Batch Requests](#stage-1-generate-batch-requests)
5. [Stage 2: Expand Batches for pass@k Evaluation](#stage-2-expand-batches-for-passk-evaluation)
6. [Stage 3: Submit Batches to the Anthropic API](#stage-3-submit-batches-to-the-anthropic-api)
7. [Stage 4: Retrieve Batch Results](#stage-4-retrieve-batch-results)
8. [Stage 5: Convert Responses to Evaluation Format](#stage-5-convert-responses-to-evaluation-format)
9. [Stage 6: Clean Generated Code](#stage-6-clean-generated-code)
10. [Stage 7: Split by Problem Type](#stage-7-split-by-problem-type)
11. [Stage 8: Run on Unity via SLURM](#stage-8-run-on-unity-via-slurm)
12. [Direct (Non-Batch) API Queries](#direct-non-batch-api-queries)
13. [Utility Scripts](#utility-scripts)
14. [Configuration Reference](#configuration-reference)

---

## Prerequisites

- **Python 3.10+** with the `requests` library (for `send_batches.py`).
  All other scripts use only the standard library.
- **Anthropic API key** exported as:
  ```bash
  export ANTHROPIC_API_KEY="sk-ant-..."
  ```
- **Prompt JSON files** from the `../prompts/` directory (e.g., `fft.json`,
  `geometry.json`, `futures_promises.json`, etc.). Each file is a JSON array
  of objects with `"name"` and `"prompt"` fields.
- **Unity HPC cluster** access with:
  - `conda` environment `hpc_llm`
  - HPX 1.5.1 (with tcmalloc) and/or HPX 1.10.0 installed
  - The `run-all.py` driver script (located in `../drivers/`)

---

## Directory Overview

### Core Pipeline Scripts

| Script | Stage | Description |
|---|---|---|
| `make_request.py` | 1 | Generate batch request `.sh`/`.json` files for Sonnet and Opus from prompt JSONs |
| `expand_batch.py` | 2 | Replicate each request N times (default 100) for pass@k sampling |
| `send_batches.py` | 3 | Submit expanded batch JSON to Anthropic in rate-limited chunks |
| `compile.py` | 5 | Aggregate batch responses with request metadata into evaluation JSON |
| `make_json_cleaned.py` | 5 | Simpler response-to-JSON converter (single batch `.sh` + outputs JSONL) |
| `clean_claude.py` | 6 | Strip markdown fences/includes, extract first C++ function body |
| `clean_geometry.py` | 6 | Lighter cleaner (remove fences and `#include` lines only) |
| `parse_out.py` | 7 | Split combined JSON into per-problem-type files |

### SLURM Job Scripts

| Script | Purpose |
|---|---|
| `sonnet_run.sh` | Run the full Sonnet evaluation with HPX on a 64-core node |
| `sonnet_batch_run.sh` | Re-run specific problem types (array job) |
| `batch_remaining_1_10.sh` | Run remaining problem types against HPX 1.10.0 (array job) |
| `sparse_la_left.sh` | Run residual scan/sparse_la problems |

### Batch Request Files

| File | Description |
|---|---|
| `claude_sonnet_4_5_batch.json` | Full Sonnet batch request (~70 prompts) |
| `claude_opus_4_5_batch.json` | Full Opus batch request (~70 prompts) |
| `claude_sonnet_4_5_batch.sh` | Sonnet curl command (futures/promises subset) |
| `claude_opus_4_5_batch.sh` | Opus curl command (futures/promises subset) |
| `sonnet_4_5_batch.sh` | Expanded Sonnet batch (x100 copies per prompt) |
| `sonnet_futures_batch.json` | Expanded futures/promises batch (x100) |
| `try.txt` | Minimal test curl snippet for direct API testing |

### Configuration

| File | Description |
|---|---|
| `launch_quick.json` | Defines how to launch compiled programs under each parallelism model (HPX, MPI, OMP, etc.) |

---

## Pipeline Overview

```
prompts/*.json
       |
       v
  make_request.py          [Stage 1] Generate batch .sh/.json for Sonnet & Opus
       |
       v
  expand_batch.py          [Stage 2] Duplicate each request 100x for pass@k
       |
       v
  send_batches.py          [Stage 3] Submit to Anthropic Batch API in chunks
       |
       v
  (poll & download)        [Stage 4] Retrieve results via curl / Anthropic dashboard
       |
       v
  compile.py               [Stage 5] Aggregate responses into evaluation JSON
  make_json_cleaned.py                (alternative simpler converter)
       |
       v
  clean_claude.py          [Stage 6] Strip markdown, extract C++ function bodies
  clean_geometry.py                   (alternative lighter cleaner)
       |
       v
  parse_out.py             [Stage 7] Split into per-problem-type JSON files
       |
       v
  SLURM job scripts        [Stage 8] Compile & run generated code on Unity
  (sonnet_run.sh, etc.)
```

---

## Stage 1: Generate Batch Requests

`make_request.py` reads one or more prompt JSON files and produces ready-to-run
batch request files for **both** `claude-sonnet-4-5` and `claude-opus-4-5`.

```bash
python make_request.py ../prompts/fft.json ../prompts/geometry.json \
    ../prompts/graph.json ../prompts/histogram.json \
    ../prompts/reduce.json ../prompts/scan.json \
    ../prompts/sort.json ../prompts/search.json \
    ../prompts/stencil.json ../prompts/transform.json \
    ../prompts/la.json ../prompts/locking_contention.json \
    ../prompts/futures_promises.json \
    --out-dir . \
    --max-tokens 2048 \
    --thinking-budget 1400
```

**Outputs:**
- `claude_sonnet_4_5_batch.sh` -- curl command wrapping the full Sonnet batch
- `claude_opus_4_5_batch.sh` -- curl command wrapping the full Opus batch

Each request includes:
- **System prompt:** "You are an expert modern C++ developer specializing in
  high-performance HPX code. Provide concise, correct implementations."
- **Thinking:** enabled with a budget of 1400 tokens
- **Sampling:** `max_tokens=2048`, `top_p=0.95`

You can also override the system prompt:

```bash
python make_request.py ../prompts/fft.json \
    --system "Your custom system prompt here."
```

**To produce a standalone JSON file** (without the curl wrapper), extract the
JSON payload from the generated `.sh` file, or modify the script. The existing
`.json` batch files (e.g., `claude_sonnet_4_5_batch.json`) are pre-built
examples of this.

---

## Stage 2: Expand Batches for pass@k Evaluation

For pass@k evaluation (e.g., pass@1, pass@10, pass@100), each prompt must be
sent multiple independent times. `expand_batch.py` duplicates every request
N times, appending `_copy_001` through `_copy_N` to each `custom_id`.

```bash
# Expand to 100 copies per prompt (default)
python expand_batch.py \
    --input claude_sonnet_4_5_batch.json \
    --output sonnet_4_5_expanded.json \
    --copies 100 \
    --top-p 0.95
```

The script can parse both raw `.json` files and `.sh` files (it extracts the
embedded JSON block automatically).

If `--output` is omitted, the output defaults to `<input_stem>_x100.json`.

**Example:** 70 prompts x 100 copies = 7,000 total requests.

---

## Stage 3: Submit Batches to the Anthropic API

`send_batches.py` submits the expanded batch JSON to Anthropic's
`/v1/messages/batches` endpoint in rate-limited chunks.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python send_batches.py \
    --batch sonnet_4_5_expanded.json \
    --chunk-size 1000 \
    --sleep 60
```

**Key options:**
- `--chunk-size N` -- Number of requests per API call (default: 1000)
- `--sleep S` -- Seconds to wait between chunks (default: 60)
- `--offset N` -- Skip the first N requests (for resuming after failures)

**Example: resuming after a partial failure:**

```bash
# First 2000 requests succeeded, resume from offset 2000
python send_batches.py \
    --batch sonnet_4_5_expanded.json \
    --chunk-size 1000 \
    --sleep 60 \
    --offset 2000
```

Each submitted chunk returns a **batch ID** (printed to stdout). Record these
IDs for polling and retrieval.

---

## Stage 4: Retrieve Batch Results

Batch result retrieval is done manually via curl. Use the batch IDs returned
by `send_batches.py`.

**Check batch status:**

```bash
curl https://api.anthropic.com/v1/messages/batches/{BATCH_ID} \
    --header "x-api-key: $ANTHROPIC_API_KEY" \
    --header "anthropic-version: 2023-06-01"
```

Poll until `processing_status` is `"ended"`.

**Download results:**

```bash
curl https://api.anthropic.com/v1/messages/batches/{BATCH_ID}/results \
    --header "x-api-key: $ANTHROPIC_API_KEY" \
    --header "anthropic-version: 2023-06-01" \
    -o batch_results.jsonl
```

The results file is JSONL (one JSON object per line), where each object
contains a `custom_id` and a `result` with the model's response.

---

## Stage 5: Convert Responses to Evaluation Format

Two scripts are available for converting batch responses into the JSON format
expected by the ParEval evaluation framework.

### Option A: `compile.py` (recommended for expanded batches)

Handles the `_copy_###` suffixes from expanded batches, grouping all
generations under the original prompt name.

```bash
python compile.py \
    --requests claude_sonnet_4_5_batch.json \
    --responses ./results_dir/ \
    > sonnet_combined.json
```

- `--requests` -- The original batch request file (JSON)
- `--responses` -- Directory containing one or more response JSONL files, or a
  single JSONL file

The output is a JSON array printed to stdout. Each entry includes:
- `problem_type`, `language`, `parallelism_model` (auto-inferred)
- `prompt` (reconstructed from request metadata)
- `outputs` (list of all generated code strings, sorted by copy index)

### Option B: `make_json_cleaned.py` (for single-copy batches)

Simpler converter that takes the batch `.sh` file (for prompt lookup) and a
single outputs JSONL file.

```bash
python make_json_cleaned.py \
    claude_sonnet_4_5_batch.sh \
    batch_results.jsonl
```

**Output:** `batch_results_cleaned.json` (written next to the input file).

---

## Stage 6: Clean Generated Code

Claude's raw output includes markdown code fences, `#include` directives, and
explanatory text. The cleaning scripts strip these to extract bare function
implementations for compilation.

### `clean_claude.py` (primary cleaner)

Extracts the first C++ function implementation body using brace-matching.

```bash
python clean_claude.py \
    --input sonnet_combined.json \
    --output sonnet_cleaned.json
```

If `--output` is omitted, writes to `<input_stem>_cleaned.json`.

### `clean_geometry.py` (lighter cleaner)

Only removes code fence markers and `#include` lines. Useful for geometry
problem types or cases where the full function-extraction regex is too
aggressive.

```bash
python clean_geometry.py sonnet_geometry.json \
    -o sonnet_geometry_cleaned.json
```

---

## Stage 7: Split by Problem Type

`parse_out.py` splits a combined JSON file into separate files per
`problem_type` for targeted SLURM execution.

```bash
python parse_out.py sonnet_cleaned.json ./output_by_type/
```

**Output:** One file per problem type in `./output_by_type/`:
```
fft.json
geometry.json
graph.json
histogram.json
reduce.json
scan.json
dense_la.json
sparse_la.json
sort.json
search.json
stencil.json
transform.json
locking_contention.json
futures_promises.json
```

---

## Stage 8: Run on Unity via SLURM

The SLURM scripts compile and execute the generated C++ code on Unity using the
`run-all.py` driver from `../drivers/`. Several job scripts are provided for
different scenarios.

### Environment Setup

All SLURM scripts assume:

```bash
module load conda/latest
conda activate hpc_llm

# HPX 1.5.1 with tcmalloc:
source /work/pi_mrobson_smith_edu/.hpx_tcmalloc_1.5.1

# Or HPX 1.10.0:
source /work/pi_mrobson_smith_edu/.hpx_1_10_0
```

> **Note:** Replace `/work/pi_mrobson_smith_edu/` with your own Unity workspace
> path. The HPX source scripts set up compiler flags, library paths, and other
> environment variables required to build HPX programs.

### Full Evaluation Run (`sonnet_run.sh`)

Runs all problem types on a single 64-core AMD EPYC 7763 node:

```bash
sbatch sonnet_run.sh
```

Key SLURM parameters:
- 1 node, 64 cores, 500 GB RAM, 30-hour time limit
- `--constraint=amd7763`, partition `cpu`

The script executes:

```bash
python run-all.py \
    "/work/pi_mrobson_smith_edu/ParEval_amt/generate/claude/sonnet_revised.json" \
    -o "/work/pi_mrobson_smith_edu/ParEval_amt/generate/claude/sonnet_run.json" \
    --yes-to-all \
    --include-models "hpx" \
    --build-timeout 30 \
    --run-timeout 45 \
    --log-build-errors \
    --log-runs
```

> **Note:** Update the input/output paths in the script to match your file
> locations before submitting.

### Per-Problem-Type Array Jobs (`batch_remaining_1_10.sh`)

Runs specific problem types in parallel as a SLURM array job against
HPX 1.10.0:

```bash
sbatch batch_remaining_1_10.sh
```

This dispatches 7 array tasks (indices 0-6), each processing a different
problem type (futures_promises, locking_contention, scan, search, sort,
stencil, transform). The launch configuration is loaded from
`launch_quick.json`.

### Re-run Specific Problem Types (`sonnet_batch_run.sh`)

For targeted re-runs of specific problem types (e.g., to validate results):

```bash
sbatch sonnet_batch_run.sh
```

### Residual Runs (`sparse_la_left.sh`)

For smaller leftover problems:

```bash
sbatch sparse_la_left.sh
```

### `run-all.py` Driver Options

The driver script (in `../drivers/`) accepts:

| Flag | Description |
|---|---|
| `--yes-to-all` | Skip confirmation prompts |
| `--include-models "hpx"` | Only run HPX parallelism model |
| `--build-timeout N` | Seconds before killing a build (default: 30) |
| `--run-timeout N` | Seconds before killing a run (default: 45-60) |
| `--log-build-errors` | Log compilation errors to output JSON |
| `--log-runs` | Log runtime output to output JSON |
| `--launch-configs FILE` | Path to launch configuration JSON (e.g., `launch_quick.json`) |

---

## Direct (Non-Batch) API Queries

For testing or generating a small number of responses without the batch
pipeline, you can query the Anthropic Messages API directly using curl.

The file `try.txt` provides an example:

```bash
curl https://api.anthropic.com/v1/messages/batches \
  --header "x-api-key: $ANTHROPIC_API_KEY" \
  --header "anthropic-version: 2023-06-01" \
  --header "content-type: application/json" \
  --data '{
    "requests": [
      {
        "custom_id": "sonnet_05_fft_inverse_fft",
        "params": {
          "model": "claude-sonnet-4-5",
          "max_tokens": 2048,
          "top_p": 0.95,
          "system": [
            {
              "type": "text",
              "text": "You are an expert modern C++ developer specializing in high-performance HPX code. Provide concise, correct implementations."
            }
          ],
          "thinking": {
            "type": "enabled",
            "budget_tokens": 1400
          },
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": "<your prompt here>"
                }
              ]
            }
          ]
        }
      }
    ]
  }'
```

To use **Opus** instead of Sonnet, change the `"model"` field:

```json
"model": "claude-opus-4-5"
```

To send a **synchronous (non-batch)** single-message request directly, use the
`/v1/messages` endpoint instead:

```bash
curl https://api.anthropic.com/v1/messages \
  --header "x-api-key: $ANTHROPIC_API_KEY" \
  --header "anthropic-version: 2023-06-01" \
  --header "content-type: application/json" \
  --data '{
    "model": "claude-sonnet-4-5",
    "max_tokens": 2048,
    "top_p": 0.95,
    "system": [
      {
        "type": "text",
        "text": "You are an expert modern C++ developer specializing in high-performance HPX code. Provide concise, correct implementations."
      }
    ],
    "thinking": {
      "type": "enabled",
      "budget_tokens": 1400
    },
    "messages": [
      {
        "role": "user",
        "content": "Your prompt text here"
      }
    ]
  }'
```

The synchronous endpoint returns the model response immediately in the HTTP
response body (no polling required).

---

## Utility Scripts

### `get_additional.py`

Filters a combined JSON file, drops certain problem types (fft, geometry,
graph, histogram, reduce), and writes the first 5 entries per remaining type
to separate `sonnet_<type>.json` files.

```bash
python get_additional.py combined.json
```

### `tmp.py`

One-off utility that filters for `dense_la` and `sparse_la` problem types
from an archive file. Hardcoded to Unity paths.

---

## Configuration Reference

### `launch_quick.json`

Defines how compiled programs are launched under each parallelism model. The
HPX configuration used in this project:

```json
"hpx": {
    "format": "srun -N 1 -n 1 -c {num_threads} {exec_path} {args} --hpx:threads={num_threads}",
    "params": [
        {"num_threads": 1}
    ]
}
```

Other supported models: `serial`, `omp`, `mpi`, `mpi+omp`, `kokkos`, `cuda`,
`hip`.

### Model Parameters

Both Sonnet and Opus use identical parameters:

| Parameter | Value |
|---|---|
| `max_tokens` | 2048 |
| `top_p` | 0.95 |
| `thinking.budget_tokens` | 1200--1400 |
| System prompt | "You are an expert modern C++ developer specializing in high-performance HPX code. Provide concise, correct implementations." |

### Problem Types (70 prompts)

| Category | Count | Description |
|---|---|---|
| FFT | 5 | Inverse FFT, DFT, conjugate, split, out-of-place |
| Geometry | 5 | Convex hull, perimeter, triangles, closest pair |
| Graph | 5 | Edge count, components, degrees, shortest path |
| Histogram | 5 | Pixel counts, binning, quadrants, quartiles |
| Reduce | 5 | XOR, min, max, and other reductions |
| Scan | 5 | Prefix sum, partial min, reverse prefix sum |
| Dense LA | 5 | LU decomposition, solve, GEMM, AXPY, GEMV |
| Sparse LA | 5 | Sparse solve, SpMM, SpMV, sparse AXPY/LU |
| Locking/Contention | 5 | Histogram freq, BFS, A*, locking matrix |
| Futures/Promises | 5 | Heat distribution, Jacobi, pi, integral, wave |
| Sort | 5 | Various parallel sorting algorithms |
| Search | 5 | Various parallel search algorithms |
| Stencil | 5 | Various stencil computations |
| Transform | 5 | Various parallel transforms |
