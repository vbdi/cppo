# VLM Evaluation Pipeline

Our evaluation pipeline is adopted from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## Setup VLMEvalKit Environment

Follow instruction of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to set up the evaluation framework.

## Pipeline Overview

The evaluation consists of three sequential steps:

1. **Model Inference**: Run the model on test datasets and collect a group of `N` predictions
2. **Post-processing Group Results**: Extract grouped predictions into separate evaluation folders
3. **Performance Summary**: Aggregate results across multiple evaluation runs and compute statistics for `avg@N`

## Complete Example Workflow

Here's a complete example of running the entire pipeline for `Qwen2.5-VL-3B-Instruct`, with each step described in details after this example.

```bash
# Step 1: Run inference
bash run_eval.sh

# Wait for inference to complete...

# Step 2: Find the session name
SESSION_NAME=$(ls -dt ./outputs/Qwen2.5-VL-3B-Instruct/eval_num_g/Qwen2.5-VL-3B-Instruct/T* | head -1 | xargs basename)
echo "Session: $SESSION_NAME"

# Step 3: Post-process group results
python postprocess_group_results.py \
    ./outputs \
    Qwen2.5-VL-3B-Instruct \
    eval_num_g \
    $SESSION_NAME

# Step 4: Generate performance summary
python create_performance_summary.py \
    ./outputs/Qwen2.5-VL-3B-Instruct \
    $SESSION_NAME

# View results
cat ./eval_summary/Qwen2.5-VL-3B-Instruct.csv
```

## Step 1: Model Inference

Add the inference configuration to the config file: `vlmeval/config.py`. Set `group_size` variable to the number answers to be generated for each question.

Run the model inference on specified datasets using VLMEvalKit.

```bash
./run_eval.sh
```

#### Configuration

Edit `run_eval.sh` to configure the evaluation:

- **`idx`**: Select which model to evaluate (0-17). Model options include:
  - 0: `Qwen2.5-VL-3B-Instruct`
  - 1: `Qwen2.5-VL-3B-grpo-virl39k`
  - 2: `Qwen2.5-VL-3B-cppo-virl39k`
  - 3: `Qwen2.5-VL-3B-cppo-geometry3k`
  - And more (see the model list in the script) or add your own model configuration to the config file: `vlmeval/config.py`

- **`data`**: Space-separated list of datasets to evaluate on
  - Default: `LogicVista MathVista_MINI WeMath DynaMath MMMU_Pro_V MathVision_MINI MathVerse_MINI MathVision`

- **`EVAL_NUM`**: Evaluation batch identifier (default: `g` for grouped predictions)

### Output

The inference creates an output directory structure:
```
./outputs/
├── <model_name>/
│   └── eval_num_g/
│       └── <model_name>/
│           └── T<timestamp>_G<hash>/
│               ├── <model_name>_LogicVista.xlsx
│               ├── <model_name>_MathVista_MINI.xlsx
│               └── ... (one file per dataset)
```

---

## Step 2: Post-processing Group Results

Extract individual predictions from grouped evaluation results into separate `eval_num_*` folders.

### Command
```bash
python postprocess_group_results.py <base_dir> <model_name> <eval_num> <session_name> [datasets]
```

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `<base_dir>` | Base output directory path | `./outputs` |
| `<model_name>` | Model identifier | `Qwen2.5-VL-3B-Instruct` |
| `<eval_num>` | Evaluation number folder | `eval_num_g` |
| `<session_name>` | Session timestamp folder name | `T20260213_G8a4e7add` |
| `[datasets]` | Optional: Comma-separated dataset names | `LogicVista,MathVision_MINI` |

### Example Usage

```bash
# With custom datasets
python postprocess_group_results.py \
    ./outputs \
    Qwen2.5-VL-3B-Instruct \
    eval_num_g \
    T20260213_G8a4e7add \
    "LogicVista,MathVision_MINI,MathVista_MINI"
```

### Output

Creates separate evaluation folders for each grouped prediction:
```
./outputs/
├── <model_name>/
│   ├── eval_num_g/      (original grouped results)
│   ├── eval_num_0/      (first prediction per group)
│   ├── eval_num_1/      (second prediction per group)
│   ├── eval_num_2/      (third prediction per group)
│   └── ...
```

Each folder contains the same directory structure as `eval_num_g` but with individual predictions extracted.

---

## Step 3: Creating Performance Summary

Aggregate performance metrics across all evaluation folders and compute statistics.

### Command
```bash
python create_performance_summary.py <root_dir> <session_name>
```

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `<root_dir>` | Root directory containing evaluation results | `./outputs/Qwen2.5-VL-3B-Instruct` |
| `<session_name>` | Session timestamp folder name | `T20260213_G8a4e7add` |


### Example Usage

```bash
# Basic usage
python create_performance_summary.py \
    ./outputs/Qwen2.5-VL-3B-Instruct \
    T20260213_G8a4e7add
```

### Output

Generates a CSV summary file: `./eval_summary/<model_name>.csv`

The CSV contains:
- One row per evaluation folder (eval_num_0, eval_num_1, etc.)
- Performance metrics for each dataset:
  - LogicVista accuracy
  - MathVision accuracy
  - And more...
- **AVERAGE row**: Mean performance across all evaluation folders
- **STDDEV row**: Standard deviation across evaluation folders

---

