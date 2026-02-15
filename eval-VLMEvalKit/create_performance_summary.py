import os
import pandas as pd
import re
import sys


EXTRACTION_MAP = {
    "LogicVista_gpt4o-mini_score": (0, "acc"),
    "MathVision_MINI_gpt-4o-mini_score": (0, "acc"),
    "MathVista_MINI_gpt-4o-mini_score": (0, "acc"),
    "WeMath_gpt4o-mini_score": (0, "Score (Strict)"),
    "DynaMath_gpt-4o-mini_score": (0, "Overall"),
    "MathVerse_MINI_gpt-4o-mini_score": (0, "Overall"),
    "MMMU_Pro_V_acc": (0, "Overall"),
    "MMBench_DEV_EN_V11_acc": (0, "Overall"),
    "HallusionBench_score": (0, "aAcc"),
    "MMVet_gpt-4-turbo_score": (6, "acc"),
    "RealWorldQA_acc": (0, "Overall"),
}

# Parse command line arguments
if len(sys.argv) < 3:
    print("Usage: python create_performance_summary.py <root_dir> <session_name>")
    print("Example: python create_performance_summary.py ./outputs/Qwen2.5-VL-3B-Instruct T20260213_G8a4e7add")
    sys.exit(1)

ROOT_DIR = sys.argv[1]
SESSION_NAME = sys.argv[2]


results = []

def get_last_folder_name(path_string):
    normalized_path = os.path.normpath(path_string)
    last_folder = os.path.basename(normalized_path)
    return last_folder

def eval_num_sort_key(name: str) -> int:
    """
    Extract numeric suffix from eval_num_X for proper numeric sorting.
    """
    match = re.search(r"eval_num_(\d+)", name)
    return int(match.group(1)) if match else float("inf")

rows = []

for eval_dir in sorted(os.listdir(ROOT_DIR), key=eval_num_sort_key):
    eval_path = os.path.join(ROOT_DIR, eval_dir)

    if not os.path.isdir(eval_path) or get_last_folder_name(eval_path) == "eval_num_g":
        continue
    
    row_data = {
        "eval_folder": eval_dir,
        **{key: None for key in EXTRACTION_MAP}
    }
    
    for dataset_name  in EXTRACTION_MAP.keys():
        exp_name = get_last_folder_name(ROOT_DIR)
        csv_file = os.path.join(eval_path, exp_name, SESSION_NAME, f"{exp_name}_{dataset_name}.csv")

        row_idx, column_name = EXTRACTION_MAP[dataset_name]

        try:
            df = pd.read_csv(csv_file)

            value = df.loc[row_idx, column_name]
            if '%' in str(value):
                value = float(str(value).strip('%')) / 100.0
                row_data[dataset_name] = float(df.iloc[row_idx][column_name].strip('%')) / 100.0
            else:
                row_data[dataset_name] = df.iloc[row_idx][column_name]

            results.append({
                "eval_folder": eval_dir,
                "file": dataset_name,
                "row": row_idx,
                "column": column_name,
                "value": value
            })

        except Exception as e:
            print(f"ERROR reading {csv_file}: {e}")

    rows.append(row_data)

# Save results to CSV
df_out = pd.DataFrame(rows)

# Ensure numeric columns where possible
for col in EXTRACTION_MAP.keys():
    df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

# Compute averages
avg_row = {
    "eval_folder": "AVERAGE"
}

for col in EXTRACTION_MAP.keys():
    avg_row[col] = df_out[col].mean()

# Compute standard deviations
std_row = {
    "eval_folder": "STDDEV"
}
for col in EXTRACTION_MAP.keys():
    std_row[col] = df_out[col].std()

# Append average and stddev rows
df_out = pd.concat(
    [df_out, pd.DataFrame([avg_row, std_row])],
    ignore_index=True
)

# Ensure correct column order
df_out = df_out[["eval_folder", *EXTRACTION_MAP.keys()]]


OUTPUT_CSV = os.path.join("./eval_summary", f"{exp_name}.csv")
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Performance summary saved to {OUTPUT_CSV}")