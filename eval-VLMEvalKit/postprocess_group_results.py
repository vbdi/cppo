import pandas as pd
import os
import ast
import copy
import sys

def safe_eval(x):
    try:
        return ast.literal_eval(str(x))
    except Exception as e:
        print('Bad row:',e,x)
        return x

# Parse command line arguments
if len(sys.argv) < 5:
    print("Usage: python postprocess_group_results.py <base_dir> <model_name> <eval_num> <session_name> [datasets]")
    print("Example: python postprocess_group_results.py ./outputs Qwen2.5-VL-3B-Instruct eval_num_g T20260213_G8a4e7add")
    sys.exit(1)

base_dir = sys.argv[1]
model_name = sys.argv[2]
eval_num = sys.argv[3]
session_folder = sys.argv[4]

# Define datasets - can be overridden via command line
if len(sys.argv) > 5:
    datasets = sys.argv[5].split(',')
else:
    datasets = ["MathVision_MINI","LogicVista","MathVerse_MINI","MathVista_MINI","WeMath","MMMU_Pro_V","DynaMath"]

for dataset in datasets:
    file_path = f'{model_name}/{session_folder}/{model_name}_{dataset}.xlsx'

    df = pd.read_excel(f'{base_dir}/{eval_num}/{file_path}')
    df['prediction'] = df['prediction'].apply(safe_eval)
    group_size=len(list(df['prediction'])[0])
    # print(df['prediction'][0][:])
    print(len(df['prediction'][0][:]))
    print(len(df['prediction']))
    for idx in range(0,group_size):
        if f'eval_num_{idx}'==eval_num:
            continue
        try:
            os.mkdir(f'{base_dir}/eval_num_{idx}')
        except:
            pass
        pred = []
        for row in range(len(df['prediction'])):
            pred.append(df['prediction'][row][idx])
        new_df = copy.deepcopy(df)                 
        new_df['prediction'] = pred


        file_names = file_path.split('/')
        dir = f'{base_dir}/eval_num_{idx}/'
        for fold_idx in range(len(file_names)-1):
            dir = f'{dir}/{file_names[fold_idx]}'
            try:
                os.mkdir(f'{dir}')
            except:
                pass

        # file_name = file_path.split('/')[-1]
        new_df.to_excel(f'{base_dir}/eval_num_{idx}/{file_path}')