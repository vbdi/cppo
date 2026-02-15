#!/bin/bash

#### DATASET:
data="LogicVista MathVista_MINI WeMath DynaMath MMMU_Pro_V MathVision_MINI MathVerse_MINI MathVision"

#### Model Inference Configuration is set in the vlmeval/configs.py
#### MODELS: You can change the index to select different models for evaluation. The mapping of index to model is as follows:
models=(
    "Qwen2.5-VL-3B-Instruct"  # idx: 0
    "Qwen2.5-VL-3B-grpo-virl39k"  # idx: 1
    "Qwen2.5-VL-3B-cppo-virl39k"  # idx: 2
    "Qwen2.5-VL-3B-cppo-geometry3k"  # idx: 3
    "Qwen2.5-VL-7B-Instruct"  # idx: 4
    "Qwen2.5-VL-7B-grpo-virl39k"  # idx: 5
    "Qwen2.5-VL-7B-cppo-virl39k" # idx: 6
    "Qwen2.5-VL-3B-visionary-r1" # idx: 7
    "OpenVLThinker-3B" # idx: 8
    "Qwen2.5-VL-3B-shuffle-r1" # idx: 9
    "PAPO-G-H-Qwen2.5-VL-3B" # idx: 10
    "PAPO-G-H-Qwen2.5-VL-7B" # idx: 11
    "Qwen2.5-VL-7B-vision-sr1" # idx: 12
    "Qwen2.5-VL-3B-perception-r1-7b" # idx: 13
    "OpenVLThinker-7B" # idx: 14
    "NoisyRollout-Qwen2.5-VL-7B" # idx: 15
    "Semantic-Back-Qwen2.5-VL-7B" # idx: 16
    "Vision_Matters_Qwen2.5-VL-7B" # idx: 17
)

idx=0  # Change this index to select different models for evaluation
model=${models[$idx]}  # Change the index to select different models for evaluation

#### EVALUATION EXPERIMENT:
EVAL_NUM=g # for group generation
OUTPUT_dir="./outputs/${model}/eval_num_${EVAL_NUM}"

#### RUN:
python run.py \
        --model ${model} \
        --data ${data} \
        --verbos \
        --mode infer \
        --work-dir ${OUTPUT_dir} \
        --use-vllm
