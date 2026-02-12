#!/bin/bash


EXP_NAME="Qwen2.5-VL-3B-Instruct-geometry3k-cppo"
TRIAL_NAME="trail0"

ALLOCATION_MODE="vllm:d2+d2"
N_NODES=1
N_GPUS=4

MODEL_PATH="../pretrained_models/Qwen2.5-VL-3B-Instruct"
TRAIN_DATASET="../data/geometry3k"
VALID_DATASET="../data/geometry3k"

## Batch settings:
BATCH_SIZE=2
VALID_BATCH_SIZE=2
MAX_TOKENS_PER_MB=4096
N_MBS=1
EPOCHS=2

### CPPO:
CPL_LOSS_COEF=0.01
CPL_USE_VISION_MASK=true
CPL_VISION_TOP_PERCENT=0.5
CPL_USE_ADVANTAGE_GATING=true
CPL_TAU=0.1

### ROLLOUT CONFIG:
MAX_CONCURRENT_ROLLOUT=128
MAX_HEAD_OFFPOLICYNESS=4

### GENERATION CONFIG:
GROUP_SIZE=4
MAX_NEW_TOKENS=4096

### SAVING:
LOG_SAVE_DIR="logs"
CHECKPOINT_SAVE_DIR="checkpoints"
SAVE_FREQ_EPOCHS=1
TENSORBOARD_PATH="tensorboard_logs/${EXP_NAME}/${TRIAL_NAME}"


python -m areal.launcher.local \
    cppo/geometry3k_cppo.py --config cppo/geometry3k_cppo.yaml \
    experiment_name=${EXP_NAME} \
    trial_name=${TRIAL_NAME} \
    allocation_mode=${ALLOCATION_MODE} \
    cluster.n_nodes=${N_NODES} \
    cluster.n_gpus_per_node=${N_GPUS} \
    train_dataset.path=${TRAIN_DATASET} \
    valid_dataset.path=${VALID_DATASET} \
    actor.path=${MODEL_PATH} \
    train_dataset.batch_size=${BATCH_SIZE} \
    valid_dataset.batch_size=${VALID_BATCH_SIZE} \
    actor.mb_spec.max_tokens_per_mb=${MAX_TOKENS_PER_MB} \
    actor.mb_spec.n_mbs=${N_MBS} \
    actor.cpl_loss_coef=${CPL_LOSS_COEF} \
    actor.cpl_use_vision_mask=${CPL_USE_VISION_MASK} \
    actor.cpl_vision_top_percent=${CPL_VISION_TOP_PERCENT} \
    actor.cpl_use_advantage_gating=${CPL_USE_ADVANTAGE_GATING} \
    actor.cpl_tau=${CPL_TAU} \
    rollout.max_concurrent_rollouts=${MAX_CONCURRENT_ROLLOUT} \
    rollout.max_head_offpolicyness=${MAX_HEAD_OFFPOLICYNESS} \
    gconfig.n_samples=${GROUP_SIZE} \
    gconfig.max_new_tokens=${MAX_NEW_TOKENS} \
    cluster.fileroot=${LOG_SAVE_DIR} \
    saver.fileroot=${CHECKPOINT_SAVE_DIR} \
    saver.freq_epochs=${SAVE_FREQ_EPOCHS} \
    stats_logger.tensorboard.path=${TENSORBOARD_PATH} \
    total_train_epochs=${EPOCHS} \
