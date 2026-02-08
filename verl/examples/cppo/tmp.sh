set -x
ENGINE=${1:-vllm}

export USE_OPTIMIZED_MODEL=0
export VLLM_ASCEND_ENABLE_NZ=0


# GEOMETRY3K ##############################################################################################################################
PROJECT_NAME="Qwen2.5-VL-3B-Instruct-geometry3k"
TRAIN_METHOD="CPPO"

MODEL_PATH="/data/nfs/asif/pretrained_models/Qwen2.5-VL-3B-Instruct"

# ##########################
# EXP_NAME="debug" ## with topk:20% randking, coef = 0.01, 2 GPUS, filter vision mask based on the positive advantages, use_contrastive_formulation=False, agg_after_summation = False, agg_before_summation = True
# rm -rf /home/ahmad/verl/tensorboard_log/Qwen2.5-VL-3B-Instruct-geometry3k/debug
# TRAIN_BATCH_SIZE=32
# PPO_MINI_BATCH_SIZE=16
# PPO_MICRO_BATCH_SIZE_PER_GPU=8
# N=4
# SAVE_FREQ=62

# ### KL_neg and KL_pos only on vision tokens with advantage gating
# ### use_vision_mask = True, use_KL_pos = True, use_advantage_gating = True,  use_contrastive_formulation=False, agg_after_summation = False, csl_loss_coef = 0.01, topk=50%
# EXP_NAME="exp1_$TRAIN_METHOD" ## with topk:50% randking, coef = 0.01, 2 GPUS, filter vision mask based on the positive advantages, use_contrastive_formulation=False, agg_after_summation = False, agg_before_summation = True
# TRAIN_BATCH_SIZE=32
# PPO_MINI_BATCH_SIZE=16
# PPO_MICRO_BATCH_SIZE_PER_GPU=5
# N=5
# SAVE_FREQ=62
# MAX_RESPONSE_LENGTH=1024
# ##########################
# EXP_NAME="exp2_$TRAIN_METHOD" ## with topk:50% randking, coef = 0.01, 2 GPUS, filter vision mask based on the positive advantages, use_contrastive_formulation=False, agg_after_summation = False, agg_before_summation = True
# TRAIN_BATCH_SIZE=32
# PPO_MINI_BATCH_SIZE=8
# PPO_MICRO_BATCH_SIZE_PER_GPU=2
# N=5
# SAVE_FREQ=62
# MAX_RESPONSE_LENGTH=1024
# ##########################
# EXP_NAME="exp3_$TRAIN_METHOD" ## with topk:50% randking, coef = 0.01, 2 GPUS, filter vision mask based on the positive advantages, use_contrastive_formulation=False, agg_after_summation = False, agg_before_summation = True
# TRAIN_BATCH_SIZE=32
# PPO_MINI_BATCH_SIZE=16
# PPO_MICRO_BATCH_SIZE_PER_GPU=4
# N=4
# SAVE_FREQ=62
# MAX_RESPONSE_LENGTH=1024
##########################
EXP_NAME="exp4_$TRAIN_METHOD" ## with topk:50% randking, coef = 0.01, 2 GPUS, filter vision mask based on the positive advantages, use_contrastive_formulation=False, agg_after_summation = False, agg_before_summation = True
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=8
N=4
SAVE_FREQ=31
# MAX_RESPONSE_LENGTH=1024


EPOCHS=15
N_GPUS=16
TEST_FREQ=10
USE_KL_LOSS=True
KL_LOSS_COEF=0.01



TRAIN_DATA="/data/nfs/ahmad/dataset/geometry3k/data_forVerl/train.parquet"
TEST_DATA="/data/nfs/ahmad/dataset/geometry3k/data_forVerl/test.parquet"

SAVE_DIR="/data/nfs/ahmad/verl_logs/chechpoints/$PROJECT_NAME/$EXP_NAME"

REWARD_MANAGER="naive"

filter_overlong_prompts=True
val_before_train=False