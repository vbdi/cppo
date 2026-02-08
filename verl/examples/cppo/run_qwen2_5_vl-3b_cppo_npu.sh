set -x
ENGINE=${1:-vllm}

export USE_OPTIMIZED_MODEL=0
export VLLM_ASCEND_ENABLE_NZ=0

TRAIN_METHOD="CPPO" 
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=8
N=4
SAVE_FREQ=31
TEST_FREQ=31
EPOCHS=15
N_GPUS=16

## TODO:
PROJECT_NAME="Qwen2.5-VL-3B-Instruct-geometry3k-CPPO"  # TODO: change if needed
EXP_NAME="exp1_$TRAIN_METHOD" # TODO: change if needed
USE_CPL_LOSS=True
CPL_LOSS_COEF=0.01
CPL_USE_VISION_MASK=True
CPL_VISION_TOP_PERCENT=0.5
CPL_USE_ADVANTAGE_GATING=True
CPL_ETA=0.1

TRAIN_DATA="../data/geometry3k_verl/train.parquet" # TODO : change if needed
TEST_DATA="../data/geometry3k_verl/test.parquet"  # TODO : change if needed

SAVE_DIR="checkpoints/$PROJECT_NAME/$EXP_NAME" # TODO: change if needed
MODEL_PATH="pretrained_models/Qwen2.5-VL-3B-Instruct" # TODO: change if needed


filter_overlong_prompts=True
val_before_train=False

python3 -m recipe.cppo.cppo_main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${TEST_DATA} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation='right' \
    data.image_key=images \
    data.custom_cls.path="recipe/cppo/cppo_dataset.py" \
    data.custom_cls.name=cppo_RLHFDataset \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.use_vision_cpl_loss=${USE_CPL_LOSS} \
    actor_rollout_ref.actor.cpl_loss_coef=${CPL_LOSS_COEF} \
    actor_rollout_ref.actor.cpl_use_vision_mask=${CPL_USE_VISION_MASK} \
    actor_rollout_ref.actor.cpl_vision_top_percent=${CPL_VISION_TOP_PERCENT} \
    actor_rollout_ref.actor.cpl_use_advantage_gating=${CPL_USE_ADVANTAGE_GATING} \
    actor_rollout_ref.actor.cpl_eta=${CPL_ETA} \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.val_before_train=${val_before_train} \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.rollout_data_dir=outputs/${PROJECT_NAME}/${EXP_NAME}/rollout_dump_train \
    trainer.validation_data_dir=outputs/${PROJECT_NAME}/${EXP_NAME}/rollout_dump_val \
    trainer.total_epochs=${EPOCHS} $@
