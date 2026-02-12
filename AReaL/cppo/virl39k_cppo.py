"""CPPO Training Script using PPOTrainer"""

import re
import sys

from areal.api.cli_args import GRPOConfig, load_expr_config
from cppo.dataset import get_virl39k_cppo_rl_dataset
from areal.dataset import get_custom_dataset

from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from cppo.engine import PPOTrainer_CPPO as PPOTrainer

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict_str: str) -> float:
    """Reward function for format correctness (requires <think> tags and \\boxed{})."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    """Reward function for answer accuracy."""
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def custom_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
    """Combined reward function: accuracy with format bonus.
    
    Args:
        prompt: Prompt string
        completions: Generated completion
        prompt_ids: Tokenized prompt
        completion_ids: Tokenized completion
        answer: Ground truth answer
        **kwargs: Additional arguments
        
    Returns:
        Combined reward score between 0 and 1
    """
    format_reward_val = format_reward(completions)
    acc_reward_val = acc_reward(completions, answer)
    
    # Weight accuracy 90%, format 10%
    format_score = 0.1
    score = (1.0 - format_score) * acc_reward_val + format_score * format_reward_val
    
    return score


def main(args):
    """Main training function using PPOTrainer with CPPO support."""
    config, _ = load_expr_config(args, GRPOConfig)
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    # Create dataset and dataloaders with CPPO semantic image augmentations
    train_dataset = get_virl39k_cppo_rl_dataset(
        path=config.train_dataset.path,
        split="train",
        img_folder_path=config.train_dataset.path",
        processor=processor,
        max_length=config.train_dataset.max_length,
        sem_preserving_transforms=None,
        sem_changing_transforms=None
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )
    
    # Prepare workflow configuration dictionaries
    workflow_kwargs = dict(
        reward_fn=custom_reward_fn,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        processor=config.tokenizer_path,
        enable_thinking=False,
    )
    
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)
    
    # Use PPOTrainer context manager for simplified training
    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        print(f"Starting CPPO training with Geometry3K dataset using PPOTrainer...")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(valid_dataset)}")
        
        # Run training with CPPO workflow
        trainer.train(
            workflow="cppo.vision_rlvr.CPPO_VisionRLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="cppo.vision_rlvr.CPPO_VisionRLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
