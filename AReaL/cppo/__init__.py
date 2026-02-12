"""CPPO: Contrastive Perception Policy Optimization for Vision-Language Models with AReaL framework."""


from cppo.engine import (
    FSDPEngine_CPPO,
    FSDPPPOActor_CPPO,
    PPOTrainer_CPPO,
    top_percent_positive_mask_per_chunk,
    agg_loss_masked_mean,
)

from cppo.vision_rlvr import CPPO_VisionRLVRWorkflow

from cppo.dataset import (
    get_geometry3k_cppo_rl_dataset,
    get_virl39k_cppo_rl_dataset,
    RandomOcclusion,
    RandomZoomCrop,
    BlackSquareCover,
    GaussianBlur,
)

__all__ = [
    "FSDPEngine_CPPO",
    "FSDPPPOActor_CPPO",
    "PPOTrainer_CPPO",
    "CPPO_VisionRLVRWorkflow",
    "get_geometry3k_cppo_rl_dataset",
    "get_virl39k_cppo_rl_dataset",
    "RandomOcclusion",
    "RandomZoomCrop",
    "BlackSquareCover",
    "GaussianBlur",
    "top_percent_positive_mask_per_chunk",
    "agg_loss_masked_mean",
]