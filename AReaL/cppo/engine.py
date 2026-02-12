import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

from collections.abc import Callable
from typing import Any

from areal.api.cli_args import PPOActorConfig
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.ppo.actor import PPOActor
from areal.experimental.trainer.rl import PPOTrainer
from areal.utils.data import KLEstimator
from areal.engine.core import compute_total_loss_weight
from areal.utils.data import MicroBatchList
from areal.api.scheduler_api import Scheduler

def top_percent_positive_mask_per_chunk(mask, values, percent=0.5):
    """Select top percent of positive values in each chunk of the mask."""
    new_mask = torch.zeros_like(mask)
    is_start = (mask[1:] == 1) & (mask[:-1] == 0)
    is_end   = (mask[:-1] == 1) & (mask[1:] == 0)
    start_idxs = torch.where(is_start)[0] + 1
    end_idxs   = torch.where(is_end)[0] + 1

    if mask[0] == 1:
        start_idxs = torch.cat([torch.tensor([0], device=mask.device), start_idxs])
    if mask[-1] == 1:
        end_idxs = torch.cat([end_idxs, torch.tensor([len(mask)], device=mask.device)])

    for s, e in zip(start_idxs, end_idxs):
        vals = values[s:e]
        pos_mask = vals > 0
        pos_indices = torch.where(pos_mask)[0]
        if len(pos_indices) == 0:
            continue
        pos_vals = vals[pos_indices]
        num_keep = math.ceil(percent * len(pos_vals))
        if num_keep > 0:
            top_rel_idx = torch.topk(pos_vals, num_keep).indices
            top_idx = pos_indices[top_rel_idx] + s
            new_mask[top_idx] = 1

    return new_mask

def agg_loss_masked_mean(loss_values, mask, axis=None):    
    """Compute masked mean of loss values."""
    valid_values = torch.where(mask.bool(), loss_values, 0.0)
    s = valid_values.sum(axis=axis)
    return s / (mask.sum(axis=axis) + 1e-8)

class FSDPEngine_CPPO(FSDPEngine):
    """FSDP Engine with CPPO.
    
    This engine extends FSDPEngine to support semantic-based contrastive learning,
    which uses auxiliary images with semantic-preserving and semantic-changing
    transformations to improve model perception.
    """
    
    def __init__(self, config):
        super().__init__(config)
        # CPPO-specific hyperparameters (read from config with defaults)
        self.cpl_loss_coef = getattr(config, 'cpl_loss_coef', 0.01)
        self.cpl_use_vision_mask = getattr(config, 'cpl_use_vision_mask', True)
        self.cpl_vision_top_percent = getattr(config, 'cpl_vision_top_percent', 0.5)
        self.cpl_use_advantage_gating = getattr(config, 'cpl_use_advantage_gating', True)
        self.cpl_tau = getattr(config, 'cpl_tau', 0.1)
    
    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        """Prepare microbatch list with CPPO-specific multi-modal inputs.
        
        Extends parent method to handle semantic-preserving and semantic-changing images.
        """
        # Call parent to get base mb_list
        mb_list = super()._prepare_mb_list(input_)

        # CPPO: Post-process to extract semantic image variants
        assert mb_list.padded_mbs is not None
        for i, mb in enumerate(mb_list.padded_mbs):
            # Handle semantic-preserving images
            if "multi_modal_input_sem_preserving" in mb:
                image_grid_thw_list = [
                    item["image_grid_thw"]
                    for item in mb["multi_modal_input_sem_preserving"]
                    if "image_grid_thw" in item
                ]
                if image_grid_thw_list:
                    mb["image_grid_thw_sem_preserving"] = torch.cat(image_grid_thw_list, dim=0)

                pixel_values_list = [
                    item["pixel_values"]
                    for item in mb["multi_modal_input_sem_preserving"]
                    if "pixel_values" in item
                ]
                if pixel_values_list:
                    mb["pixel_values_sem_preserving"] = torch.cat(pixel_values_list, dim=0)

            # Handle semantic-changing images
            if "multi_modal_input_sem_changing" in mb:
                image_grid_thw_list = [
                    item["image_grid_thw"]
                    for item in mb["multi_modal_input_sem_changing"]
                    if "image_grid_thw" in item
                ]
                if image_grid_thw_list:
                    mb["image_grid_thw_sem_changing"] = torch.cat(image_grid_thw_list, dim=0)

                pixel_values_list = [
                    item["pixel_values"]
                    for item in mb["multi_modal_input_sem_changing"]
                    if "pixel_values" in item
                ]
                if pixel_values_list:
                    mb["pixel_values_sem_changing"] = torch.cat(pixel_values_list, dim=0)

        return mb_list
    
    def _compute_cppo_loss(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        mb_input: dict[str, Any],
        ulysses_pad_size: int = 0,
        pad_length: int = 0,
    ) -> torch.Tensor | None:
        """Compute CPL loss for CPPO.
        
        Returns None if semantic image variants are not available.
        """
        # Check if semantic image variants are available
        if "pixel_values_sem_preserving" not in inputs or "pixel_values_sem_changing" not in inputs:
            return None

        kl_estimator = KLEstimator(kl_estimator="k1", apply_clamp=True)

        # Compute logits for semantic-changing images
        removing_output = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=None,
            position_ids=inputs['position_ids'],
            pixel_values=inputs['pixel_values_sem_changing'],
            image_grid_thw=inputs['image_grid_thw_sem_changing'],
            use_cache=False,
        )
        removing_logits = removing_output.logits.squeeze(0)
        log_probs_noimg, H_noimg = self._compute_logprobs_entropy(
            removing_logits, inputs, ulysses_pad_size
        )
        if pad_length > 0:
            log_probs_noimg = log_probs_noimg[:-pad_length]
            H_noimg = H_noimg[:-pad_length]

        # Compute logits for semantic-preserving images
        preserving_output = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=None,
            position_ids=inputs['position_ids'],
            pixel_values=inputs['pixel_values_sem_preserving'],
            image_grid_thw=inputs['image_grid_thw_sem_preserving'],
            use_cache=False,
        )
        preserving_logits = preserving_output.logits.squeeze(0)
        log_probs_preserving, _ = self._compute_logprobs_entropy(
            preserving_logits, inputs, ulysses_pad_size
        )
        if pad_length > 0:
            log_probs_preserving = log_probs_preserving[:-pad_length]

        # Get original logprobs and entropy for CPL computation
        logprobs, entropy = self._compute_logprobs_entropy(logits, inputs, ulysses_pad_size)
        if pad_length > 0:
            logprobs = logprobs[:-pad_length]
            entropy = entropy[:-pad_length]

        if self.cpl_use_vision_mask:
            with torch.no_grad():
                delta_H = H_noimg.detach() - entropy.detach()

                eps_ = 1e-6
                # Normalize by baseline entropy
                norm_delta = torch.relu(delta_H) / (H_noimg + eps_)

                # Apply loss mask
                norm_delta = norm_delta * mb_input['loss_mask']

                # Mask out non-positive changes by setting them to -inf
                masked_vals = norm_delta.masked_fill(delta_H <= 0, float("-inf"))
                vision_mask = top_percent_positive_mask_per_chunk(
                    mb_input['loss_mask'], masked_vals, self.cpl_vision_top_percent
                )
        else:
            vision_mask = mb_input['loss_mask']
        
        if self.cpl_use_advantage_gating:
            advantages = inputs['advantages'][0, :-pad_length] if pad_length > 0 else inputs['advantages'][0]
            vision_mask = torch.where(advantages > 0, vision_mask, 0)

        kl_semantics_preserving = kl_estimator(log_probs=logprobs, log_probs_base=log_probs_preserving)
        kl_semantics_changing = kl_estimator(log_probs=logprobs, log_probs_base=log_probs_noimg)
        margin = kl_semantics_preserving - kl_semantics_changing
        cpl = F.softplus(margin/self.cpl_tau)
        cpl_loss = agg_loss_masked_mean(cpl, vision_mask, axis=None)
        return cpl_loss
    
    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        # torch.distributed.breakpoint(rank=0)
        self._ensure_ready()
        self.optimizer_zero_grad()

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.dp_group
        )

        # Step 3: Forward-backward using process_output_fn callback with CPPO logic
        def process_output_cppo(
            logits: torch.Tensor,
            ctx_dict: dict[str, Any],
            loss_multiplier: float = 1.0,
        ) -> torch.Tensor:
            """Process outputs and compute both PPO and CPPO losses."""
            ctx = ctx_dict  # FSDPTrainContext
            mb_input = ctx["mb_input"]
            ulysses_pad_size = ctx.get("ulysses_pad_size", 0)
            pad_length = ctx.get("pad_length", 0)
            inputs = ctx["model_inputs"]

            # Compute standard PPO loss
            if not self.config.is_critic:
                logprobs, entropy = self._compute_logprobs_entropy(
                    logits, inputs, ulysses_pad_size
                )
                vocab_min_logits, vocab_max_logits = self._get_vocab_min_max_logits(
                    logits, ulysses_pad_size
                )
                if pad_length > 0:
                    logprobs = logprobs[:-pad_length]
                    entropy = entropy[:-pad_length]
                    vocab_min_logits = vocab_min_logits[: -pad_length]
                    vocab_max_logits = vocab_max_logits[: -pad_length]

                loss = loss_fn(
                    logprobs,
                    entropy,
                    mb_input,
                    vocab_min_logits=vocab_min_logits,
                    vocab_max_logits=vocab_max_logits,
                )
            else:
                values = self._compute_values(logits.squeeze(-1), ulysses_pad_size)
                if pad_length > 0:
                    values = values[:-pad_length]
                loss = loss_fn(values, mb_input)

            # CPPO:
            cpl_loss = self._compute_cppo_loss(
                logits, inputs, mb_input, ulysses_pad_size, pad_length
            )

            # Combine losses
            if cpl_loss is not None:
                loss += cpl_loss * self.cpl_loss_coef

            loss_scale = loss_weight_fn(mb_input) / total_loss_weight * loss_multiplier
            return loss * loss_scale


        self.forward_backward_batch(mb_list, process_output_cppo, forward_only=False)

        # Step 4: Optimizer step
        return self.optimizer_step()



class FSDPPPOActor_CPPO(FSDPEngine_CPPO):
    """FSDP-based PPO Actor with CPPO support."""
    
    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> dict[str, Any]:
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        
        self.actor.ppo_update(*args, **kwargs)
    
    @classmethod
    def as_controller(cls, config: PPOActorConfig, scheduler: Scheduler):
        from areal.engine.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)

from areal.engine.ppo.actor import PPOActorController
from areal.utils.environ import is_single_controller
class PPOTrainer_CPPO(PPOTrainer):
    """PPO Trainer with CPPO support."""
 
    def _create_actor(
        self, actor_config: PPOActorConfig
    ) -> FSDPPPOActor_CPPO | PPOActorController:
        if self.allocation_mode.train_backend == "fsdp":
            
            actor_cls = FSDPPPOActor_CPPO
        else:
            raise ValueError(
                f"Invalid backend: {self.allocation_mode.train_backend}, expected fsdp"
            )
        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor