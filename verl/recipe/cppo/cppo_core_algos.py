# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

from typing import Any, Callable, Optional

import numpy as np
import torch
import verl.utils.torch_functional as verl_F


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: ``dp_size``, ``batch_num_tokens``, and ``global_batch_size`` are only compatible with the new model engine
        for now, while the legacy model engines conduct the aggregation outside ``agg_loss``.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    # TODO: Consider the numerical stability?

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size. When appling manual aggregation,
            scaling up the ``loss`` by ``dp_size`` can cancel out FSDP averaging.
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    # NOTE: `masked_sum` is more robust than multiplying the `mask`.
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        # if batch_num_tokens > 0:
        loss = verl_F.masked_sum(loss_mat, loss_mask) / (batch_num_tokens * dp_size + 1e-8)
        # else:
        #     loss = verl_F.masked_sum(loss_mat, loss_mask)
    elif loss_agg_mode.startswith("seq-mean"):
        # TODO: Correct and unify the denominator logic.
        if global_batch_size is not None:
            seq_denominator = global_batch_size * dp_size
        else:  # The default logic which is only correct when the batch sizes are even.
            local_bsz = loss_mat.shape[0]
            seq_denominator = local_bsz

        if loss_agg_mode.startswith("seq-mean-token-sum"):
            seq_losses = verl_F.masked_sum(loss_mat, loss_mask, axis=-1)  # token-sum per sequence

            if loss_agg_mode == "seq-mean-token-sum":
                pass  # TODO: Add assertation.
            elif loss_agg_mode == "seq-mean-token-sum-norm":
                if loss_scale_factor is None:
                    loss_scale_factor = loss_mask.shape[-1]
                seq_losses = seq_losses / loss_scale_factor
            else:
                raise ValueError(f"Invalid {loss_agg_mode=}")
        elif loss_agg_mode == "seq-mean-token-mean":
            token_counts = torch.sum(loss_mask, dim=-1)  # per-sequence token count
            # token-mean per sequence
            seq_losses = verl_F.masked_sum(loss_mat, loss_mask, axis=-1) / (token_counts + 1e-8)
        else:
            raise ValueError(f"Invalid {loss_agg_mode=}")
        loss = torch.sum(seq_losses) / seq_denominator  # seq-mean
    else:
        raise ValueError(f"Invalid {loss_agg_mode=}")

    return loss