"""CPPO Vision RLVR Workflow - extends VisionRLVRWorkflow with semantic-based image augmentations."""

from typing import Any, cast
from collections.abc import Callable

import os
import aiofiles
import aiofiles.os
import colorama
import asyncio
import uuid
import torch
from areal.workflow.vision_rlvr import VisionRLVRWorkflow
from areal.api.engine_api import InferenceEngine
from areal.utils.image import image2base64
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.utils.data import concat_padded_tensors


class CPPO_VisionRLVRWorkflow(VisionRLVRWorkflow):
    """Vision RLVR Workflow with CPPO support.
    
    This workflow extends VisionRLVRWorkflow to handle semantic-preserving and
    semantic-changing image augmentations, which are used during training to
    compute contrastive loss.
    """

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Run an episode with support for semantic image variants.
        
        Args:
            engine: Inference engine for generation
            data: Input data containing images, messages, and optionally
                  semantic-preserving and semantic-changing image variants
                  
        Returns:
            Dictionary of tensors containing input_ids, loss_mask, logprobs,
            multi-modal inputs, advantages, rewards, and optionally semantic
            image variants.
        """
        processor_callable = cast(Callable[..., dict[str, Any]], self.processor)
        processed_input = processor_callable(
            images=data["images"],
            text=data["messages"],
            padding=False,
            return_tensors="pt",
        )

        input_ids: list[int] = processed_input["input_ids"].tolist()[0]

        byte_images = image2base64(data["images"])
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            image_data=byte_images,
            vision_msg_vllm=[data["messages_chat"]]
            if "messages_chat" in data
            else None,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        prompt_str = self.tokenizer.decode(input_ids)

        # Generate single response and compute reward
        resp, reward = await self._collect_samples(engine, req, prompt_str, data)

        # Build result tensor dict with batch dim 1
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        # Build multi-modal input
        multi_modal_input = [
            {
                "pixel_values": processed_input["pixel_values"],
            }
        ]
        if "image_grid_thw" in processed_input:
            multi_modal_input[0]["image_grid_thw"] = processed_input["image_grid_thw"]
        
        # CPPO: Process semantic-preserving images
        processed_input_sem_preserving = None
        if "images_sem_preserving" in data:
            processed_input_sem_preserving = self.processor(
                images=data["images_sem_preserving"],
                text=data["messages"],
                padding=False,
                return_tensors="pt",
            )

        # CPPO: Process semantic-changing images
        processed_input_sem_changed = None
        if "images_sem_changing" in data:
            processed_input_sem_changed = self.processor(
                images=data["images_sem_changing"],
                text=data["messages"],
                padding=False,
                return_tensors="pt",
            )
        
        # CPPO: Multi-modal input for semantic-preserving images
        multi_modal_input_sem_preserving = None
        if processed_input_sem_preserving is not None:
            multi_modal_input_sem_preserving = [
                {
                    "pixel_values": processed_input_sem_preserving["pixel_values"],
                }
            ]
            if "image_grid_thw" in processed_input_sem_preserving:
                multi_modal_input_sem_preserving[0]["image_grid_thw"] = processed_input_sem_preserving[
                    "image_grid_thw"
                ]
        
        # CPPO: Multi-modal input for semantic-changing images
        multi_modal_input_sem_changing = None
        if processed_input_sem_changed is not None:
            multi_modal_input_sem_changing = [
                {
                    "pixel_values": processed_input_sem_changed["pixel_values"],
                }
            ]
            if "image_grid_thw" in processed_input_sem_changed:
                multi_modal_input_sem_changing[0]["image_grid_thw"] = processed_input_sem_changed[
                    "image_grid_thw"
                ]

        return {
            "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "multi_modal_input": multi_modal_input,
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
            "multi_modal_input_sem_preserving": multi_modal_input_sem_preserving,
            "multi_modal_input_sem_changing": multi_modal_input_sem_changing,
        }