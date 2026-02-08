# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from PIL import Image
import random
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import torchvision.transforms as T

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image

logger = logging.getLogger(__name__)

from PIL import Image
import random

import numpy as np

class RandomOcclusion:
    def __init__(self, p=0.5, scale=(0.05, 0.15),iter=10):
        self.p = p
        self.scale = scale
        self.iter = iter

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = img.copy()  # <--- make a copy
        w, h = img.size
        for ii in range(self.iter):
            occ_w = int(random.uniform(*self.scale) * w)
            occ_h = int(random.uniform(*self.scale) * h)
            x = random.randint(0, w - occ_w)
            y = random.randint(0, h - occ_h)

            # random noise patch
            noise = np.uint8(np.random.rand(occ_h, occ_w, 3) * 255)
            patch = Image.fromarray(noise, mode="RGB")
            img.paste(patch, (x, y))
        return img

class RandomZoomCrop:
    def __init__(self, scale=(0.8, 1.0)):
        """
        scale: tuple, fraction of the original area to crop
               e.g., (0.8, 1.0) means crop 80%-100% of the area
        """
        self.scale = scale

    def __call__(self, img):
        w, h = img.size
        # choose a random crop size
        scale_factor = random.uniform(*self.scale)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)

        # choose a random top-left corner for the crop
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)

        # crop the image
        cropped = img.crop((left, top, left + new_w, top + new_h))

        # resize back to original size
        zoomed = cropped.resize((w, h), Image.BILINEAR)
        return zoomed

class GaussianBlur:
    """
    A torchvision-like GaussianBlur transform implemented in PyTorch.
    - Takes only hyperparameters (kernel_size, sigma)
    - Called directly with a PIL Image
    - No device usage anywhere
    - Keeps image dimensions unchanged
    """

    def __init__(self, kernel_size=11, sigma=1.5):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(sigma, (int, float)):
            sigma = (sigma, sigma)
        self.sigma = sigma

    def _make_kernel_1d(self, kernel_size, sigma, ref_tensor):
        """Build 1D Gaussian kernel inheriting dtype/device from ref_tensor."""
        radius = kernel_size // 2
        x = ref_tensor.new_tensor(range(-radius, radius + 1), dtype=ref_tensor.dtype)
        kernel = torch.exp(-(x ** 2) / (2 * sigma * sigma))
        kernel /= kernel.sum()
        return kernel

    def __call__(self, img: Image.Image):
        """Apply Gaussian blur to a PIL image."""
        # Convert PIL → tensor
        t = to_tensor(img).unsqueeze(0)  # NCHW
        C = t.shape[1]
        ref = t  # to inherit device & dtype

        # Sample sigma (range allowed)
        if self.sigma[0] == self.sigma[1]:
            sigma = float(self.sigma[0])
        else:
            sigma = float(ref.new_empty(1).uniform_(*self.sigma))

        # Create 1D Gaussian kernels
        kx = self._make_kernel_1d(self.kernel_size[0], sigma, ref)
        ky = self._make_kernel_1d(self.kernel_size[1], sigma, ref)

        # Expand into depthwise convolution kernels
        kx = kx.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        ky = ky.view(1, 1, 1, -1).repeat(C, 1, 1, 1)

        # Apply separable convolution with SAME padding
        with torch.backends.mkldnn.flags(enabled=False):
            t = F.conv2d(t, kx, padding="same", groups=C)
            t = F.conv2d(t, ky, padding="same", groups=C)

        # Back to PIL
        t = t.squeeze(0).clamp(0, 1)
        return to_pil_image(t)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class cppo_RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        #### image modifications:
        augment= True
        self.augment = augment or config.get("return_augmented_images", False)
        # Define SEMANTICS-PRESERVING augmentations
        self.sem_preserving_transforms = [
            T.ColorJitter(brightness=(0.2, 1.3), contrast=(0.2, 1.8), saturation=(0.2, 1.8)),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            # T.RandomHorizontalFlip(p=1.0),
            T.RandomRotation(degrees=10),
            # T.RandomResizedCrop(224, scale=(0.9, 1.0)),
            # T.GaussianBlur(kernel_size=3),
            # GaussianBlur(kernel_size=3,sigma=1),
        ]

        # Define SEMANTICS-CHANGING augmentations
        self.sem_changing_transforms = [
            RandomOcclusion(p=1.0,iter=50),
            RandomZoomCrop(scale=(0.6,0.7)),
            # T.RandomPosterize(bits=2),
            # T.RandomSolarize(threshold=128),
            # T.RandomInvert(p=1.0),
            # T.RandomAffine(degrees=45, shear=20),
            # T.RandomVerticalFlip(p=1.0),
        ]

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                    videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None
                    
                    # print(doc["extra_info"]["index"])
                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item, 
                   sem_preserve_idx: int = None, 
                   sem_change_idx: int = None, 
                   with_augmented: bool = False):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images
                # import ipdb; ipdb.set_trace()
                if self.augment or with_augmented:
                    # import ipdb; ipdb.set_trace()
                    sem_preserving_imgs, sem_changing_imgs = [], []

                    for img in images:
                        pil_img = Image.fromarray(img.numpy()) if not isinstance(img, Image.Image) else img
                        # import ipdb; ipdb.set_trace()
                        # ---- semantics-preserving ----
                        if sem_preserve_idx is not None:
                            transform = self.sem_preserving_transforms[sem_preserve_idx]
                        else:
                            transform = random.choice(self.sem_preserving_transforms)
                        sem_preserving_imgs.append(transform(pil_img))

                        # ---- semantics-changing ----
                        if sem_change_idx is not None:
                            transform = self.sem_changing_transforms[sem_change_idx]
                        else:
                            transform = random.choice(self.sem_changing_transforms)
                        sem_changing_imgs.append(transform(pil_img))
                        # import ipdb; ipdb.set_trace()
                    row_dict["sem_preserving_images"] = sem_preserving_imgs
                    row_dict["sem_changing_images"] = sem_changing_imgs

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            # import ipdb; ipdb.set_trace()
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            model_inputs_sem_preserving_imgs = self.processor(text=[raw_prompt], images=sem_preserving_imgs, videos=videos, return_tensors="pt")
            model_inputs_sem_changing_imgs = self.processor(text=[raw_prompt], images=sem_changing_imgs, videos=videos, return_tensors="pt")
            input_ids_sem_preserving_imgs = model_inputs_sem_preserving_imgs.pop("input_ids")
            input_ids_sem_changing_imgs = model_inputs_sem_changing_imgs.pop("input_ids")
            input_ids_sem_preserving_imgs = model_inputs_sem_preserving_imgs.pop("attention_mask")
            input_ids_sem_changing_imgs = model_inputs_sem_changing_imgs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")
            
            if "second_per_grid_ts" in model_inputs_sem_preserving_imgs:
                model_inputs_sem_preserving_imgs.pop("second_per_grid_ts")
                
            if "second_per_grid_ts" in model_inputs_sem_changing_imgs:
                model_inputs_sem_changing_imgs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["sem_preserving_multi_modal_inputs"] = dict(model_inputs_sem_preserving_imgs)
                row_dict["sem_changing_multi_modal_inputs"] = dict(model_inputs_sem_changing_imgs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
                row_dict["sem_preserving_multi_modal_inputs"].pop("second_per_grid_ts", None)
                row_dict["sem_changing_multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()



class RLHFDataset_counterFactual1(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        #### image modifications:
        augment= True
        self.augment = augment or config.get("return_augmented_images", False)
        # Define SEMANTICS-PRESERVING augmentations
        self.sem_preserving_transforms = [
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # T.RandomHorizontalFlip(p=1.0),
            T.RandomRotation(degrees=10),
            # T.RandomResizedCrop(224, scale=(0.9, 1.0)),
            # T.GaussianBlur(kernel_size=3),
            GaussianBlur(kernel_size=3,sigma=1),
        ]

        # Define SEMANTICS-CHANGING augmentations
        self.sem_changing_transforms = [
            T.RandomPosterize(bits=2),
            T.RandomSolarize(threshold=128),
            T.RandomInvert(p=1.0),
            T.RandomAffine(degrees=45, shear=20),
            T.RandomVerticalFlip(p=1.0),
        ]

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                    videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None
                    
                    # print(doc["extra_info"]["index"])
                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item, 
                   sem_preserve_idx: int = None, 
                   sem_change_idx: int = None, 
                   with_augmented: bool = False):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images
                # import ipdb; ipdb.set_trace()
                if self.augment or with_augmented:
                    # import ipdb; ipdb.set_trace()
                    sem_preserving_imgs, sem_changing_imgs = [], []

                    for img in images:
                        pil_img = Image.fromarray(img.numpy()) if not isinstance(img, Image.Image) else img
                        # import ipdb; ipdb.set_trace()
                        # ---- semantics-preserving ----
                        if sem_preserve_idx is not None:
                            transform = self.sem_preserving_transforms[sem_preserve_idx]
                        else:
                            transform = random.choice(self.sem_preserving_transforms)
                        sem_preserving_imgs.append(transform(pil_img))

                        # ---- semantics-changing ----
                        if sem_change_idx is not None:
                            transform = self.sem_changing_transforms[sem_change_idx]
                        else:
                            transform = random.choice(self.sem_changing_transforms)
                        sem_changing_imgs.append(transform(pil_img))
                        # import ipdb; ipdb.set_trace()
                    row_dict["sem_preserving_images"] = sem_preserving_imgs
                    row_dict["sem_changing_images"] = sem_changing_imgs

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            # import ipdb; ipdb.set_trace()
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            model_inputs_sem_preserving_imgs = self.processor(text=[raw_prompt], images=sem_preserving_imgs, videos=videos, return_tensors="pt")
            model_inputs_sem_changing_imgs = self.processor(text=[raw_prompt], images=sem_changing_imgs, videos=videos, return_tensors="pt")
            input_ids_sem_preserving_imgs = model_inputs_sem_preserving_imgs.pop("input_ids")
            input_ids_sem_changing_imgs = model_inputs_sem_changing_imgs.pop("input_ids")
            input_ids_sem_preserving_imgs = model_inputs_sem_preserving_imgs.pop("attention_mask")
            input_ids_sem_changing_imgs = model_inputs_sem_changing_imgs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")
            
            if "second_per_grid_ts" in model_inputs_sem_preserving_imgs:
                model_inputs_sem_preserving_imgs.pop("second_per_grid_ts")
                
            if "second_per_grid_ts" in model_inputs_sem_changing_imgs:
                model_inputs_sem_changing_imgs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["sem_preserving_multi_modal_inputs"] = dict(model_inputs_sem_preserving_imgs)
                row_dict["sem_changing_multi_modal_inputs"] = dict(model_inputs_sem_changing_imgs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
                row_dict["sem_preserving_multi_modal_inputs"].pop("second_per_grid_ts", None)
                row_dict["sem_changing_multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
