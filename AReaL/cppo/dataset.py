"""CPPO Geometry3K Dataset with semantic image augmentations."""

from io import BytesIO
from typing import Any

from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torchvision import transforms
import io
import random
import math
from math import ceil
from typing import Union

import numpy as np
from PIL import Image, ImageDraw
import json
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image


class RandomOcclusion:
    """Apply random occlusion with noise patches to an image."""
    
    def __init__(self, p=0.5, scale=(0.05, 0.15), iter=10):
        """Initialize random occlusion.
        
        Args:
            p: Probability of applying occlusion
            scale: Tuple of (min, max) for occlusion patch size as fraction of image
            iter: Number of patches to apply
        """
        self.p = p
        self.scale = scale
        self.iter = iter

    def __call__(self, img):
        """Apply random occlusion to image."""
        if random.random() > self.p:
            return img
        img = img.copy()  # Make a copy to avoid modifying original
        w, h = img.size
        for ii in range(self.iter):
            occ_w = int(random.uniform(*self.scale) * w)
            occ_h = int(random.uniform(*self.scale) * h)
            x = random.randint(0, w - occ_w)
            y = random.randint(0, h - occ_h)

            # Random noise patch
            noise = np.uint8(np.random.rand(occ_h, occ_w, 3) * 255)
            patch = Image.fromarray(noise, mode="RGB")
            img.paste(patch, (x, y))
        return img


class RandomZoomCrop:
    """Random zoom and crop augmentation."""
    
    def __init__(self, scale=(0.8, 1.0)):
        """Initialize random zoom crop.
        
        Args:
            scale: Tuple of (min, max) fraction of original area to crop
        """
        self.scale = scale

    def __call__(self, img):
        """Apply random zoom crop to image."""
        w, h = img.size
        # Choose a random crop size
        scale_factor = random.uniform(*self.scale)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)

        # Choose a random top-left corner for the crop
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)

        # Crop the image
        cropped = img.crop((left, top, left + new_w, top + new_h))

        # Resize back to original size
        zoomed = cropped.resize((w, h), Image.BILINEAR)
        return zoomed


class BlackSquareCover:
    """Cover image regions with black squares."""
    
    def __init__(
        self,
        patch_size: int = 14,
        coverage: Union[float, int] = 0.8,
        seed: Union[int, None] = 42,
        max_trials: int = 1_000_000,
    ):
        """Initialize black square cover.
        
        Parameters
        ----------
        patch_size : int
            Side length of each square patch (in pixels).
        coverage : float or int
            Target fraction (0..1) or percentage (0..100) of total pixels to cover.
        seed : int or None
            Random seed for reproducibility.
        max_trials : int
            Safety cap on the number of random placements to attempt.
        """
        self.patch_size = int(patch_size)
        self.coverage = float(coverage)
        self.seed = seed
        self.max_trials = max_trials

    def __call__(self, img: Image.Image):
        """Apply black squares to cover the target percentage of the image."""
        # Normalize coverage to [0, 1]
        cov = self.coverage
        if cov > 1.0:
            cov = cov / 100.0
        cov = max(0.0, min(1.0, cov))

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        img = img.convert("RGB")
        W, H = img.size
        s = self.patch_size
        if s <= 0:
            raise ValueError("patch_size must be a positive integer")

        total_pixels = W * H
        target_pixels = int(ceil(cov * total_pixels))
        if target_pixels == 0:
            return img

        covered = np.zeros((H, W), dtype=bool)
        draw = ImageDraw.Draw(img)
        n_patches = 0
        trials = 0

        def place_patch(x: int, y: int):
            x2 = min(x + s, W)
            y2 = min(y + s, H)
            draw.rectangle([x, y, x2 - 1, y2 - 1], fill=(0, 0, 0))
            covered[y:y2, x:x2] = True

        while covered.sum() < target_pixels and trials < self.max_trials:
            x = random.randint(0, max(0, W - 1))
            y = random.randint(0, max(0, H - 1))
            place_patch(x, y)
            n_patches += 1
            trials += 1

        actual_coverage = covered.mean()
        return img


class GaussianBlur:
    """Gaussian blur transform using PyTorch."""
    
    def __init__(self, kernel_size=11, sigma=1.5):
        """Initialize Gaussian blur.
        
        Args:
            kernel_size: Size of the blur kernel
            sigma: Standard deviation of the Gaussian kernel
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(sigma, (int, float)):
            sigma = (sigma, sigma)
        self.sigma = sigma

    def _make_kernel_1d(self, kernel_size, sigma, ref_tensor):
        """Build 1D Gaussian kernel."""
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

def convert_image(
    image: dict[str, Any] | ImageObject | str,
    min_size: int | None =28,
    max_pixels: int | None = 512,
    augment=None,
) -> ImageObject:
    """Convert and augment an image.
    
    Args:
        image: PIL Image or image object
        min_size: Minimum image size
        max_pixels: Maximum number of pixels (resize if exceeded)
        augment: Optional augmentation transform to apply
        
    Returns:
        Processed PIL Image
    """

    width, height = image.size
    # Check if both sides already >= min_size
    if max(width, height) < min_size:

        # Compute scale factor to make the smaller side equal to min_size
        scale = max(min_size / width, min_size / height)

        # Calculate new dimensions
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        # Resize while preserving aspect ratio
        image = image.resize((new_width, new_height), Image.LANCZOS)

    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = (
            int(image.width * resize_factor),
            int(image.height * resize_factor),
        )
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    if augment:
        try:
            image = augment(image)
        except Exception as e:
            print(f'Error during augmentation: {e}')
    return image

def get_virl39k_cppo_rl_dataset(
    path: str,
    split: str,
    img_folder_path: str,
    processor,
    max_length: int | None = None,
    sem_preserving_transforms=None,
    sem_changing_transforms=None
):
    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    dataset = load_dataset("parquet", data_files=path, split=split)

    # Define SEMANTICS-PRESERVING augmentations (color/perspective changes)
    if sem_preserving_transforms is None:
        sem_preserving_transforms = [
            transforms.ColorJitter(brightness=(0.2, 1.3), contrast=(0.2, 1.8), saturation=(0.2, 1.8)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomRotation(degrees=10),
            GaussianBlur(kernel_size=3, sigma=1),
        ]
    
    # Define SEMANTICS-CHANGING augmentations (structural changes)
    if sem_changing_transforms is None:
        sem_changing_transforms = [
            RandomOcclusion(p=1.0, iter=50),
            RandomZoomCrop(scale=(0.6, 0.7)),
        ]
    
    def process(example):
        problem = example["question"]
        if "<image>" not in problem:
                problem = "<image>\n" + problem
        prompt = problem + " " + instruction_following
        answer = extract_boxed_content(example["answer"])

        img_path = [os.path.join(img_folder_path, img) for img in example["image"]]

        processed_images = [
            convert_image(Image.open(path_), min_size=28, max_pixels=320*320) for path_ in img_path
        ]

        # Semantic-preserving augmentations
        transform1 = random.choice(sem_preserving_transforms)
        processed_images_sem_preserving = [
            convert_image(Image.open(path_), min_size=28, max_pixels=320*320, augment=transform1) for path_ in img_path
        ]
        
        # Semantic-changing augmentations
        transform2 = random.choice(sem_changing_transforms)
        processed_images_sem_changing = [
            convert_image(Image.open(path_), min_size=28, max_pixels=320*320, augment=transform2) for path_ in img_path
        ]
        
        image_processor_type = processor.image_processor.image_processor_type.lower()
        if "qwen" in image_processor_type:
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        elif "gemma3" in image_processor_type:
            image_token = processor.boi_token
        else:
            image_token = processor.image_token if processor is not None else "<image>"

        messages = [
            {
                "role": "user",
                "content": prompt.replace("<image>", image_token)
            }
        ]

        messages_chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ""}},
                    {
                        "type": "text",
                        "text": prompt.replace("<image>", ""),
                    },
                ],
            }
        ]

        # Apply chat template
        messages = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        return {
            "messages": messages,
            "messages_chat": messages_chat,
            "images": processed_images,
            "answer": answer,
            "images_sem_changing": processed_images_sem_changing,
            "images_sem_preserving": processed_images_sem_preserving,
        }

    dataset = dataset.map(process).remove_columns([
                                    "question",
                                    "PassRate_32BTrained",
                                    "PassRate_7BBase",
                                    "category",
                                    "source",
                                    "qid",
                                    "image",
                                ])

    # Filter out sequences longer than max_length if max_length is provided
    if max_length is not None:

        def filter_length(sample):
            # Process the sample to get the total token count including image tokens
            processed_input = processor(
                text=[sample["messages"]],
                images=sample["images"],
                padding=False,
                return_tensors="pt",
                return_length=True,
                return_attention_mask=False,
            )
            total_tokens = len(processed_input["input_ids"].squeeze(0))
            return total_tokens <= max_length

        dataset = dataset.filter(filter_length)

    return dataset


def get_geometry3k_cppo_rl_dataset(
    path: str,
    split: str,
    processor,
    max_length: int | None = None,
    sem_preserving_transforms=None,
    sem_changing_transforms=None
):
    """Load Geometry3K dataset with CPPO semantic image augmentations.
    
    Args:
        path: Path to the dataset
        split: Dataset split (train/test)
        processor: Image processor/tokenizer
        max_length: Maximum sequence length
        sem_preserving_transforms: Transforms that preserve semantics
        sem_changing_transforms: Transforms that change semantics
        
    Returns:
        Processed dataset with semantic image variants
    """
    dataset = load_dataset(path=path, split=split)

    # Define SEMANTICS-PRESERVING augmentations (color/perspective changes)
    if sem_preserving_transforms is None:
        sem_preserving_transforms = [
            transforms.ColorJitter(brightness=(0.2, 1.3), contrast=(0.2, 1.8), saturation=(0.2, 1.8)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomRotation(degrees=10),
            GaussianBlur(kernel_size=3, sigma=1),
        ]
    
    # Define SEMANTICS-CHANGING augmentations (structural changes)
    if sem_changing_transforms is None:
        sem_changing_transforms = [
            RandomOcclusion(p=1.0, iter=50),
            RandomZoomCrop(scale=(0.6, 0.7)),
        ]

    def process(sample):
        """Process a single sample with semantic augmentations."""
        # Original images
        processed_images = [
            convert_image(image, min_size=28, max_pixels=448*448) for image in sample["images"]
        ]

        # Semantic-preserving augmentations
        transform1 = random.choice(sem_preserving_transforms)
        processed_images_sem_preserving = [
            convert_image(image, min_size=28, max_pixels=448*448, augment=transform1) for image in sample["images"]
        ]

        # Semantic-changing augmentations
        transform2 = random.choice(sem_changing_transforms)
        processed_images_sem_changing = [
            convert_image(image, min_size=28, max_pixels=448*448, augment=transform2) for image in sample["images"]
        ]

        # Determine image token based on processor type
        image_processor_type = processor.image_processor.image_processor_type.lower()
        if "qwen" in image_processor_type:
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        elif "gemma3" in image_processor_type:
            image_token = processor.boi_token
        else:
            image_token = processor.image_token if processor is not None else "<image>"
        
        # Create instruction following prompt
        instruction_following = (
            r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            r"The reasoning process MUST BE enclosed within <think> </think> tags. "
            r"The final answer MUST BE put in \boxed{}."
        )
        
        # Prepare messages for chat template
        messages = [
            {
                "role": "user",
                "content": sample["problem"]
                .replace("<image>", image_token)
                .replace("different", "")
                + instruction_following,
            }
        ]
        
        # Messages for chat API (without inline image token)
        messages_chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ""}},
                    {
                        "type": "text",
                        "text": sample["problem"]
                        .replace("<image>", "")
                        .replace("different", "")
                        + instruction_following,
                    },
                ],
            }
        ]

        # Apply chat template
        messages = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        
        return {
            "messages": messages,
            "images": processed_images,
            "messages_chat": messages_chat,
            "images_sem_changing": processed_images_sem_changing,
            "images_sem_preserving": processed_images_sem_preserving,
        }

    dataset = dataset.map(process).remove_columns(["problem"])

    # Filter out sequences longer than max_length if specified
    if max_length is not None:
        def filter_length(sample):
            # Process the sample to get the total token count including image tokens
            processed_input = processor(
                text=[sample["messages"]],
                images=sample["images"],
                padding=False,
                return_tensors="pt",
                return_length=True,
                return_attention_mask=False,
            )
            total_tokens = len(processed_input["input_ids"].squeeze(0))
            return total_tokens <= max_length

        dataset = dataset.filter(filter_length)

    return dataset