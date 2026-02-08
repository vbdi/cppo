### Preprocessing TIGER-Lab/ViRL39K dataset

import argparse
import os

import datasets
from PIL import Image
from verl.utils.hdfs_io import copy, makedirs
from mathruler.grader import extract_boxed_content

def resize_min_side(image, min_size=28):
    width, height = image.size
    # Check if both sides already >= min_size
    if min(width, height) >= min_size:
        return image  # No resizing needed

    # Compute scale factor to make the smaller side equal to min_size
    scale = max(min_size / width, min_size / height)

    # Calculate new dimensions
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    # Resize while preserving aspect ratio
    return image.resize((new_width, new_height), Image.LANCZOS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_parquet", default="../data/TIGER-Lab-ViRL39K/39Krelease.parquet", help="Path to the parquet file")
    parser.add_argument("--local_img_folder", default="../data/TIGER-Lab-ViRL39K")
    parser.add_argument(
        "--local_save_dir", default="../data/virl39k_verl", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "TIGER-Lab/ViRL39K"

    train_dataset = datasets.load_dataset("parquet", data_files=local_dataset_path, split="train")

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )
    
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example["question"]
            if "<image>" not in problem:
                problem = "<image>\n" + problem
            prompt = problem + " " + instruction_following
            answer = extract_boxed_content(example["answer"])

            img_path = [os.path.join(args.local_img_folder, img) for img in example["image"]]
            images = [resize_min_side(Image.open(path)) for path in img_path]

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
