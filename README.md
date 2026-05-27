<div align="center">

# **CPPO: Contrastive Perception Policy Optimization for VLM Agents**

</div>

<div align="center">
<img src="./doc/images/teaser.png" alt="CPPO overview" width="800"/>
</div>


This repository contains the description and implementation of CPPO, a reinforcement learning framework for finetuning vision–language models (VLMs).

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.00501)
[![Hugging Face Collection](https://img.shields.io/badge/HuggingFace-Collection-yellow?logo=huggingface)](https://huggingface.co/collections/vbdai/cppo)

</div>

## 🚀 Highlights

- ✨ **Contrastive Perception Policy Optimization (CPPO)** — A framework for improving vision–language policy reinforcement learning via contrastive perception training.
- 📈 **Stronger Empirical Performance** — Demonstrates consistent gains on complex multimodal reasoning tasks.
- 🔍 **Entropy-Based Perception Token Detection** — Automatically locates informative visual tokens through perturbation sensitivity.
- 📊 **Contrastive Perception Loss (CPL)** — Encourages the policy to gain discriminative perception.
- 🧠 **No External Supervision** — Perception improvement is gained purely from information-removing and information-preserving augmentations without the use of ground-truth visual information.
- ⚡ **Research-Ready Implementation** — Includes preprocessing, training, and evaluation pipelines.

## Methodology

### 1. Entropy-Based Perception Token Detection

For each generated response, CPPO identifies perception tokens by measuring the increase in predictive entropy when the input image is replaced with an information-removing perturbation. Tokens with the largest entropy increase are selected as perception-dependent tokens. This process:

- Requires no external supervision

- Is fully model-driven

- Preserves the natural reasoning structure of the VLM

### 2. Contrastive Perception Loss (CPL)

For each detected perception token, CPPO applies a token-level contrastive loss:

- Anchor: token distribution conditioned on the original image
- Positive: distribution conditioned on an information-preserving perturbation
- Negative: distribution conditioned on an information-removing perturbation

### 3. Integration with Reinforcement Learning

CPPO augments the standard RL objective with the Contrastive Perception Loss:

- CPL is applied only to perception tokens
- CPL is gated by positive advantage, ensuring it reinforces successful trajectories

This design yields targeted perception improvement while maintaining RL stability.

<div align="center">
<img src="./doc/images/method.png" alt="CPPO methodology" width="800"/>
</div>

### Main Results
CPPO is evaluated on a wide range of multimodal reasoning benchmarks and consistently improves the baseline RL objective.

<div align="center">
<img src="./doc/images/results.png" alt="CPPO results" width="600"/>
</div>

# Pretrained Models

We provide CPPO-trained checkpoints on HuggingFace.

| Model | Training Dataset | HuggingFace Link |
|-------|------------------|------------------|
| CPPO-3B | VIRL-39K | [vbdai/CPPO-3B](https://huggingface.co/vbdai/CPPO-3B) |
| CPPO-7B | VIRL-39K | [vbdai/CPPO-7B](https://huggingface.co/vbdai/CPPO-7B) |

You can load these models using the Hugging Face `transformers` library:

```python
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Load model and processor
model_name = "path/to/cppo-3B"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# Instruction template
instruction_following = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \boxed{}."
)

# Prepare prompt with instruction following
prompt = "Your question here. " + instruction_following
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": Image.open("path/to/image.jpg"),
            },
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

# Generate output
inputs = processor(text=[text], images=image_inputs, padding=True, returvaluationn_tensors="pt").to("cuda")
outputs = model.generate(**inputs,ninstruction to evaluate each checkopoint und with `avg@N` for different benchmarks m
### An Example Workflow
ax_new_tokens=4096)
generated_ids = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
]
response = processor.decode(generated_ids[0], skip_special_tokens=True)
print(response)
```

# Training
We provide implementations of CPPO for both **synchronous** and **asynchronous** training regimes.  
Each variant builds on a widely used large-scale RL framework while adding the CPPO implementation.

## Training with CPPO in Synchronous Settings
The synchronous pipeline is built on top of [**verl**](https://github.com/verl-project/verl), where rollout generation and training are synchronized.

## Training with CPPO in Asynchronous Settings
For higher throughput and improved hardware utilization, CPPO is also integrated with [**AReaL**](https://github.com/inclusionAI/AReaL), which decouples generation and training resources.

## Quick Start
To train models such as `Qwen2.5-VL-3B` or `Qwen2.5-VL-7B`:

1. Navigate to the `verl` or `AReaL` directory.
2. Follow the provided environment and dataset setup instructions.
3. Launch training using the CPPO examples.

```bash
### For synchronous training
cd verl
bash examples/cppo/run_qwen2_5_vl-3b_virl39k.sh
```

```bash
### For asynchronous training
cd AReaL
bash examples/cppo/run_qwen2_5_vl-3b_geometry3k.sh
```

# Evaluation

For evaluating models on multimodal reasoning benchmarks, we provide an evaluation pipeline based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## Quick Start with Evaluation

Navigate to the evaluation directory and follow the instruction to setup the framework and evaluate each checkpoint (`avg@N`).  

### An Example Workflow

```bash
cd eval-VLMEvalKit

# Step 1: Run inference
bash run_eval.sh

# Wait for inference to complete...

# Step 2: Find the session name
SESSION_NAME=$(ls -dt ./outputs/Qwen2.5-VL-3B-Instruct/eval_num_g/Qwen2.5-VL-3B-Instruct/T* | head -1 | xargs basename)
echo "Session: $SESSION_NAME"

# Step 3: Post-process group results
python postprocess_group_results.py \
    ./outputs \
    Qwen2.5-VL-3B-Instruct \
    eval_num_g \
    $SESSION_NAME

# Step 4: Generate performance summary
python create_performance_summary.py \
    ./outputs/Qwen2.5-VL-3B-Instruct \
    $SESSION_NAME

# View results
cat ./eval_summary/Qwen2.5-VL-3B-Instruct.csv
```

## Citation

If you find this work useful, please consider giving us a star and citing our work.
```
@article{rezaei2026cppo,
    title={CPPO: Contrastive Perception for Vision Language Policy Optimization},
    author={Rezaei, Ahmad and Gholami, Mohsen and Ranjbar Alvar, Saeed and Cannons, Kevin and Hossain, Mohammad Asiful and Weimin, Zhou and Zhou, Shunbo and Zhang, Yong and Akbari, Mohammad},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2026}
}
```

