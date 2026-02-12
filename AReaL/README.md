# Asynchronous Training with AReaL Framework

This training pipeline is adopted from [AReaL](https://github.com/inclusionAI/AReaL).

## Dataset

We train CPPO with two multimodal reasoning datasets:

- **TIGER-Lab/ViRL39K**
- **hiyouga/geometry3k**

Download datasets and place them under the `data` folder of main repository's directory.

## Install AReaL

Follow offical instruction of AReaL to set up the training framework environment.

## Train 3B model

Set the proper variables in the following bash file and run:

```
bash examples/cppo/run_qwen2_5_vl-3b_geometry3k.sh
```