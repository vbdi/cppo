# Synchronous Training with verl Framework

This training pipeline is adopted from [verl](https://github.com/verl-project/verl).

## Dataset

We train CPPO with two multimodal reasoning datasets:

- **TIGER-Lab/ViRL39K**
- **hiyouga/geometry3k**

Download datasets and place them under the `data` folder of main repository's directory.

## Preparing the Dataset

After downloading, preprocess the dataset using the following. Set the correct data path .

```
python examples/data_preprocess/virl39k.py --local_dataset_parquet "../data/TIGER-Lab-ViRL39K/39Krelease.parquet" --local_img_folder "../data/TIGER-Lab-ViRL39K" --local_save_dir "../data/virl39k_verl"
```

## Install verl

Follow offical instruction of verl to set up the training framework.


## Train 3B model

Set the proper variables in the following bash file and run:

```
bash examples/cppo/run_qwen2_5_vl-3b_virl39k.sh
```

## Train 7B model

Set the proper variables in the following bash file and run:

```
bash examples/cppo/run_qwen2_5_vl-7b_virl39k.sh
```