# Synchronous Training with verl Framework

## Dataset

We have tested CPPO with two datasets: **TIGER-Lab/ViRL39K** and **hiyouga/geometry3k**. 
To train CPPO, you will first need to download the each dataset and place it in the data folder.

## Preparing the Dataset

After downloading, preprocess the dataset using the following. Set the correct data path in the following code.

```
python examples/data_preprocess/virl39k.py --local_dataset_parquet "../data/TIGER-Lab-ViRL39K/39Krelease.parquet" --local_img_folder "../data/TIGER-Lab-ViRL39K" --local_save_dir "../data/virl39k_verl"
```

## Install verl

Follow offical instruction of verl to set up the training framework.


## Train 3B model

Set the appropriate variables in the following bash file and run:

```
bash examples/cppo/run_qwen2_5_vl-3b_cppo.sh
```

## Train 7B model

Set the appropriate variables in the following bash file and run:

```
bash examples/cppo/run_qwen2_5_vl-7b_cppo.sh
```



