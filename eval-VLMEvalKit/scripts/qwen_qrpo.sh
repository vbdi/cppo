export LMUData=/home/ma-user/work/.cache/LMUData

source ~/anaconda3/etc/profile.d/conda.sh

conda activate /home/ma-user/work/conda_envs/verl_evalkit
data=DynaMath #LogicVista #MathVista_MINI #DynaMath, DynaMath # 
# export CUDA_VISIBLE_DEVICES=5  
# export REASONER_AT_INFERENCE=True
cd /home/ma-user/work/ahmad/vlm/VLMEvalKit
python  run_mohsen.py \
        --data  $data \
        --model Qwen2.5-VL-3B-Instruct2 \
        --verbos
