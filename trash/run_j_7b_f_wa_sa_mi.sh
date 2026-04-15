#!/bin/bash
#SBATCH --account=rrg-xintang
#SBATCH --job-name=janus_4x_fast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/gongzx/MM2026//scratch/gongzx/MM2026/geneval_7b_f_wa_sa_mi/SLURM-%j.out

# ================= 1. 统一变量配置 =================
BASELINE_WINDOW=16
K_JUMP=1.2
K_MEAN=1.2
JUMP_FLOOR=1.5
MEAN_FLOOR=3.0
MIN_WINDOW=3
MAX_WINDOW=12
BEAM_SIZE=3
CFG_WEIGHT=5.0

RES_DIR="/scratch/gongzx/MM2026/geneval_7b_f_wa_sa_mi/kj${K_JUMP}_km${K_MEAN}_min${MIN_WINDOW}_max${MAX_WINDOW}"
MODEL_PATH="/scratch/gongzx/MM2026/Janus-Pro-7B"
METADATA_PATH="/home/gongzx/share/MM2026/Janus/geneval/prompts/evaluation_metadata.jsonl"
# ===============================================================

set -e

# 1. 环境准备 (生成任务)
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn python/3.11
source /home/gongzx/share/MM2026/Janus/MM-Janus-pro/Janus/bin/activate

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export HF_HOME=/scratch/gongzx/.cache/huggingface
export PYTHONUNBUFFERED=1

mkdir -p $RES_DIR

cd /home/gongzx/share/MM2026/Janus/MM-Janus-pro/

torchrun --nproc_per_node=4 /home/gongzx/share/MM2026/Janus/MM-Janus-pro/for_geneval_a_f_nbatr.py\
    --model_path $MODEL_PATH \
    --metadata_path $METADATA_PATH \
    --output_dir $RES_DIR \
    --cfg_weight $CFG_WEIGHT \
    --baseline_window $BASELINE_WINDOW \
    --k_jump $K_JUMP \
    --k_mean $K_MEAN \
    --jump_floor $JUMP_FLOOR \
    --mean_floor $MEAN_FLOOR \
    --min_window_size $MIN_WINDOW \
    --max_window_size $MAX_WINDOW \
    --beam_size $BEAM_SIZE

deactivate

# 2. 环境切换 (评估任务)
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv scipy-stack python/3.11
source /home/gongzx/share/MM2026/Janus/Infinity/bin/activate
echo "Infinity 环境已就绪"

cd /home/gongzx/share/MM2026/Janus/Infinity

python -u evaluation/gen_eval/evaluate_images.py \
    $RES_DIR \
    --outfile $RES_DIR/results.jsonl \
    --model-config evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
    --model-path weights/mask2former

python -u evaluation/gen_eval/summary_scores.py \
    $RES_DIR/results.jsonl \
    > $RES_DIR/final_score.txt

echo "任务完成！结果已存至 $RES_DIR/final_score.txt"