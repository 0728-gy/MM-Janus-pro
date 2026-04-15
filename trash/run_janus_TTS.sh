#!/bin/bash
#SBATCH --account=rrg-xintang
#SBATCH --job-name=janus_4x_fast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/gongzx/MM2026/geneval_results/SLURM-%j.out  # 修改了拼写 SLUM -> SLURM-

# ================= 1. 统一变量配置 =================
SIGMA_SUM=46
D=6
SIGMA_SINGLE=3.5
MODEL_SUM=1
MODEL_SINGLE=0
RES_DIR="/scratch/gongzx/MM2026/geneval_results_new_entropy/janus_pro_7b_sigma_sum_${SIGMA_SUM}_d_${D}_sigma_sin${SIGMA_SINGLE}_mode_su${MODEL_SUM}_sin${MODEL_SINGLE}"
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

# 【修复】：去掉了 --model_sum 这一行末尾的 \
torchrun --nproc_per_node=4 for_geneval.py \
    --model_path $MODEL_PATH \
    --metadata_path $METADATA_PATH \
    --output_dir $RES_DIR \
    --sigma_sum $SIGMA_SUM \
    -d $D \
    --sigma_single $SIGMA_SINGLE \
    --model_single $MODEL_SINGLE \
    --model_sum $MODEL_SUM

deactivate

# 4. 环境切换 (评估任务)
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