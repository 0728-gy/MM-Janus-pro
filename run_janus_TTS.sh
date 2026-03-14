#!/bin/bash
#SBATCH --account=rrg-xintang
#SBATCH --job-name=janus_4x_fast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gongzx/geneval_results/janus_pro_7b_1/%j.out

# ================= 1. 统一变量配置 (修改这里即可) =================
# 结果输出主目录
RES_DIR="/scratch/gongzx/geneval_results/janus_pro_7b_1" #记得要改上面的！！！！
# 模型路径
MODEL_PATH="/scratch/gongzx/models/Janus-Pro-7B"
# 提示词元数据路径
METADATA_PATH="/home/gongzx/share/MM2026/Janus/geneval/prompts/evaluation_metadata.jsonl"
# ===============================================================

# 开启错误即停止
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

# 3. 运行程序
cd /home/gongzx/share/MM2026/Janus/MM-Janus-pro/

# 修正：torchrun 的参数是 --nproc_per_node，而不是 --ntasks
torchrun --nproc_per_node=4 for_geneval.py \
    --model_path $MODEL_PATH \
    --metadata_path $METADATA_PATH \
    --output_dir $RES_DIR

deactivate

# 4. 环境切换 (评估任务)
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv scipy-stack python/3.11
source /home/gongzx/share/MM2026/Janus/Infinity/bin/activate
echo "Infinity 环境已就绪"

cd /home/gongzx/share/MM2026/Janus/Infinity

# 第一步：目标检测打分 (加上 -u 确保 log 实时刷新)
python -u evaluation/gen_eval/evaluate_images.py \
    $RES_DIR \
    --outfile $RES_DIR/results.jsonl \
    --model-config evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
    --model-path weights/mask2former

# 第二步：汇总分数 (使用变量并生成记录文件)
python -u evaluation/gen_eval/summary_scores.py \
    $RES_DIR/results.jsonl \
    > $RES_DIR/final_score.txt

echo "任务完成！结果已存至 $RES_DIR/final_score.txt"