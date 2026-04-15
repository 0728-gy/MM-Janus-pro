#!/bin/bash
#SBATCH --account=rrg-xintang
#SBATCH --job-name=dpg_7b_c_a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/gongzx/MM2026/dpg_7b_c_a/SLURM-%j.out

# ================= 1. 统一变量配置 =================
# 生成算法参数
BASELINE_WINDOW=8
K_JUMP=2
K_MEAN=2
JUMP_FLOOR=1.5
MEAN_FLOOR=2
MIN_WINDOW=3
MAX_WINDOW=12
BEAM_SIZE=3
CFG_WEIGHT=5.0


# DPG-Bench 格式参数
RESOLUTION=384
PIC_NUM=4  # 拼图模式
PROCESSES=4
PORT=29500

# 路径配置（请务必确保这些路径存在）
RES_DIR="/scratch/gongzx/MM2026/dpg_7b_c_a/kj${K_JUMP}_km${K_MEAN}_min${MIN_WINDOW}_max${MAX_WINDOW}"
MODEL_PATH="/scratch/gongzx/MM2026/Janus-Pro-7B"
DPG_CSV_PATH="/home/gongzx/share/MM2026/Janus/ELLA/dpg_bench/dpg_bench.csv"

# 脚本文件路径
GEN_SCRIPT="/home/gongzx/share/MM2026/Janus/MM-Janus-pro/for_dpg.py"
EVAL_SCRIPT="/home/gongzx/share/MM2026/Janus/ELLA/dpg_bench/compute_dpg_bench.py"



# ===============================================================

set -e
export HF_HOME=/scratch/gongzx/.cache/huggingface
export PYTHONUNBUFFERED=1
mkdir -p $RES_DIR

# ================= 2. 第一阶段：生成图像 (Janus 环境) =================
echo "========================================"
echo "阶段 1: 使用 Janus 环境生成图像..."
echo "========================================"

module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn python/3.11
source /home/gongzx/share/MM2026/Janus/MM-Janus-pro/Janus/bin/activate

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# 运行生成任务
accelerate launch --num_processes 4 --main_process_port 29500 \
    $GEN_SCRIPT \
    --model_path $MODEL_PATH \
    --csv_path $DPG_CSV_PATH \
    --output_dir $RES_DIR \
    --resolution $RESOLUTION \
    --pic_num $PIC_NUM \
    --cfg_weight $CFG_WEIGHT \
    --baseline_window $BASELINE_WINDOW \
    --k_jump $K_JUMP \
    --k_mean $K_MEAN \
    --jump_floor $JUMP_FLOOR \
    --mean_floor $MEAN_FLOOR \
    --min_window_size $MIN_WINDOW \
    --max_window_size $MAX_WINDOW \
    --beam_size $BEAM_SIZE

echo "图像生成完成！"
deactivate

# ================= 3. 第二阶段：评测图像 (DPG_bench 环境) =================
echo "========================================"
echo "阶段 2: 切换到 DPG_bench 环境进行评测..."
echo "========================================"

# 清理并加载评测环境所需的模块
module purge
module load StdEnv/2023 gcc/12.3 python/3.11 arrow/17.0.0 
source /scratch/gongzx/MM2026/DPG_bench/DPG_bench/bin/activate

# 评测任务
# 注意：即使环境切换了，之前定义的 shell 变量 ($RES_DIR, $DPG_CSV_PATH 等) 依然有效





accelerate launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT \
  $EVAL_SCRIPT \
  --image-root-path $RES_DIR \
  --resolution $RESOLUTION \
  --csv $DPG_CSV_PATH \
  --pic-num $PIC_NUM \
  --vqa-model mplug

echo "========================================"
echo "全部任务完成！"
echo "结果目录: $RES_DIR"
echo "========================================"

deactivate