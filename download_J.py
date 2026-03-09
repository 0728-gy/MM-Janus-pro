import os

# 1. 强制设置镜像站环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 如果之前安装了 hf_transfer，开启它可以提速；没装也没关系
try:
    import hf_transfer
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    print("已启用 hf_transfer 加速")
except ImportError:
    print("未安装 hf_transfer，将使用默认下载模式")

from huggingface_hub import snapshot_download

# 3. 配置下载参数
repo_id = "deepseek-ai/Janus-Pro-7B"
local_dir = "/share/home/u11154/JingyiLiu/MM2026/Janus/model_weights"

print(f"开始下载 {repo_id} 到 {local_dir}...")

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, # 别用软链接，直接下到文件夹里
        revision="main",              # 明确指定分支
        ignore_patterns=["*.png", "*.md"] # 过滤掉没用的文件
    )
    print("\n✅ 下载成功！")
except Exception as e:
    print(f"\n❌ 下载失败！错误原因: \n{e}")
    print("\n--- 调试建议 ---")
    print("1. 如果报错是 'Connection Error'，说明 logini03 节点没有外网访问权限。")
    print("2. 尝试执行: curl -I https://hf-mirror.com 检查网络。")