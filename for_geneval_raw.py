import argparse
import json
import os

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # 分布式环境初始化
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Distributed init complete. Total processes: {world_size}")

    # 加载模型
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    # 注意：确保使用 trust_remote_code=True
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    @torch.inference_mode()
    def generate(mmgpt, vl_chat_processor, prompt,
                 temperature=0.95, parallel_size=4, cfg_weight=5.0,
                 image_token_num_per_image=576, img_size=384, patch_size=16):
        
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(device)
        
        # 构造 CFG 输入 (cond + uncond)
        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.long).to(device)
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0: # Unconditional branch: replace content with pad
                tokens[i, 1:-1] = vl_chat_processor.pad_id
        
        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.long).to(device)

        past_key_values = None
        for i in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            # CFG Guidance
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            
            # 准备下一轮输入 (复制成两份用于并行计算 cond/uncond)
            next_token_expanded = torch.cat([next_token, next_token], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token_expanded)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        # 解码图像
        dec = mmgpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        
        return dec

    # 读取数据
    with open(args.metadata_path, "r") as f:
        metadata = [json.loads(line) for line in f if line.strip()]

    # 数据并行切分
    local_indices = list(range(rank, len(metadata), world_size))
    
    for idx in local_indices:
        item = metadata[idx]
        text_prompt = item["prompt"]
        save_dir = os.path.join(args.output_dir, f"{idx:0>5}")

        # 检查是否已完成 (容错重启)
        samples_dir = os.path.join(save_dir, "samples")
        if os.path.exists(samples_dir) and len(os.listdir(samples_dir)) >= 4:
            continue
        os.makedirs(samples_dir, exist_ok=True)
        with open(os.path.join(save_dir, "metadata.jsonl"), "w") as f:
            json.dump(item, f)
        
        # 格式化 Prompt
        conversation = [
            {"role": "User", "content": text_prompt},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        full_prompt = sft_format + vl_chat_processor.image_start_tag
        
        # 生成
        try:
            visual_imgs = generate(vl_gpt, vl_chat_processor, full_prompt)
            for j, img in enumerate(visual_imgs):
                PIL.Image.fromarray(img).save(os.path.join(samples_dir, f"{j}.jpg"))
            
            if rank == 0 and idx % 10 == 0:
                print(f"[Rank {rank}] Processed {idx+1}/{len(metadata)}")
        except Exception as e:
            print(f"[Rank {rank}] Error at index {idx}: {e}")

        # 定期清理显存
        if idx % 5 == 0:
            torch.cuda.empty_cache()

    dist.barrier()
    if rank == 0:
        print("All processes finished!")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()