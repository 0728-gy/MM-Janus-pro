import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import torch.nn.functional as F
import PIL.Image
import math

from transformers import DynamicCache

import os
import json
import argparse
from tqdm import tqdm
import pandas as pd  # 新增 pandas 用于处理 DPG-Bench CSV

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID", 0)))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
torch.cuda.set_device(LOCAL_RANK)
DEVICE = f"cuda:{LOCAL_RANK}"

# ============================================================
# 核心生成函数（自适应窗口 + Beam Search 回溯）
# ============================================================

def expand_cache_for_beam(past_key_values, beam_size):
    new_cache = DynamicCache()
    for layer_idx in range(len(past_key_values)):
        k, v = past_key_values[layer_idx]
        new_k = k.repeat(beam_size, 1, 1, 1)
        new_v = v.repeat(beam_size, 1, 1, 1)
        new_cache.update(new_k, new_v, layer_idx)
    return new_cache


@torch.inference_mode()
def generate_single_image(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    baseline_window: int = 8,
    k_jump: float = 2,
    k_mean: float = 2,
    jump_floor: float = 1.5,
    mean_floor: float = 3.0,
    min_window_size: int = 3,
    max_window_size: int = 8,
    beam_size: int = 3,
    temperature: float = 0.95,
) -> np.ndarray:
    _device = next(mmgpt.parameters()).device

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).to(_device)

    tokens = torch.stack([input_ids.clone(), input_ids.clone()], dim=0)
    tokens[1, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    outputs = mmgpt.language_model.model(
        inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None
    )
    past_key_values = outputs.past_key_values
    last_hidden_state = outputs.last_hidden_state[:, -1, :]

    generated_tokens = []
    token_entropies = []
    delta_entropies = []
    history_hidden_states = [last_hidden_state]

    t_fast = 0
    t_slow = 0

    while len(generated_tokens) < image_token_num_per_image:

        # ----------------------
        # A. 预测下一个 Token
        # ----------------------
        logits = mmgpt.gen_head(last_hidden_state)
        logit_cond   = logits[0, :]
        logit_uncond = logits[1, :]
        final_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        probs       = torch.softmax(final_logits / temperature, dim=-1)
        probs_for_e = torch.softmax(logit_cond/ temperature, dim=-1)
        log_probs   = torch.log_softmax(logit_cond / temperature, dim=-1)
        entropy     = -torch.sum(probs_for_e * log_probs, dim=-1).item()

        next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens.append(next_token.item())
        token_entropies.append(entropy)
        t_fast = len(generated_tokens)

        delta_entropies.append(0.0 if t_fast == 1 else abs(token_entropies[-1] - token_entropies[-2]))

        # ----------------------
        # B. 更新 KV cache
        # ----------------------
        next_token_input = next_token.repeat(2)
        img_embeds    = mmgpt.prepare_gen_img_embeds(next_token_input)
        inputs_embeds = img_embeds.unsqueeze(1)

        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        history_hidden_states.append(last_hidden_state)

        # 【修改 1】：引入网格宽度 W，切断跨行 Delta 计算
        W = 24  # 对于 576 像素的网格，宽度固定为 24
        curr_col = (t_fast - 1) % W
        
        # 如果是整张图的第一个 token，或者是每一行的第一个 token，Delta 归零
        if t_fast == 1 or curr_col == 0:
            delta_entropies.append(0.0)
        else:
            delta_entropies.append(abs(token_entropies[-1] - token_entropies[-2]))

        window_size = t_fast - t_slow

        # ============================================================
        # C. 2D-Aware 判定机制 (含：2D预热、行内分割、行尾审计)
        # ============================================================
        trigger_backtrack = False
        d = 0
        beam_d = 0   

        W = 24  # 网格宽度
        curr_idx = t_fast - 1
        curr_row = curr_idx // W
        curr_col = curr_idx % W
        
        # 统计参数
        baseline_window = 8 
        window_size = t_fast - t_slow

        # --- 1. 构建 2D 统计基准 (L型邻域预热) ---
        base_ents = []
        base_deltas = []
        num_horiz = min(curr_col, baseline_window)
        if num_horiz > 0:
            base_ents = token_entropies[curr_idx - num_horiz : curr_idx]
            base_deltas = delta_entropies[curr_idx - num_horiz : curr_idx]
        if len(base_ents) < baseline_window and curr_row > 0:
            needed = baseline_window - len(base_ents)
            top_end_idx = (curr_row - 1) * W + curr_col + 1
            top_start_idx = max((curr_row - 1) * W, top_end_idx - needed)
            if top_end_idx > top_start_idx:
                base_ents = token_entropies[top_start_idx : top_end_idx] + base_ents
                base_deltas = delta_entropies[top_start_idx : top_end_idx] + base_deltas

        # --- 2. 开始核心逻辑判定 ---
        if len(base_ents) >= 3:
            # 计算动态阈值
            b_mean = sum(base_ents) / len(base_ents)
            b_std  = math.sqrt(sum((x - b_mean)**2 for x in base_ents) / len(base_ents))
            bd_mean = sum(base_deltas) / len(base_deltas)
            bd_std  = math.sqrt(sum((x - bd_mean)**2 for x in base_deltas) / len(base_deltas))

            jump_threshold = min(max(jump_floor, bd_mean + k_jump * bd_std), 5.0)
            mean_threshold = min(max(mean_floor, b_mean  + k_mean * b_std),  8.0)

            current_delta = delta_entropies[-1]
            avg_window_ent = sum(token_entropies[t_slow:t_fast]) / max(1, window_size)

            # --- 场景 A: 突发惊吓 (立即拦截) ---
            if current_delta > jump_threshold and window_size >= min_window_size:
                if avg_window_ent > mean_threshold:
                    trigger_backtrack = True
                    d = window_size
                    print(f"[{t_fast}] 突发回溯! Δ:{current_delta:.2f} > {jump_threshold:.2f}")
                else:
                    # 如果 Δ 很大但均值 OK，说明可能只是正常的细节增加
                    # 我们可以选择提前步进 t_slow，防止窗口变大
                    t_slow = t_fast 

            # --- 场景 B: 到达行尾 (强制清算，不准漏判) ---
            elif curr_col == W - 1:
                if avg_window_ent > mean_threshold:
                    trigger_backtrack = True
                    d = window_size
                    print(f"[{t_fast}] 行尾审计不合格! AvgEnt:{avg_window_ent:.2f}")
                else:
                    # 行尾合格，清空慢指针，迎接下一行
                    t_slow = t_fast
                    print(f"[{t_fast}] 行尾审计合格，重置指针。")

            # --- 场景 C: 窗口满 (基于审计的差分提交) ---
            elif window_size >= max_window_size:
                # 1. 寻找分割点 (寻找最不安分的瞬间)
                window_deltas = delta_entropies[t_slow:t_fast]
                max_local_idx = max(range(len(window_deltas)), key=lambda i: window_deltas[i])
                split_pos = t_slow + max_local_idx + 1
                
                # 2. 【核心改进】：对即将被“提交”的前半段进行专项审计
                first_half_ents = token_entropies[t_slow : split_pos]
                avg_first_half_ent = sum(first_half_ents) / len(first_half_ents)
                
                # 3. 审计判定
                # 如果连准备提交的前半段都超过了平均阈值，说明这 12 个像素整体都不可信
                if avg_first_half_ent > mean_threshold:
                    trigger_backtrack = True
                    d = beam_d = window_size
                    print(f"[{t_fast}] 窗口分割审计失败! 前半段 AvgEnt:{avg_first_half_ent:.2f} > {mean_threshold:.2f}。执行全窗口回溯。")
                else:
                    # 只有审计通过，才允许“落袋为安”
                    t_slow = split_pos
                    print(f"[{t_fast}] 窗口分割审计合格 (前半段 AvgEnt:{avg_first_half_ent:.2f})。提交前半段，当前指针步进至 {t_slow}。")
        else:
            # 极初期（整张图最左上角）没有预热样本时
            if t_fast <= baseline_window:
                t_slow = t_fast

        
        # ----------------------
        # F. 回溯 + 平行随机采样 + 前瞻最低熵选择 (Lookahead Entropy Selection)
        # ----------------------
        if trigger_backtrack:
            if len(history_hidden_states) <= d:
                print("警告：历史不足，跳过回溯")
            else:
                generated_tokens = generated_tokens[:-d]
                token_entropies  = token_entropies[:-d]
                delta_entropies  = delta_entropies[:-d]

                current_len = past_key_values.get_seq_length()
                past_key_values.crop(current_len - d)
                history_hidden_states = history_hidden_states[:-d]
                start_hidden_state    = history_hidden_states[-1]

                # 复制 KV Cache 给所有的候选分支
                beam_hidden_state = start_hidden_state.repeat(beam_size, 1)
                past_key_values   = expand_cache_for_beam(past_key_values, beam_size)

                beam_seqs   = torch.zeros((beam_size, 0), dtype=torch.long, device=_device)
                beam_history_hiddens = [beam_hidden_state]
                beam_entropies = torch.zeros((beam_size, 0), dtype=torch.float, device=_device)

                # 1. 独立并行 Rollout d 步
                for step in range(d):
                    curr_h = beam_history_hiddens[-1]
                    b_logits = mmgpt.gen_head(curr_h)

                    b_logits_view  = b_logits.view(beam_size, 2, -1)
                    b_cond         = b_logits_view[:, 0, :]
                    b_uncond       = b_logits_view[:, 1, :]
                    
                    b_final_logits = (b_uncond + cfg_weight * (b_cond - b_uncond)).to(torch.float32)

                    b_probs       = torch.softmax(b_final_logits / temperature, dim=-1)
                    b_probs_e     = torch.softmax(b_cond  / temperature, dim=-1)
                    b_log_probs   = torch.log_softmax(b_cond  / temperature, dim=-1)
                    b_entropies_step = -torch.sum(b_probs_e * b_log_probs, dim=-1)

                    if torch.isnan(b_probs).any():
                        token_indices = torch.argmax(b_final_logits, dim=-1)
                    else:
                        token_indices = torch.multinomial(b_probs, num_samples=1).squeeze(-1)

                    beam_seqs = torch.cat([beam_seqs, token_indices.unsqueeze(1)], dim=1)
                    beam_entropies = torch.cat([beam_entropies, b_entropies_step.unsqueeze(1)], dim=1)

                    next_tokens_input = token_indices.repeat_interleave(2)
                    img_embeds = mmgpt.prepare_gen_img_embeds(next_tokens_input).unsqueeze(1)

                    b_outputs = mmgpt.language_model.model(
                        inputs_embeds=img_embeds,
                        use_cache=True,
                        past_key_values=past_key_values, 
                    )
                    new_h = b_outputs.last_hidden_state[:, -1, :]
                    beam_history_hiddens.append(new_h)

                # ========================================================
                # 2. 【核心创新：前瞻一个 Token 的熵】
                # 取出经过 d 步进化后的最终 Hidden State，计算即将生成的下一个 Token 的分布
                # ========================================================
                final_h = beam_history_hiddens[-1]
                lookahead_logits = mmgpt.gen_head(final_h)
                
                l_logits_view = lookahead_logits.view(beam_size, 2, -1)
                l_cond        = l_logits_view[:, 0, :]
                l_uncond      = l_logits_view[:, 1, :]
                l_final_logits = (l_uncond + cfg_weight * (l_cond - l_uncond)).to(torch.float32)
                
                l_probs       = torch.softmax(l_cond   / temperature, dim=-1)
                l_log_probs   = torch.log_softmax(l_cond  / temperature, dim=-1)
                lookahead_entropies = -torch.sum(l_probs * l_log_probs, dim=-1) # shape: (beam_size,)

                # 选择前瞻熵【最低】的分支，意味着该分支创造了模型最自信的上下文
                best_idx = torch.argmin(lookahead_entropies).item()
                
                if t_fast % 20 == 0 or True: # 可选：打印查看前瞻熵的差异
                    print(f"  -> 分支前瞻熵: {[round(e, 2) for e in lookahead_entropies.tolist()]}, 选择分支: {best_idx}")

                # ========================================================

                # 3. 提取最优分支并回填状态
                winner_tokens = beam_seqs[best_idx].tolist()
                generated_tokens.extend(winner_tokens)

                winner_entropies = beam_entropies[best_idx].tolist()
                first_delta = abs(winner_entropies[0] - token_entropies[-1]) if len(token_entropies) > 0 else 0.0
                token_entropies.extend(winner_entropies)
                delta_entropies.extend(
                    [first_delta] + [abs(winner_entropies[i] - winner_entropies[i-1]) for i in range(1, d)]
                )

                # 注：我们*不*把 lookahead_entropies 存进历史，因为外层主 while 循环
                # 下一步自然会基于我们选出的上下文去真实预测这个 token 并记录它的准确熵，保持时序完美对齐。

                winner_cache = DynamicCache()
                for layer_idx in range(len(past_key_values)):
                    k, v = past_key_values[layer_idx]
                    winner_cache.update(
                        k[2 * best_idx : 2 * best_idx + 2],
                        v[2 * best_idx : 2 * best_idx + 2],
                        layer_idx
                    )
                past_key_values = winner_cache

                for bh in beam_history_hiddens[1:]:
                    history_hidden_states.append(bh[2 * best_idx : 2 * best_idx + 2, :])

                last_hidden_state = history_hidden_states[-1]
                t_slow = len(generated_tokens)
                continue


    # 解码
    gen_ids = torch.tensor(generated_tokens, dtype=torch.int).unsqueeze(0).to(_device)
    dec = mmgpt.gen_vision_model.decode_code(
        gen_ids,
        shape=[1, 8, image_token_num_per_image // 24, image_token_num_per_image // 24],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return dec[0]


# ============================================================
# Prompt 编码工具
# ============================================================

def encode_prompt(vl_chat_processor: VLChatProcessor, text: str) -> str:
    conversation = [
        {"role": "User", "content": text},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag


# ============================================================
# DPG-Bench 批量生成主函数 (已重写)
# ============================================================

def generate_for_dpg_bench(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    csv_path: str,
    output_dir: str,
    resolution: int = 384,  # Janus-Pro 576 tokens 默认输出是 384x384
    pic_num: int = 4,       # 单个 Prompt 对应生成的图片数(如果>1，会拼图)
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    baseline_window: int = 8,
    k_jump: float = 2.0,
    k_mean: float = 2.0,
    jump_floor: float = 1.5,
    mean_floor: float = 3.0,
    min_window_size: int = 3,
    max_window_size: int = 8,
    beam_size: int = 3,
    temperature: float = 0.95,
):
    os.makedirs(output_dir, exist_ok=True)

    # 读取 DPG-Bench CSV 并按 item_id 去重（每个 item_id 只生成一次图像，用于所有关联命题）
    df = pd.read_csv(csv_path)
    unique_prompts = df.groupby('item_id').first().reset_index()
    items = unique_prompts[['item_id', 'text']].to_dict('records')

    if LOCAL_RANK == 0:
        print(f"Total unique DPG items: {len(items)}")
        print(f"Generating {pic_num} sub-images per item, tiled into one file (Resolution: {resolution}x{resolution}).")

    for idx, item in enumerate(tqdm(items, desc="Generating DPG Images", disable=(LOCAL_RANK != 0))):
        if idx % WORLD_SIZE != LOCAL_RANK:
            continue

        item_id = str(item['item_id'])
        prompt_text = item['text']
        
        # DPG-Bench 评测脚本要求：图像文件名不带后缀部分需完全等于 item_id
        save_path = os.path.join(output_dir, f"{item_id}.png")

        if os.path.exists(save_path):
            if LOCAL_RANK == 0:
                print(f"  Skip existing: {save_path}")
            continue

        if LOCAL_RANK == 0:
            print(f"  [{idx:05d}] Item_id: {item_id} | {prompt_text[:60]}...")

        formatted_prompt = encode_prompt(vl_chat_processor, prompt_text)
        
        sub_images = []
        for i in range(pic_num):
            img_array = generate_single_image(
                mmgpt=mmgpt,
                vl_chat_processor=vl_chat_processor,
                prompt=formatted_prompt,
                cfg_weight=cfg_weight,
                image_token_num_per_image=image_token_num_per_image,
                baseline_window=baseline_window,
                k_jump=k_jump,
                k_mean=k_mean,
                jump_floor=jump_floor,
                mean_floor=mean_floor,
                min_window_size=min_window_size,
                max_window_size=max_window_size,
                beam_size=beam_size,
                temperature=temperature,
            )
            
            # 缩放子图到指定的 resolution (用于对齐 DPG-bench 脚本中的 Crop Tuple 逻辑)
            img = PIL.Image.fromarray(img_array).resize((resolution, resolution), PIL.Image.LANCZOS)
            sub_images.append(img)

        # 处理多图拼接 (Tile) 逻辑
        if pic_num == 1:
            final_image = sub_images[0]
        else:
            # 根据 DPG 评测脚本的 crop_tuples_list: (0,0), (res,0), (0,res), (res,res)
            grid_w = resolution * 2 if pic_num > 1 else resolution
            grid_h = resolution * 2 if pic_num > 2 else resolution
            final_image = PIL.Image.new('RGB', (grid_w, grid_h))
            
            positions = [
                (0, 0),                        # 图1: top-left
                (resolution, 0),               # 图2: top-right
                (0, resolution),               # 图3: bottom-left
                (resolution, resolution)       # 图4: bottom-right
            ]
            for i in range(min(pic_num, len(positions))):
                final_image.paste(sub_images[i], positions[i])

        final_image.save(save_path, quality=95)

    if LOCAL_RANK == 0:
        print(f"\nDone. Results saved to: {output_dir}")


# ============================================================
# 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Janus-Pro DPG-Bench Gen with Adaptive-Window Beam Refinement")
    parser.add_argument("--model_path", type=str, default="/scratch/gongzx/MM2026/Janus-Pro-7B")
    
    # 替换了原有的 --metadata_path，专门用于 DPG
    parser.add_argument("--csv_path",   type=str, required=True, help="Path to dpg_bench.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    
    # DPG 专有参数
    parser.add_argument("--resolution", type=int, default=384, help="Resolution per generated sub-image")
    parser.add_argument("--pic_num",    type=int, default=4,   help="Number of images to generate and tile per item_id")
    
    # 生成核心超参数保持不变
    parser.add_argument("--cfg_weight",      type=float, default=5.0)
    parser.add_argument("--baseline_window", type=int, default=8)
    parser.add_argument("--k_jump",          type=float, default=2.0)
    parser.add_argument("--k_mean",          type=float, default=2.0)
    parser.add_argument("--jump_floor",      type=float, default=1.5)
    parser.add_argument("--mean_floor",      type=float, default=3.0)
    parser.add_argument("--min_window_size", type=int, default=3)
    parser.add_argument("--max_window_size", type=int, default=8)
    parser.add_argument("--beam_size",       type=int, default=3)
    parser.add_argument("--temperature",     type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if LOCAL_RANK == 0:
        print(f"Loading model from: {args.model_path}")

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).to(DEVICE).eval()

    if LOCAL_RANK == 0:
        print("Model loaded.")

    generate_for_dpg_bench(
        mmgpt=vl_gpt,
        vl_chat_processor=vl_chat_processor,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        resolution=args.resolution,
        pic_num=args.pic_num,
        cfg_weight=args.cfg_weight,
        baseline_window=args.baseline_window,
        k_jump=args.k_jump,
        k_mean=args.k_mean,
        jump_floor=args.jump_floor,
        mean_floor=args.mean_floor,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        beam_size=args.beam_size,
        temperature=args.temperature,
    )