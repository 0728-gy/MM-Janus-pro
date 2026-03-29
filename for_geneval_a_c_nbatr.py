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
    baseline_window: int = 16,
    k_jump: float = 2,
    k_mean: float = 2,
    jump_floor: float = 1.5,
    mean_floor: float = 3.0,
    min_window_size: int = 3,
    max_window_size: int = 12,
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

        window_size = t_fast - t_slow

        # ----------------------
        # C. 自适应触发判断
        # ----------------------
        trigger_backtrack = False
        d      = 0
        beam_d = 0   # ← 新增：beam search 实际搜索步数（默认与 d 相同）

        if t_fast <= baseline_window:
            t_slow = t_fast

        elif t_fast > baseline_window and t_fast >= 2:
            base_ents   = token_entropies[max(0, t_slow - 16) : t_slow]
            base_deltas = delta_entropies[max(0, t_slow - 16) : t_slow]

            base_ent_mean   = sum(base_ents)   / len(base_ents)
            base_ent_std    = math.sqrt(sum((x - base_ent_mean)**2 for x in base_ents)   / baseline_window)
            base_delta_mean = sum(base_deltas) / len(base_deltas)
            base_delta_std  = math.sqrt(sum((x - base_delta_mean)**2 for x in base_deltas) / baseline_window)

            jump_threshold = min(max(jump_floor, base_delta_mean + k_jump * base_delta_std), 5.0)
            mean_threshold = min(max(mean_floor, base_ent_mean   + k_mean * base_ent_std),  8.0)

            current_delta = delta_entropies[-1]

            if t_fast % 20 == 0:
                avg_window_ent = sum(token_entropies[t_slow:t_fast]) / window_size
                print(f"[{t_fast}] Δent: {current_delta:.2f} (阈: {jump_threshold:.2f}) | "
                    f"窗口均值ent: {avg_window_ent:.2f} (阈: {mean_threshold:.2f})")

            # —— 原有双重触发：Δentropy 突变 ——
            if current_delta > jump_threshold and window_size >= min_window_size:
                avg_window_ent = sum(token_entropies[t_slow:t_fast]) / window_size
                if avg_window_ent > mean_threshold:
                    trigger_backtrack = True
                    d = beam_d = window_size
                    print(f"Token {t_fast}: 触发回溯! "
                        f"(Δent: {current_delta:.2f} > {jump_threshold:.2f}, "
                        f"窗口: {window_size}, 均值ent: {avg_window_ent:.2f} > {mean_threshold:.2f})")
                else:
                    t_slow=t_fast
                    
            elif window_size >= max_window_size:                  # 1. 找窗口内 Δentropy 最大处
                    window_deltas     = delta_entropies[t_slow:t_fast]
                    max_local_idx     = max(range(len(window_deltas)), key=lambda i: window_deltas[i])
                    split_pos         = t_slow + max_local_idx   # 第一段: [t_slow, split_pos)

                    first_half_size   = split_pos - t_slow
                    # 退化保护：若 split_pos 落在边界，退回到整段
                    if first_half_size < 1:
                        first_half_size = window_size
                        split_pos       = t_fast

                    avg_first_half_ent = sum(token_entropies[t_slow:split_pos]) / first_half_size

                    if avg_first_half_ent > mean_threshold:
                        # 第一段质量差 → 回滚整窗口，beam search 只覆盖第一段
                        trigger_backtrack = True
                        d      = window_size       # 回滚步数（整个窗口都丢）
                        beam_d = window_size 
                        
                        print(f"Token {t_fast}: 触发回溯! (窗口分割@{split_pos}, "
                            f"第一段均值ent: {avg_first_half_ent:.2f} > {mean_threshold:.2f}, "
                            f"回滚={d}, beam_search={beam_d}步)")
                    else:
                        # 第一段质量OK → slow 推进到分割点，第二段留待后续
                        t_slow = split_pos
                        print(f"Token {t_fast}: 窗口分割，第一段提交 "
                            f"(均值ent: {avg_first_half_ent:.2f} ≤ {mean_threshold:.2f})，slow→{t_slow}")

        # ----------------------
        # D. 回溯 + 一致性检验
        # ----------------------
        if trigger_backtrack:
            if len(history_hidden_states) <= d:
                print(f"警告：历史不足，跳过回溯")
            else:
                # 回滚
                generated_tokens = generated_tokens[:-d]
                token_entropies  = token_entropies[:-d]
                delta_entropies  = delta_entropies[:-d]
                t_fast = len(generated_tokens)

                current_len = past_key_values.get_seq_length()
                past_key_values.crop(current_len - d)
                history_hidden_states = history_hidden_states[:-d]
                current_hidden = history_hidden_states[-1]  # [2, hidden_dim]
# 回溯准备（接在 if trigger_backtrack 内部回滚之后）
                path_tokens = []
                path_hiddens = []     # 记录过程中的 hidden states，用于恢复 history_hidden_states
                path_entropies = []   # 记录过程中的 熵，用于恢复 token_entropies

                for step in range(d):
                    if step == 0:
                        # 第一步：基于起点 [2, hidden] 计算，取 top-k
                        logits = mmgpt.gen_head(current_hidden)
                        logit_cond   = logits[0]
                        logit_uncond = logits[1]
                        final_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                        probs = torch.softmax(final_logits / temperature, dim=-1)

                        # 计算这第0步 token 本身的分布熵（用于事后记录）
                        probs_for_e = torch.softmax(logit_cond / temperature, dim=-1)
                        log_probs_e = torch.log_softmax(logit_cond / temperature, dim=-1)
                        step_0_ent = -torch.sum(probs_for_e * log_probs_e, dim=-1).item()
                        step_entropies = [step_0_ent] * beam_size 

                        topk_tokens = torch.topk(probs, beam_size).indices  # [beam_size]
                        chosen_tokens = topk_tokens

                        # 扩展 cache 为 beam_size 份
                        path_cache = expand_cache_for_beam(past_key_values, beam_size)

                    else:
                        # 后续步：current_hidden 是 [beam_size*2, hidden]
                        b_logits = mmgpt.gen_head(current_hidden).view(beam_size, 2, -1)
                        b_cond   = b_logits[:, 0, :]
                        b_uncond = b_logits[:, 1, :]
                        b_final  = b_uncond + cfg_weight * (b_cond - b_uncond)

                        # 各自贪心
                        chosen_tokens = torch.argmax(b_final, dim=-1)  # [beam_size]

                        # 计算这一步 token 本身的分布熵（用于事后记录）
                        b_probs_for_e = torch.softmax(b_cond / temperature, dim=-1)
                        b_log_for_e   = torch.log_softmax(b_cond / temperature, dim=-1)
                        step_entropies = (-torch.sum(b_probs_for_e * b_log_for_e, dim=-1)).tolist() # [beam_size] 长度的 list

                    # 记录轨迹
                    path_tokens.append(chosen_tokens)
                    path_entropies.append(step_entropies)

                    # 并行推进所有路径
                    tokens_exp = chosen_tokens.repeat_interleave(2)
                    img_embeds = mmgpt.prepare_gen_img_embeds(tokens_exp).unsqueeze(1)
                    out = mmgpt.language_model.model(
                        inputs_embeds=img_embeds,
                        use_cache=True,
                        past_key_values=path_cache
                    )
                    current_hidden = out.last_hidden_state[:, -1, :]  # [beam_size*2, hidden]
                    path_hiddens.append(current_hidden)

                # ==========================================
                # d 步走完，对每条路径前瞻 1 步 (算 d+1 步的熵)
                # ==========================================
                b_logits = mmgpt.gen_head(current_hidden).view(beam_size, 2, -1)
                b_cond   = b_logits[:, 0, :]  # 使用 cond 算熵以保持一致性

                b_probs_la = torch.softmax(b_cond / temperature, dim=-1)
                b_log_la   = torch.log_softmax(b_cond / temperature, dim=-1)
                lookahead_entropies = -torch.sum(b_probs_la * b_log_la, dim=-1)  # [beam_size]

                # 选前瞻熵最低的路径
                best_idx = torch.argmin(lookahead_entropies).item()
                print(f"  [回溯] 各路径第{d+1}步前瞻熵: {lookahead_entropies.tolist()} → 选第{best_idx}条")

                # ==========================================
                # 提交最优路径的 token 序列 和 历史状态
                # ==========================================
                for step in range(d):
                    # 取出最优路径在这个 step 的数据
                    best_tok = path_tokens[step][best_idx].item()
                    best_ent = path_entropies[step][best_idx]
                    # 切片取出这条路径对应的 [2, hidden]
                    best_hid = path_hiddens[step][2*best_idx : 2*best_idx+2] 

                    generated_tokens.append(best_tok)
                    token_entropies.append(best_ent)
                    
                    if len(token_entropies) > 1:
                        delta_entropies.append(abs(token_entropies[-1] - token_entropies[-2]))
                    else:
                        delta_entropies.append(0.0)
                        
                    history_hidden_states.append(best_hid)

                # 恢复 cache（取 best_idx 对应的那份，丢弃其他的）
                winner_cache = DynamicCache()
                for layer_idx in range(len(path_cache)):
                    k, v = path_cache[layer_idx]
                    winner_cache.update(
                        k[2*best_idx : 2*best_idx+2],
                        v[2*best_idx : 2*best_idx+2],
                        layer_idx
                    )
                past_key_values = winner_cache

                # 同步主循环的外层状态
                last_hidden_state = history_hidden_states[-1]
                t_fast = len(generated_tokens)
                t_slow = t_fast
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
# GenEval 批量生成主函数
# ============================================================

def generate_for_geneval(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    metadata_path: str,
    output_dir: str,
    num_images_per_prompt: int = 4,
    cfg_weight: float = 10.0,
    image_token_num_per_image: int = 576,
    baseline_window: int = 16,
    k_jump: float = 1.0,
    k_mean: float = 1.0,
    jump_floor: float = 1.5,
    mean_floor: float = 3.0,
    min_window_size: int = 3,
    max_window_size: int = 12,
    beam_size: int = 3,
    temperature: float = 0.95,
    start_idx: int = 0,
):
    os.makedirs(output_dir, exist_ok=True)

    prompts = []
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    if LOCAL_RANK == 0:
        print(f"Total prompts: {len(prompts)}, generating {num_images_per_prompt} images each.")
        print(f"Starting from index: {start_idx}")

    for global_idx, meta in enumerate(tqdm(prompts, desc="Prompts", disable=(LOCAL_RANK != 0))):
        if global_idx < start_idx:
            continue
        if global_idx % WORLD_SIZE != LOCAL_RANK:
            continue

        prompt_text = meta["prompt"]
        prompt_dir  = os.path.join(output_dir, f"{global_idx:05d}")
        samples_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        meta_out_path = os.path.join(prompt_dir, "metadata.jsonl")
        with open(meta_out_path, "w") as mf:
            mf.write(json.dumps(meta) + "\n")

        formatted_prompt = encode_prompt(vl_chat_processor, prompt_text)

        for img_idx in range(num_images_per_prompt):
            save_path = os.path.join(samples_dir, f"{img_idx}.jpg")

            if os.path.exists(save_path):
                print(f"  Skip existing: {save_path}")
                continue

            print(f"  [{global_idx:05d}] Image {img_idx+1}/{num_images_per_prompt} | {prompt_text[:60]}...")

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

            PIL.Image.fromarray(img_array).save(save_path, quality=95)

    print(f"\nDone. Results saved to: {output_dir}")


# ============================================================
# 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Janus-Pro GenEval with Adaptive-Window Beam Refinement")
    parser.add_argument("--model_path", type=str,
                        default="/scratch/gongzx/MM2026/Janus-Pro-7B")
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--output_dir",    type=str, required=True)
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--cfg_weight",    type=float, default=5.0)
    parser.add_argument("--baseline_window", type=int, default=16)
    parser.add_argument("--k_jump",        type=float, default=1.0)
    parser.add_argument("--k_mean",        type=float, default=1.0)
    parser.add_argument("--jump_floor",    type=float, default=1.5)
    parser.add_argument("--mean_floor",    type=float, default=3.0)
    parser.add_argument("--min_window_size", type=int, default=3)
    parser.add_argument("--max_window_size", type=int, default=12)
    parser.add_argument("--beam_size",     type=int,   default=3)
    parser.add_argument("--temperature",   type=float, default=0.95)
    parser.add_argument("--start_idx",     type=int,   default=0)
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

    generate_for_geneval(
        mmgpt=vl_gpt,
        vl_chat_processor=vl_chat_processor,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        num_images_per_prompt=args.num_images_per_prompt,
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
        start_idx=args.start_idx,
    )