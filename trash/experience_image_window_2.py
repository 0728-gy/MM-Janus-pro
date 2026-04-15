import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import torch.nn.functional as F
import PIL.Image
from transformers import DynamicCache
import math


# specify the path to the model
model_path = "/scratch/gongzx/MM2026/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


conversation = [
    {
        "role": "User",
        "content": "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate_image_with_tts(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    baseline_window: int = 16,      # 用于计算阈值的滑动窗口大小
    k_jump: float = 2.5,            # Δentropy 突变阈值系数
    k_mean: float = 2.5,            # 窗口均值熵阈值系数
    jump_floor: float = 1.5,        # Δentropy 阈值下限
    mean_floor: float = 3.0,        # 均值熵阈值下限
    min_window_size: int = 3,       # 触发回溯的最小窗口长度
    max_window_size: int = 12,      # 强制提交的最大窗口长度
    beam_size: int = 3,
    temperature: float = 0.95
):
    # ==========================
    # 1. 初始化与预热 (Prefill)
    # ==========================
    _device = next(mmgpt.parameters()).device
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).cuda()

    tokens = torch.stack([input_ids.clone(), input_ids.clone()], dim=0)
    tokens[1, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    outputs = mmgpt.language_model.model(
        inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None
    )
    past_key_values = outputs.past_key_values
    last_hidden_state = outputs.last_hidden_state[:, -1, :]

    # ==========================
    # 2. 状态变量初始化
    # ==========================
    generated_tokens = []
    token_entropies = []
    delta_entropies = []       # |ent[i] - ent[i-1]|，与 token_entropies 等长（第0步填0）
    

    history_hidden_states = []
    history_hidden_states.append(last_hidden_state)

    t_fast = 0
    t_slow = 0

    print(f"Start generating {image_token_num_per_image} tokens...")

    # ==========================
    # 3. 主生成循环
    # ==========================
    while len(generated_tokens) < image_token_num_per_image:
        

        # ----------------------
        # A. 预测下一个 Token
        # ----------------------
        logits = mmgpt.gen_head(last_hidden_state)  # [2, vocab_size]

        logit_cond   = logits[0, :]
        logit_uncond = logits[1, :]
        final_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        probs       = torch.softmax(final_logits / temperature, dim=-1)
        probs_for_e = torch.softmax(logit_cond / temperature, dim=-1)
        log_probs   = torch.log_softmax(logit_cond / temperature, dim=-1)
        entropy     = -torch.sum(probs_for_e * log_probs, dim=-1).item()

        next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens.append(next_token.item())
        token_entropies.append(entropy)
        t_fast = len(generated_tokens)

        # 计算 Δentropy（第1步没有前驱，填0）
        if t_fast == 1:
            delta_entropies.append(0.0)
        else:
            delta_entropies.append(abs(token_entropies[-1] - token_entropies[-2]))

        

        # ----------------------
        # B. 正常推进（更新 KV cache）
        # ----------------------
        next_token_input = next_token.repeat(2)
        img_embeds    = mmgpt.prepare_gen_img_embeds(next_token_input)
        inputs_embeds = img_embeds.unsqueeze(1)

        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values
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
        # D. 回溯 + Beam Search
        # ----------------------
        if trigger_backtrack:
            if len(history_hidden_states) <= d:
                print(f"警告：历史不足，跳过回溯")
            else:
                # 回滚 token 序列和熵序列
                generated_tokens = generated_tokens[:-d]
                token_entropies  = token_entropies[:-d]
                delta_entropies  = delta_entropies[:-d]

                t_fast = len(generated_tokens)
                

                # 回滚 KV cache
                current_len = past_key_values.get_seq_length()
                past_key_values.crop(current_len - d)
                history_hidden_states = history_hidden_states[:-d]
                start_hidden_state    = history_hidden_states[-1]

                # Beam Search 初始化
                beam_hidden_state = start_hidden_state.repeat(beam_size, 1)
                past_key_values   = expand_cache_for_beam(past_key_values, beam_size)

                beam_scores = torch.full((beam_size,), -1e9, device=_device)
                beam_scores[0] = 0.0
                beam_seqs   = torch.zeros((beam_size, 0), dtype=torch.long, device=_device)
                beam_history_hiddens = [beam_hidden_state]
                beam_entropies = []  # 在 beam search 循环外初始化

                # Beam Search 循环
                for step in range(d):
                    
                    curr_h = beam_history_hiddens[-1]
                    b_logits = mmgpt.gen_head(curr_h)

                    b_logits_view  = b_logits.view(beam_size, 2, -1)
                    b_cond         = b_logits_view[:, 0, :]
                    b_uncond       = b_logits_view[:, 1, :]
                    b_final_logits = b_uncond + cfg_weight * (b_cond - b_uncond)

                    b_probs_for_e = torch.softmax(b_cond/temperature, dim=-1)
                    b_log_probs_e = torch.log_softmax(b_cond/temperature, dim=-1)
                    b_log_probs      = F.log_softmax(b_final_logits/temperature, dim=-1)
                    next_scores      = beam_scores.unsqueeze(1) + b_log_probs
                    next_scores_flat = next_scores.view(-1)

                    b_entropies = -torch.sum(b_probs_for_e * b_log_probs_e, dim=-1)  # [beam_size]
                    
                    topk_scores, topk_indices = torch.topk(next_scores_flat, beam_size)
                    beam_indices  = topk_indices // b_final_logits.shape[-1]
                    token_indices = topk_indices %  b_final_logits.shape[-1]

                    b_entropies = b_entropies[beam_indices] 
                    beam_entropies.append(b_entropies)

                    beam_scores = topk_scores
                    beam_seqs   = torch.cat([beam_seqs[beam_indices], token_indices.unsqueeze(1)], dim=1)

                    cache_indices = []
                    for b_idx in beam_indices:
                        cache_indices.append(2 * b_idx)
                        cache_indices.append(2 * b_idx + 1)
                    cache_indices_tensor = torch.tensor(cache_indices, dtype=torch.long, device=_device)

                    for i in range(len(beam_history_hiddens)):
                        beam_history_hiddens[i] = beam_history_hiddens[i][cache_indices_tensor]
                    past_key_values.reorder_cache(cache_indices_tensor)

                    next_tokens_input = token_indices.repeat_interleave(2)
                    img_embeds = mmgpt.prepare_gen_img_embeds(next_tokens_input).unsqueeze(1)

                    b_outputs = mmgpt.language_model.model(
                        inputs_embeds=img_embeds,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    new_h = b_outputs.last_hidden_state[:, -1, :]
                    beam_history_hiddens.append(new_h)
                    

                # 选出最优 beam 并恢复状态
                best_idx      = torch.argmax(beam_scores).item()
                winner_tokens = beam_seqs[best_idx].tolist()
                generated_tokens.extend(winner_tokens)

                winner_entropies = [beam_entropies[step][best_idx].item() for step in range(d)]
                # 熵占位：用回溯前那段的均值，避免 Δentropy 失真
                token_entropies.extend(winner_entropies)
                delta_entropies.extend(
                    [0.0] + [abs(winner_entropies[i] - winner_entropies[i-1]) for i in range(1, d)]
                )
               

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

    # ==========================
    # 4. 解码
    # ==========================
    print("Generation finished. Decoding image...")
    gen_ids = torch.tensor(generated_tokens, dtype=torch.int).unsqueeze(0).cuda()
    dec = mmgpt.gen_vision_model.decode_code(
        gen_ids,
        shape=[1, 8, image_token_num_per_image // 24, image_token_num_per_image // 24]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return dec[0]


def expand_cache_for_beam(past_key_values, beam_size):
    new_cache = DynamicCache()
    for layer_idx in range(len(past_key_values)):
        k, v = past_key_values[layer_idx]
        new_k = k.repeat(beam_size, 1, 1, 1)
        new_v = v.repeat(beam_size, 1, 1, 1)
        new_cache.update(new_k, new_v, layer_idx)
    return new_cache


# 调用示例
img_array= generate_image_with_tts(
    vl_gpt, vl_chat_processor, prompt,
    baseline_window=16,   # 滑动基线窗口
    k_jump=1.0,
    k_mean=1.0,
    jump_floor=1.5,
    mean_floor=3,
    min_window_size=3,
    max_window_size=12,
)
PIL.Image.fromarray(img_array).save("/scratch/gongzx/MM2026/ex_image/result_8.jpg")