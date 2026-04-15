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
    k_jump: float = 1.5,            # Δentropy 突变阈值系数
    k_mean: float = 1.0,            # 窗口均值熵阈值系数
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
        #    前 baseline_window 个 token 不做回溯
        # ----------------------
        trigger_backtrack = False
        d = 0

        if t_fast <= baseline_window:   # ← 加这两行
            t_slow = t_fast    

        elif t_fast > baseline_window and t_fast >= 2:
            # 用前 baseline_window 个 token 计算局部基线
            base_ents   = token_entropies[max(0, t_slow-16) : t_slow]
            base_deltas = delta_entropies[max(0, t_slow-16) : t_slow]

            base_ent_mean   = sum(base_ents)   / len(base_ents)
            base_ent_std    = math.sqrt(sum((x - base_ent_mean) ** 2 for x in base_ents)   / baseline_window)
            base_delta_mean = sum(base_deltas) / len(base_deltas)
            base_delta_std  = math.sqrt(sum((x - base_delta_mean) ** 2 for x in base_deltas) / baseline_window)

            # 自适应阈值（带下限保护）
            jump_threshold = max(jump_floor, base_delta_mean + k_jump * base_delta_std)
            mean_threshold = max(mean_floor, base_ent_mean   + k_mean * base_ent_std)
            jump_threshold = min(jump_threshold, 5.0)
            mean_threshold = min(mean_threshold, 8.0)

            current_delta = delta_entropies[-1]

            if t_fast % 20 == 0:
                avg_window_ent = sum(token_entropies[t_slow:t_fast]) / window_size
                print(f"[{t_fast}] Δent: {current_delta:.2f} (阈: {jump_threshold:.2f}) | "
                      f"窗口均值ent: {avg_window_ent:.2f} (阈: {mean_threshold:.2f})")

            # 双重触发：Δentropy 突变 且 窗口够大
            if current_delta > jump_threshold and window_size >= min_window_size:
                avg_window_ent = sum(token_entropies[t_slow:t_fast]) / window_size
                if avg_window_ent > mean_threshold:
                    trigger_backtrack = True
                    d = window_size
                    print(f"Token {t_fast}: 触发回溯! "
                          f"(Δent: {current_delta:.2f} > {jump_threshold:.2f}, "
                          f"窗口: {window_size}, 均值ent: {avg_window_ent:.2f} > {mean_threshold:.2f})")
                else:
                    # 质量OK，安全提交，slow 追上 fast
                    t_slow = t_fast

        # 兜底：窗口超过上限强制提交
        # 改后：到上限时也判断质量
            if not trigger_backtrack and window_size >= max_window_size:
                avg_window_ent = sum(token_entropies[t_slow:t_fast]) / window_size
                if avg_window_ent > mean_threshold:
                    trigger_backtrack = True
                    d = window_size
                    print(f"Token {t_fast}: 触发回溯! (窗口上限, 均值ent: {avg_window_ent:.2f} > {mean_threshold:.2f})")
                else:
                    t_slow = t_fast

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