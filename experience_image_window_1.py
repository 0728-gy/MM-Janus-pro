import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import torch.nn.functional as F
import PIL.Image

from transformers import DynamicCache

# specify the path to the model
model_path = "/share/home/u11154/JingyiLiu/MM2026/Janus/model_weights/Janus-Pro-7B"
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
#一张悉尼歌剧院与埃菲尔铁塔并肩而立的特写、高对比度照片，背景是翻腾着能量、布满爆炸般黄色星辰和蓝色辐射漩涡的深夜星空。


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
    ema_alpha: float = 0.1,      # 统计更新灵敏度 (0.05~0.15)
    k_jump: float = 2.5,         # 突变敏感度 (k个标准差)
    k_mean: float = 2.0,         # 质量敏感度
    jump_floor: float = 1.0,     # 突变保底阈值
    mean_floor: float = 2.5,     # 质量保底阈值
    min_window_size: int = 3,    # 触发回溯的最小窗口
    max_window_size: int = 12,    # 强制提交的最大窗口上限  # 熵阈值 (根据图像生成的分布可能需要调整，建议 2.0-5.0)
    beam_size: int = 3,    # Beam Search 宽度
    temperature: float = 0.95
):
    # ==========================
    # 1. 初始化与预热 (Prefill)
    # ==========================
    # 编码 Prompt
    _device = next(mmgpt.parameters()).device
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).cuda()
    
    # 构造 CFG Batch: [Cond, Uncond] (Batch Size = 2)
    # 构造 CFG Batch: [Cond, Uncond] (Batch Size = 2)
    # 1. 先复制两份原始的 input_ids (Cond 和 Uncond)
    tokens = torch.stack([input_ids.clone(), input_ids.clone()], dim=0)
    
    # 2. 对 Unconditional 分支 (索引为 1) 进行处理：
    # 保留第一个 Token (BOS) 和最后一个 Token (Image Start Tag)，
    # 将中间的所有描述性文字替换为 pad_id。
    tokens[1, 1:-1] = vl_chat_processor.pad_id 
    
    # 3. 将 Token 转换为 Embedding
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    # 运行 Text 部分，获取初始 KV Cache 和 最后一个 Hidden State
    outputs = mmgpt.language_model.model(
        inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None
    )
    past_key_values = outputs.past_key_values
    
    # 获取最后一个文本 Token 的输出，准备输入给 Image Gen Head
    # shape: [2, hidden_dim]
    last_hidden_state = outputs.last_hidden_state[:, -1, :]

    # ==========================
    # 2. 状态变量初始化
    # ==========================
    generated_tokens = []    # 存储生成的 token id
    token_entropies = []     # 存储熵
    token_scis = []      # ← 新增
    token_cfg_gaps = []  # ← 新增
    
    # 存储历史的 hidden_states，用于回溯时恢复现场
    # 列表元素 shape: [2, hidden_dim]
    history_hidden_states = [] 
    
    t_fast = 0
    t_slow = 0

    sci_ema_mean = 0.0
    sci_ema_var = 0.0
    diff_ema_mean = 0.0
    diff_ema_var = 0.0

    print(f"Start generating {image_token_num_per_image} tokens...")
    
    history_hidden_states.append(last_hidden_state)
    # ==========================
    # 3. 主生成循环
    # ==========================
    while len(generated_tokens) < image_token_num_per_image:
        
        # 保存当前用于生成的 hidden_state (对应 step t 的输入状态)
        

        # ----------------------
        # A. 预测下一个 Token
        # ----------------------
        # 这里的输入是 Transformer 的输出 (Hidden State)
        logits = mmgpt.gen_head(last_hidden_state) # [2, vocab_size]
        
        # CFG 计算
        logit_cond = logits[0, :]
        logit_uncond = logits[1, :]
        cfg_gap = torch.mean(torch.abs(logit_cond - logit_uncond)).item()
        final_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond) # [vocab_size]
        
        # 计算概率与熵
        probs = torch.softmax(final_logits / temperature, dim=-1) # temp=1.0 usually for sampling
        log_probs = torch.log_softmax(final_logits / temperature, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).item()
        sci = entropy * cfg_gap
        
        # 采样 (通常图像生成需要一定的随机性，这里用 Multinomial)
        next_token = torch.multinomial(probs, num_samples=1) # [1]
        
        # 临时存储结果
        generated_tokens.append(next_token.item())
        token_entropies.append(entropy)
        token_scis.append(sci)          # ← 新增
        token_cfg_gaps.append(cfg_gap)  # ← 新增
        # ----------------------
        # C. 正常推进 (无回溯)
        # ----------------------
        # 准备下一次迭代的输入
        # next_token: [1] -> 需要复制两份 [cond, uncond]
        # 注意: 即使 Uncond 分支，输入的图像 Token 也是一样的，区别在于之前的 Prompt 是 Pad
        next_token_input = next_token.repeat(2) # [2]
        
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_input)
        inputs_embeds = img_embeds.unsqueeze(1) # [2, 1, hidden_dim]

        # Forward
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds, 
            use_cache=True, 
            past_key_values=past_key_values
        )
        
        # 更新 last_hidden_state
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        history_hidden_states.append(last_hidden_state)
        
        # 指针移动
        t_fast = len(generated_tokens)
        window_size = t_fast - t_slow
        trigger_backtrack = False
        d = 0  # 记录需要回溯的动态窗口大小

        # 至少需要 2 个 token 才能计算 ΔSCI
        if t_fast >= 2:
            # 1. 计算局部突变 (ΔSCI)
            delta_sci = abs(token_scis[-1] - token_scis[-2])
            
            # 2. 如果发生显著突变，且窗口积累了足够的 token 供评估
            if delta_sci > jump_threshold and window_size >= min_window_size:
                
                # 提取当前 slow 到 fast 之间的 token_sci
                current_window_scis = token_scis[-window_size:]
                avg_sci = sum(current_window_scis) / window_size
                
                # 3. 质量审视：判断这段区间的平均不确定性是否超标
                if avg_sci > mean_threshold:
                    trigger_backtrack = True
                    d = window_size  # 把当前窗口大小作为 Beam Search 的深度
                    
                    print(f"Token {t_fast}: 触发自适应回溯! "
                        f"(突变 ΔSCI: {delta_sci:.2f}, 窗口大小: {window_size}, 平均SCI: {avg_sci:.2f})")
                else:
                    # 发生了突变，但平均质量还算及格。
                    # 说明平稳度过了这个边缘交界处，将当前窗口“安全提交”。
                    t_slow = t_fast

        # 4. 工程安全兜底 (防显存爆炸)
        # 如果模型一直画得很平滑（无突变），但窗口已经达到了最大限制，强制移动慢指针提交。
        if not trigger_backtrack and window_size >= max_window_size:
            t_slow = t_fast
        

        if trigger_backtrack:
        

            # ✅ 问题4：边界保护
            if len(history_hidden_states) <= d:
                print(f"警告：历史不足，跳过回溯")
                # 正常推进，不回溯
            else:
                # --- 1. 回溯状态 ---
                generated_tokens = generated_tokens[:-d]
                token_entropies = token_entropies[:-d]
                token_scis = token_scis[:-d]          # ← 补上
                token_cfg_gaps = token_cfg_gaps[:-d]  # ← 补上

                # ✅ 问题2：crop 用绝对长度
                current_len = past_key_values.get_seq_length()
                past_key_values.crop(current_len - d)

                history_hidden_states = history_hidden_states[:-d]
                start_hidden_state = history_hidden_states[-1]  # 安全

                # --- 2. Beam Search 初始化 ---
                # ✅ 问题3：正确扩展 hidden state 和 cache
                beam_hidden_state = start_hidden_state.repeat(beam_size, 1)  # [2*k, hidden_dim]
                past_key_values = expand_cache_for_beam(past_key_values, beam_size)

                beam_scores = torch.full((beam_size,), -1e9, device=_device)
                beam_scores[0] = 0.0 
                beam_seqs = torch.zeros((beam_size, 0), dtype=torch.long, device=next(mmgpt.parameters()).device)
                beam_history_hiddens = [beam_hidden_state]

            # --- 3. Beam Search 循环 (运行 d 步) ---
                for step in range(d):
                    # 3.1 预测
                    curr_h = beam_history_hiddens[-1]
                    b_logits = mmgpt.gen_head(curr_h) # [2*k, vocab]
                    
                    # 3.2 CFG (批量处理)
                    # Reshape 到 [k, 2, vocab] -> [k, vocab]
                    b_logits_view = b_logits.view(beam_size, 2, -1)
                    b_cond = b_logits_view[:, 0, :]
                    b_uncond = b_logits_view[:, 1, :]
                    b_final_logits = b_uncond + cfg_weight * (b_cond - b_uncond)
                    
                    b_log_probs = F.log_softmax(b_final_logits, dim=-1) # [k, vocab]
                    
                    # 3.3 计算得分并选择 TopK
                    # 当前总分 = 历史得分 + 新词概率
                    # [k, 1] + [k, vocab] -> [k, vocab] -> flatten
                    next_scores = beam_scores.unsqueeze(1) + b_log_probs
                    next_scores_flat = next_scores.view(-1)
                    
                    # 选出全局 Top K
                    topk_scores, topk_indices = torch.topk(next_scores_flat, beam_size)
                    
                    # 解析索引
                    beam_indices = topk_indices // b_final_logits.shape[-1] # 属于哪个旧 Beam
                    token_indices = topk_indices % b_final_logits.shape[-1]  # 具体的 Token ID
                    
                    # 3.4 更新 Beam 状态
                    beam_scores = topk_scores
                    
                    # 更新序列记录
                    # [k, current_len] -> select -> append
                    beam_seqs = torch.cat([beam_seqs[beam_indices], token_indices.unsqueeze(1)], dim=1)
                    
                    # 记录熵 (为了数据完整性，这里主要记录赢家的路径)
                    # 简单起见，这里只做路径选择，不重新计算熵列表
                    
                    # 3.5 准备下一步输入
                    # KV Cache 重排 (注意：每个 Beam 对应 2 个 Cache entry: 2*idx, 2*idx+1)
                    # 构造 [2*k] 的索引映射
                    cache_indices = []
                    for b_idx in beam_indices:
                        cache_indices.append(2 * b_idx)
                        cache_indices.append(2 * b_idx + 1)

                    # 转成 tensor
                    cache_indices_tensor = torch.tensor(cache_indices, dtype=torch.long, device=_device)

                    # ------------------------------------------------------------------
                    # ✅ 修复 Bug 2：同步重排历史隐状态，防止路径交叉产生缝合怪！
                    for i in range(len(beam_history_hiddens)):
                        beam_history_hiddens[i] = beam_history_hiddens[i][cache_indices_tensor]

                    
                    past_key_values.reorder_cache(cache_indices_tensor)

                    # 计算 Image Embedding 进行下一步 Forward
                    # token_indices: [k]
                    # 需要扩展成 [2*k] (cond, uncond 使用相同的 image input)
                    next_tokens_input = token_indices.repeat_interleave(2) # [t1, t1, t2, t2...]
                    
                    img_embeds = mmgpt.prepare_gen_img_embeds(next_tokens_input) # [2*k, hidden_dim]
                    img_embeds = img_embeds.unsqueeze(1) # [2*k, 1, hidden_dim]
                    
                    # Forward Pass
                    b_outputs = mmgpt.language_model.model(
                        inputs_embeds=img_embeds, use_cache=True, past_key_values=past_key_values
                    )
                    
                    # 保存新的 Hidden State
                    new_h = b_outputs.last_hidden_state[:, -1, :] # [2*k, hidden_dim]
                    beam_history_hiddens.append(new_h)
            
                # --- 4. 选出赢家并恢复 ---
                best_idx = torch.argmax(beam_scores).item()  # ✅ 不要硬编码0，取真正最高分
                winner_tokens = beam_seqs[best_idx].tolist()
                generated_tokens.extend(winner_tokens)
                token_entropies.extend([0.0] * d)
                token_scis.extend([0.0] * d)        # ← 补上
                token_cfg_gaps.extend([0.0] * d) 

                winner_cache = DynamicCache()
                for layer_idx in range(len(past_key_values)):
                    k, v = past_key_values[layer_idx]
                    winner_cache.update(
                        k[2*best_idx : 2*best_idx+2], 
                        v[2*best_idx : 2*best_idx+2], 
                        layer_idx
                    )
                past_key_values = winner_cache

                # ✅ 问题4：带边界检查地恢复 hidden states
                assert best_idx < beam_size
                for bh in beam_history_hiddens[1:]:
                    wh = bh[2*best_idx : 2*best_idx+2, :]
                    history_hidden_states.append(wh)

                last_hidden_state = history_hidden_states[-1]
                t_slow = t_fast
                continue

        

    # ==========================
    # 4. 解码 (Decoding)
    # ==========================
    print("Generation finished. Decoding image...")
    
    # 转换格式 [1, 576]
    gen_ids = torch.tensor(generated_tokens, dtype=torch.int).unsqueeze(0).cuda()
    
    # 调用视觉解码器
    dec = mmgpt.gen_vision_model.decode_code(
        gen_ids, 
        shape=[1, 8, image_token_num_per_image//24, image_token_num_per_image//24] # 假设 24x24 patches
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    
    # 反归一化
    
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    
    return dec[0]

def expand_cache_for_beam(past_key_values, beam_size):
    new_cache = DynamicCache()
    for layer_idx in range(len(past_key_values)):
        k, v = past_key_values[layer_idx]
        # [2, heads, seq, dim] -> [2*beam_size, heads, seq, dim]
        # repeat(beam_size,1,1,1): [cond,uncond,cond,uncond,...] ✅ 配对正确
        new_k = k.repeat(beam_size, 1, 1, 1)
        new_v = v.repeat(beam_size, 1, 1, 1)
        new_cache.update(new_k, new_v, layer_idx)
    return new_cache

# 调用示例
img_array = generate_image_with_tts(
    vl_gpt, vl_chat_processor, prompt,
    min_window_size=3,
    max_window_size=12,
    ema_alpha=0.1,    # 统计更新灵敏度
    k_jump=2.5,       # 2.5倍标准差以上视为突变
    k_mean=2.0,       # 2.0倍标准差以上视为质量不合格
    jump_floor=1.0,   # 基础波动阈值
    mean_floor=3.0    # 基础质量阈值
)
PIL.Image.fromarray(img_array).save("result_6.jpg")




