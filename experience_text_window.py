import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import torch.nn.functional as F
from transformers import DynamicCache


# ============================================================
# 辅助函数：将 batch_size=1 的 KV Cache 扩展为 beam_size 份
# ============================================================
def expand_cache_for_beam(past_key_values: DynamicCache, beam_size: int) -> DynamicCache:
    """
    将 [1, heads, seq, dim] 的 KV Cache 复制 beam_size 份，
    得到 [beam_size, heads, seq, dim]。
    """
    new_cache = DynamicCache()
    for layer_idx in range(len(past_key_values)):
        k, v = past_key_values[layer_idx]
        new_cache.update(
            k.repeat(beam_size, 1, 1, 1),
            v.repeat(beam_size, 1, 1, 1),
            layer_idx,
        )
    return new_cache


# ============================================================
# 主生成函数
# ============================================================
@torch.inference_mode()
def generate_text(
    vl_gpt: MultiModalityCausalLM,
    tokenizer,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    sigma: float,
    sigma_single :float,
    model_sum: int=1,
    model_single: int=0,
    max_new_tokens: int = 512,
    d: int = 4,
    beam_size: int = 4,
):
    """
    Janus-Pro 文字生成，支持 TTS回溯机制。

    Args:
        vl_gpt            : 加载好的 MultiModalityCausalLM 模型
        tokenizer         : 对应的 tokenizer
        inputs_embeds     : 由 prepare_inputs_embeds 得到的初始 Embedding
        attention_mask    : 初始的 attention_mask
        sigma             : 触发回溯的滑动窗口熵阈值
        max_new_tokens    : 最大生成 token 数
        d                 : 滑动窗口大小 / Beam Search 步数
        beam_size         : Beam Search 宽度

    Returns:
        generated_tokens  : 生成的 token id 列表
        all_hidden_states : 每步的 last_hidden_state (List[Tensor[1, hidden_dim]])
    """
    _device = next(vl_gpt.parameters()).device
    eos_token_id = tokenizer.eos_token_id

    generated_tokens: list[int] = []
    all_hidden_states: list[torch.Tensor] = []  # 每个元素 shape [1, hidden_dim]
    token_entropies: list[float] = []

    # ------------------------------------------------------------------
    # 1. Prefill：处理整段 Prompt（图像 + 文字）
    # ------------------------------------------------------------------
    outputs = vl_gpt.language_model.model(
        inputs_embeds=inputs_embeds,
        past_key_values=None,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values: DynamicCache = outputs.past_key_values
    last_hidden_state = outputs.last_hidden_state[:, -1, :]  # [1, hidden_dim]
    all_hidden_states.append(last_hidden_state)

    # ------------------------------------------------------------------
    # 2. 主生成循环
    # ------------------------------------------------------------------
    # ★ 修复 1：t_fast / t_slow 须在每步末尾 +1，原代码从未递增导致回溯永远不触发
    t_fast = 0
    t_slow = 0

    while len(generated_tokens) < max_new_tokens:

        # ── A. 预测下一个 Token ──────────────────────────────────────────
        logits = vl_gpt.language_model.lm_head(last_hidden_state)  # [1, vocab]

        probs    = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy  = -torch.sum(probs * log_probs, dim=-1).item()
        

        

        token_entropies.append(entropy)

        next_token = torch.argmax(logits, dim=-1)  # [1]  贪心解码
        generated_tokens.append(next_token.item())

        if next_token.item() == eos_token_id:
            break

        # ── B. 正常推进：更新 attention_mask 并做下一步 Forward ──────────
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((1, 1), dtype=torch.long, device=_device)],
            dim=-1,
        )
        next_embeds = vl_gpt.language_model.get_input_embeddings()(next_token).unsqueeze(1)
        outputs = vl_gpt.language_model.model(
            inputs_embeds=next_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values  = outputs.past_key_values
        last_hidden_state = outputs.last_hidden_state[:, -1, :]  # [1, hidden_dim]
        all_hidden_states.append(last_hidden_state)

        # ★ 修复 1（续）：每步结束后推进双指针
        t_fast += 1
        t_slow += 1

        # ── C. TTS 回溯检测 ──────────────────────────────────────────────
        # 让 t_slow 始终落后 t_fast d 步，形成滑动窗口
        if t_fast - t_slow < d:
            t_slow -= 1

        trigger_backtrack = False
        if t_fast - t_slow == d:
            current_window_entropy = sum(token_entropies[-d:])

            if current_window_entropy > sigma and len(generated_tokens)>d and model_sum ==1:
                trigger_backtrack = True
                print(f"在第{len(generated_tokens)}个token回溯")
            
            count_num =0
            for i in token_entropies[-d:]:
                if i > sigma_single:
                    count_num+=1
                    
                
                if  len(generated_tokens)>d and model_single==1 and count_num ==d:
                    trigger_backtrack =True
                    print(f"在第{len(generated_tokens)}个token回溯")
        if not trigger_backtrack:
            continue  # 无需回溯，直接进行下一步

        # ── D. 回溯 + Beam Search ────────────────────────────────────────
        if len(all_hidden_states) <= d:
            # 历史不足，跳过本次回溯
            print("警告：历史不足，跳过回溯")
            continue

        # -- D1. 撤销最近 d 步 --
        generated_tokens  = generated_tokens[:-d]
        token_entropies   = token_entropies[:-d]
        attention_mask    = attention_mask[:, :-d]
        all_hidden_states = all_hidden_states[:-d]

        current_len = past_key_values.get_seq_length()
        past_key_values.crop(current_len - d)

        start_hidden_state = all_hidden_states[-1]  # [1, hidden_dim]

        # -- D2. 将 Cache 和 hidden_state 扩展为 beam_size 份 --
        # ★ 修复 3：提取 expand_cache_for_beam，与图像生成代码保持一致
        past_key_values      = expand_cache_for_beam(past_key_values, beam_size)
        beam_attention_mask  = attention_mask.repeat(beam_size, 1)  # [beam, seq]

        # beam_history_hiddens 存储每步所有 beam 的 hidden state
        # 每个元素 shape: [beam_size, hidden_dim]
        beam_hidden_state    = start_hidden_state.repeat(beam_size, 1)
        beam_history_hiddens = [beam_hidden_state]

        beam_scores    = torch.full((beam_size,), -1e9, device=_device)
        beam_scores[0] = 0.0
        beam_sequences = torch.zeros((beam_size, 0), dtype=torch.long, device=_device)
        # 每步、每 beam 的熵：shape 最终为 [beam_size, d]
        beam_step_entropies: list[torch.Tensor] = []

        # -- D3. Beam Search 循环（共 d 步）--
        for step in range(d):
            curr_h = beam_history_hiddens[-1]  # [beam_size, hidden_dim]

            # 预测 logits
            b_logits    = vl_gpt.language_model.lm_head(curr_h)       # [beam_size, vocab]
            b_log_probs = F.log_softmax(b_logits, dim=-1)
            b_probs     = F.softmax(b_logits, dim=-1)
            b_entropy   = -torch.sum(b_probs * b_log_probs, dim=-1)   # [beam_size]

            # 累积得分：[beam_size, 1] + [beam_size, vocab] -> [beam_size, vocab]
            next_scores = beam_scores.unsqueeze(1) + b_log_probs
            top_scores, top_indices = torch.topk(next_scores.view(-1), beam_size)

            beam_indices  = top_indices // b_logits.shape[-1]  # 来自哪个旧 beam
            token_indices = top_indices %  b_logits.shape[-1]  # 具体 token id

            # 更新得分与序列
            beam_scores    = top_scores
            beam_sequences = torch.cat(
                [beam_sequences[beam_indices], token_indices.unsqueeze(-1)], dim=-1
            )

            # ★ 修复 4：先对历史 hidden states 做同步重排，再追加新的 —— 防止路径交叉
            for i in range(len(beam_history_hiddens)):
                beam_history_hiddens[i] = beam_history_hiddens[i][beam_indices]

            # ★ 修复 5：重排后再记录熵（记录的是本步被选中 beam 对应的熵）
            beam_step_entropies.append(b_entropy[beam_indices])  # [beam_size]

            # 重排 KV Cache 与 attention_mask
            past_key_values.reorder_cache(beam_indices)
            beam_attention_mask = beam_attention_mask[beam_indices]
            beam_attention_mask = torch.cat(
                [beam_attention_mask,
                 torch.ones((beam_size, 1), dtype=torch.long, device=_device)],
                dim=-1,
            )

            # 下一步 Forward
            next_embeds = vl_gpt.language_model.get_input_embeddings()(token_indices).unsqueeze(1)
            b_outputs = vl_gpt.language_model.model(
                inputs_embeds=next_embeds,
                past_key_values=past_key_values,
                attention_mask=beam_attention_mask,
                use_cache=True,
            )
            past_key_values = b_outputs.past_key_values
            new_h = b_outputs.last_hidden_state[:, -1, :]  # [beam_size, hidden_dim]
            beam_history_hiddens.append(new_h)

        # -- D4. 选出优胜 Beam --
        best_idx      = torch.argmax(beam_scores).item()
        winner_tokens = beam_sequences[best_idx].tolist()

        # 收集优胜路径的熵（每步的 beam_step_entropies 已按 beam_indices 重排）
        winner_entropies = [beam_step_entropies[s][best_idx].item() for s in range(d)]

        # 收集优胜路径的 hidden states（beam_history_hiddens[1:] 为 d 步的输出）
        winner_hiddens = [
            beam_history_hiddens[s + 1][best_idx : best_idx + 1]  # [1, hidden_dim]
            for s in range(d)
        ]

        # -- D5. 将优胜路径同步回主循环变量 --
        generated_tokens.extend(winner_tokens)
        token_entropies.extend(winner_entropies)
        all_hidden_states.extend(winner_hiddens)

        # 提取优胜 beam 的 KV Cache
        winner_cache = DynamicCache()
        for layer_idx in range(len(past_key_values)):
            k, v = past_key_values[layer_idx]
            winner_cache.update(
                k[best_idx : best_idx + 1],
                v[best_idx : best_idx + 1],
                layer_idx,
            )
        past_key_values = winner_cache

        attention_mask    = beam_attention_mask[best_idx : best_idx + 1]
        last_hidden_state = all_hidden_states[-1]

        # ★ 修复 6：winner tokens 中若含 EOS，提前退出
        if eos_token_id in winner_tokens:
            break

        # slow 指针同步到 fast 指针
        t_slow = t_fast
        # ★ 修复 2：回溯处理完毕后 continue，避免执行循环尾部的多余 forward
        continue

    return generated_tokens, all_hidden_states


# ============================================================
# 模型加载
# ============================================================
model_path = "/scratch/gongzx/models/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# ============================================================
# 准备输入
# ============================================================
conversation = [
    {
        "role": "<|User|>",
        "content": "<image_placeholder>\nConvert the formula into latex code.",
        "images": ["/home/gongzx/share/MM2026/Janus/MM-Janus-pro/images/equation.png"],
    },
    {"role": "<|Assistant|>", "content": ""},
]

pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# ============================================================
# 推理
# ============================================================
tokens, hiddens = generate_text(
    vl_gpt=vl_gpt,
    tokenizer=tokenizer,
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    max_new_tokens=512,
    d=4,
    sigma=100,
    beam_size=4,
)

answer = tokenizer.decode(tokens, skip_special_tokens=True)
print(f"生成内容: {answer}")