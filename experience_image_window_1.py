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

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


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
    cfg_weight: float = 10.0,
    image_token_num_per_image: int = 576,
    d: int = 8,           
    sigma: float = 8.0,   
    beam_size: int = 3    
):
    _device = next(mmgpt.parameters()).device
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).cuda()
    
    tokens = torch.stack([input_ids, torch.ones_like(input_ids) * vl_chat_processor.pad_id], dim=0)
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    outputs = mmgpt.language_model.model(
        inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None
    )
    past_key_values = outputs.past_key_values
    
    last_hidden_state = outputs.last_hidden_state[:, -1, :]

    generated_tokens = []    
    token_entropies = []     
    
    # [FIX 2]: 在循环开始前初始化，确保 index -1 永远指向最新生成序列的最末端
    history_hidden_states = [last_hidden_state] 
    
    t_fast = 0
    t_slow = 0

    print(f"Start generating {image_token_num_per_image} tokens...")

    while len(generated_tokens) < image_token_num_per_image:
        
        # 此时 last_hidden_state 是准确对应于下一步预测的上下文状态
        logits = mmgpt.gen_head(last_hidden_state)
        
        logit_cond = logits[0, :]
        logit_uncond = logits[1, :]
        final_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond) 
        
        probs = torch.softmax(final_logits / 1.0, dim=-1) 
        log_probs = torch.log_softmax(final_logits / 1.0, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).item()
        
        next_token = torch.multinomial(probs, num_samples=1) 
        
        generated_tokens.append(next_token.item())
        token_entropies.append(entropy)

        # C. 正常推进 
        next_token_input = next_token.repeat(2) 
        
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_input)
        inputs_embeds = img_embeds.unsqueeze(1) 

        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds, 
            use_cache=True, 
            past_key_values=past_key_values
        )
        
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        # [FIX 2]: 在循环末尾追加，确保对齐
        history_hidden_states.append(last_hidden_state)
        
        t_fast += 1
        t_slow += 1
        
        if t_fast - t_slow < d:
            t_slow -= 1
            
        trigger_backtrack = False
        if t_fast - t_slow == d:
            current_window_entropy = sum(token_entropies[-d:])
            if current_window_entropy > sigma and len(generated_tokens) > d:
                trigger_backtrack = True
                print(f"Token {len(generated_tokens)}: 触发回溯 (Entropy Sum: {current_window_entropy:.2f})")
        
        if trigger_backtrack:
            if len(history_hidden_states) <= d:
                print(f"警告：历史不足，跳过回溯")
            else:
                generated_tokens = generated_tokens[:-d]
                token_entropies = token_entropies[:-d]

                current_len = past_key_values.get_seq_length()
                past_key_values.crop(current_len - d)

                history_hidden_states = history_hidden_states[:-d]
                start_hidden_state = history_hidden_states[-1] 

                beam_hidden_state = start_hidden_state.repeat(beam_size, 1)  
                past_key_values = expand_cache_for_beam(past_key_values, beam_size)

                # [FIX 1]: 必须初始化为 -inf 避免所有 Beam 第一步坍缩为同一序列
                beam_scores = torch.full((beam_size,), -1e9, device=_device)
                beam_scores[0] = 0.0 
                
                beam_seqs = torch.zeros((beam_size, 0), dtype=torch.long, device=_device)
                beam_history_hiddens = [beam_hidden_state]

                for step in range(d):
                    curr_h = beam_history_hiddens[-1]
                    b_logits = mmgpt.gen_head(curr_h) 
                    
                    b_logits_view = b_logits.view(beam_size, 2, -1)
                    b_cond = b_logits_view[:, 0, :]
                    b_uncond = b_logits_view[:, 1, :]
                    b_final_logits = b_uncond + cfg_weight * (b_cond - b_uncond)
                    
                    b_log_probs = F.log_softmax(b_final_logits, dim=-1) 
                    
                    next_scores = beam_scores.unsqueeze(1) + b_log_probs
                    next_scores_flat = next_scores.view(-1)
                    
                    topk_scores, topk_indices = torch.topk(next_scores_flat, beam_size)
                    
                    beam_indices = topk_indices // b_final_logits.shape[-1] 
                    token_indices = topk_indices % b_final_logits.shape[-1]  
                    
                    beam_scores = topk_scores
                    beam_seqs = torch.cat([beam_seqs[beam_indices], token_indices.unsqueeze(1)], dim=1)
                    
                    cache_indices = []
                    for b_idx in beam_indices:
                        cache_indices.extend([2 * b_idx.item(), 2 * b_idx.item() + 1])
                    cache_indices_tensor = torch.tensor(cache_indices, device=_device)
                    
                    past_key_values.reorder_cache(cache_indices_tensor)
                    
                    # [FIX 3]: 必须同步重排历史隐状态，否则路径提取会交叉错乱
                    for i in range(len(beam_history_hiddens)):
                        beam_history_hiddens[i] = beam_history_hiddens[i][cache_indices_tensor]
                    
                    next_tokens_input = token_indices.repeat_interleave(2) 
                    
                    img_embeds = mmgpt.prepare_gen_img_embeds(next_tokens_input) 
                    img_embeds = img_embeds.unsqueeze(1) 
                    
                    b_outputs = mmgpt.language_model.model(
                        inputs_embeds=img_embeds, use_cache=True, past_key_values=past_key_values
                    )
                    
                    new_h = b_outputs.last_hidden_state[:, -1, :] 
                    beam_history_hiddens.append(new_h)
            
                best_idx = torch.argmax(beam_scores).item() 
                winner_tokens = beam_seqs[best_idx].tolist()
                generated_tokens.extend(winner_tokens)
                token_entropies.extend([0.0] * d)

                winner_cache = DynamicCache()
                for layer_idx in range(len(past_key_values)):
                    k, v = past_key_values[layer_idx]
                    winner_cache.update(
                        k[2*best_idx : 2*best_idx+2], 
                        v[2*best_idx : 2*best_idx+2], 
                        layer_idx
                    )
                past_key_values = winner_cache

                for bh in beam_history_hiddens[1:]:
                    wh = bh[2*best_idx : 2*best_idx+2, :]
                    history_hidden_states.append(wh)

                last_hidden_state = history_hidden_states[-1]
                t_slow = t_fast
                continue
        

    print("Generation finished. Decoding image...")
    gen_ids = torch.tensor(generated_tokens, dtype=torch.int).unsqueeze(0).cuda()
    
    dec = mmgpt.gen_vision_model.decode_code(
        gen_ids, 
        shape=[1, 8, image_token_num_per_image//24, image_token_num_per_image//24] 
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
img_array = generate_image_with_tts(vl_gpt, vl_chat_processor, prompt, sigma=8.0)
PIL.Image.fromarray(img_array).save("result_2.jpg")