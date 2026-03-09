import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import torch.nn.functional as F


# 1. 模型加载 (沿用你的代码)
model_path = "/share/home/u11154/JingyiLiu/MM2026/Janus/model_weights/Janus-Pro-1B" # 或者 Janus-Pro-7B
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# 2. 准备输入 (沿用你的代码)
conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>\nConvert the formula into latex code.",
        "images": ["/share/home/u11154/JingyiLiu/MM2026/Janus/images/equation.png"],
    },
    {"role": "Assistant", "content": ""},
]

pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

# 3. 获取初始的多模态 Embedding
# 这包含了图像 Embedding 和文字 Prompt 的 Embedding
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)


@torch.inference_mode()
def generate_text(
    vl_gpt, 
    tokenizer, 
    inputs_embeds, 
    attention_mask, 
    sigma,
    max_new_tokens=512,
    d=4
):
    """
    封装后的 Janus-Pro 文字生成函数，支持 Hidden States 提取。
    
    Args:
        vl_gpt: 加载好的 MultiModalityCausalLM 模型
        tokenizer: 对应的 tokenizer
        inputs_embeds: 由 prepare_inputs_embeds 得到的初始 Embedding (图像+文本)
        attention_mask: 初始的 attention_mask
        max_new_tokens: 最大生成长度
        
    Returns:
        generated_tokens: 生成的 token id 列表
        all_hidden_states: 每一帧的 hidden_states (List of Tensors)
    """
    
    # 1. 初始化参数
    eos_token_id = tokenizer.eos_token_id
    past_key_values = None
    generated_tokens = []
    all_hidden_states = []
    token_entropys=[]
    
    # 当前步的输入 Embedding (初始为整个 Prompt 拼接后的结果)
    current_inputs_embeds = inputs_embeds
    t_fast=0
    t_slow=0

    # 2. 推理循环
    for i in range(max_new_tokens):
        # 运行 Transformer 前向传播
        # use_cache=True 配合 past_key_values 是推理加速的关键
        outputs = vl_gpt.language_model.model(
            inputs_embeds=current_inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True, 
        )
        
        
        # 获取当前步的隐藏层状态 (Hidden States) -> 供 TTS 使用
        # 取最后一层输出的最后一个位置: (batch, 1, hidden_dim)
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        all_hidden_states.append(last_hidden_state) 
       
        # 获取 Logits 并通过 lm_head 预测下一个 Token
        # 注意：这里直接取 last_hidden_state 计算 logits
        logits = vl_gpt.language_model.lm_head(last_hidden_state)
      

        probs=F.softmax(logits,dim=-1)
        log_probs=F.log_softmax(probs,dim=-1)
        entropy=-torch.sum(log_probs*probs,dim=-1)
        token_entropys.append(entropy.item()) #entropy张量变成python数字

        # 贪心搜索 (Greedy Search)
        next_token = torch.argmax(logits, dim=-1)

        # 存储生成的 Token
        generated_tokens.append(next_token.item())

        # 检查是否生成了结束符
        if next_token.item() == eos_token_id:
            break

        # 3. 为下一轮迭代准备
        # 更新 KV Cache
        past_key_values = outputs.past_key_values
        
        
        # 更新 attention_mask: 在右侧追加一个 1
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=vl_gpt.device)], 
            dim=-1
        )
        
        #TTS算法
        t_fast+=1
        t_slow+=1
        if t_fast-t_slow<d:
            t_slow-=1
        elif t_fast-t_slow==d:
            sum_entropy_of_d=sum(token_entropys[-d:])
            if sum_entropy_of_d>sigma:
                #回溯删除
                past_key_values.crop(-d)
                attention_mask=attention_mask[:,:-d]
                token_entropys=token_entropys[:-d]
                generated_tokens=generated_tokens[:-d]
                all_hidden_states=all_hidden_states[:-d]
                
                #beam search
                #设置beam size
                #扩展batch，用于beam search
                #初始化必要的序列
            
                beam_size=4
                last_token_id = generated_tokens[-1]
                past_key_values.reorder_cache(torch.zeros(beam_size, dtype=torch.long, device=vl_gpt.device))
                beam_attention_mask = attention_mask.repeat(beam_size, 1)
                beam_next_token = torch.full((beam_size,), last_token_id, device=vl_gpt.device)
                beam_scores = torch.zeros(beam_size, device=vl_gpt.device)
                beam_sequences = torch.empty((beam_size, 0), dtype=torch.long, device=vl_gpt.device)
                beam_entropies = torch.empty((beam_size, 0), device=vl_gpt.device)
                beam_hiddens = torch.empty((beam_size, 0, outputs.last_hidden_state.shape[-1]), 
                           device=vl_gpt.device, dtype=torch.bfloat16)

                #开始循环
                for _ in range(d):
                    beam_inputs_embeds = vl_gpt.language_model.get_input_embeddings()(beam_next_token).unsqueeze(1)
                    outputs = vl_gpt.language_model.model(
                        inputs_embeds=beam_inputs_embeds,
                        past_key_values=past_key_values,
                        attention_mask=beam_attention_mask,
                        use_cache=True
                    )
                    logits = vl_gpt.language_model.lm_head(outputs.last_hidden_state[:, -1, :])
                    log_probs = F.log_softmax(logits, dim=-1) # [beam_size, vocab_size]

                    #计算熵
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    current_entropy = -torch.sum(probs * log_probs, dim=-1) # [beam_size]

                    # 计算累积得分 (当前路径得分 + 新词对数概率)
                    # beam_scores[:, None] 将 [4] 变为 [4, 1] 以便广播
                    next_scores = log_probs + beam_scores[:, None]

                     # 在整个 beam_size * vocab_size 的空间里选出前 beam_size 个最高分
                    top_scores, top_indices = torch.topk(next_scores.view(-1), beam_size, sorted=True)

                    # 计算这些最高分分别属于哪个旧分支，以及对应的词 ID
                    beam_indices = top_indices // logits.shape[-1]  # 属于哪个 beam 分支
                    token_indices = top_indices % logits.shape[-1]   # 词 ID

                    current_h = outputs.last_hidden_state[:, -1, :].unsqueeze(1) # [beam_size, 1, hidden_dim]
                    beam_hiddens = torch.cat([beam_hiddens[beam_indices], current_h], dim=1)

                    beam_entropies = torch.cat([
                        beam_entropies[beam_indices], 
                        current_entropy[beam_indices].unsqueeze(-1)
                    ], dim=-1)

                    past_key_values.reorder_cache(beam_indices) 
                    beam_scores = top_scores
                    beam_next_token = token_indices

                    beam_sequences = torch.cat([beam_sequences[beam_indices], token_indices.unsqueeze(-1)], dim=-1)
                    beam_attention_mask = torch.cat([beam_attention_mask, torch.ones((beam_size, 1), device=vl_gpt.device)], dim=-1)
                
                # 4. 从集束中选出得分最高的那一条路径
                best_beam_idx = torch.argmax(beam_scores)
                winner_tokens = beam_sequences[best_beam_idx].tolist()
                winner_entropies = beam_entropies[best_beam_idx].tolist()
                winner_hiddens = beam_hiddens[best_beam_idx]
                
                # 5. 将优胜路径结果同步回主循环变量
                token_entropys.extend(winner_entropies)
                generated_tokens.extend(winner_tokens)
                all_hidden_states.extend([winner_hiddens[m:m+1] for m in range(d)])

                # 将 batch 维度缩回 1，只保留胜出的那条路径的 Cache
                past_key_values.reorder_cache(torch.tensor([best_beam_idx], device=vl_gpt.device))
                # 更新全局掩码
                attention_mask = beam_attention_mask[best_beam_idx:best_beam_idx+1]
                # 更新 next_token 为胜出序列的最后一个
                next_token = torch.tensor([winner_tokens[-1]], device=vl_gpt.device)
                #slow指针同步到fast指针位置
                t_slow=t_fast
            
        # 获取下一个输入 token 的 Embedding (只取这一个词)
        # 从第二步开始，inputs_embeds 维度应为 (batch, 1, hidden_dim)
        current_inputs_embeds = vl_gpt.language_model.get_input_embeddings()(next_token).unsqueeze(1)

    return generated_tokens, all_hidden_states


# -----------------------------------------------------------
# 使用示例：
# -----------------------------------------------------------
tokens, hiddens = generate_text(
     vl_gpt=vl_gpt,
     tokenizer=tokenizer,
     inputs_embeds=inputs_embeds,
     attention_mask=prepare_inputs.attention_mask,
     max_new_tokens=512,
     d=4,
     sigma=5
 )
 
answer = tokenizer.decode(tokens, skip_special_tokens=True)
print(f"生成内容: {answer}")