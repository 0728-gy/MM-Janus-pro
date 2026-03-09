import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# 1. 模型加载 (沿用你的代码)
model_path = "/share/home/u11154/JingyiLiu/MM2026/Janus/model_weights/Janus-Pro-7B" # 或者 Janus-Pro-7B
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
        "images": ["images/equation.png"],
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
    max_new_tokens=512
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
    
    # 当前步的输入 Embedding (初始为整个 Prompt 拼接后的结果)
    current_inputs_embeds = inputs_embeds

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
     max_new_tokens=512
 )
 
answer = tokenizer.decode(tokens, skip_special_tokens=True)
print(f"生成内容: {answer}")