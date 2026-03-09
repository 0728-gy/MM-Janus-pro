# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
import matplotlib.pyplot as plt
import cv2 # 需要 pip install opencv-python

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
        "content": "A glowing crystal ball floating above a sandstone table in the middle of a desert at sunset.",
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
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 0.95,
    parallel_size: int = 4,
    cfg_weight: float = 8,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    
    # -------------------------------------------------------
    # [新增] 1. 准备一个列表存储每一步的熵
    # -------------------------------------------------------
    all_entropies = [] 

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        # -------------------------------------------------------
        # [新增] 2. 计算当前 Token 的熵并保存
        # -------------------------------------------------------
        # probs 维度是 (parallel_size, vocab_size)
        step_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) # (parallel_size,)
        all_entropies.append(step_entropy)
        # -------------------------------------------------------

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    # -------------------------------------------------------
    # [新增] 3. 将熵列表转换为空间网格 (24x24)
    # -------------------------------------------------------
    # 转换后维度为 (parallel_size, 24, 24)
    grid_size = img_size // patch_size # 384 // 16 = 24
    entropies_np = torch.stack(all_entropies).to(torch.float32).cpu().numpy().T
    entropy_maps = entropies_np.reshape(parallel_size, grid_size, grid_size)
    # -------------------------------------------------------

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    # -------------------------------------------------------
    # [修改/新增] 4. 修改保存逻辑，将熵图和原图画在一起
    # -------------------------------------------------------


    os.makedirs('generated_samples_2', exist_ok=True)
    for i in range(parallel_size):
        # 原图数据
        current_img = visual_img[i]
        # 对应的熵网格
        current_entropy = entropy_maps[i]
        
        # 将 24x24 的熵图放大到 384x384，方便和原图叠加
        entropy_resized = cv2.resize(current_entropy, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        # 绘图：左边原图，右边带熵热力图的图
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        ax[0].imshow(current_img)
        ax[0].set_title("Generated Image")
        ax[0].axis('off')
        
        # 热力图叠加展示
        ax[1].imshow(current_img) # 底图
        im = ax[1].imshow(entropy_resized, cmap='jet', alpha=0.5) # 叠加半透明热力图
        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        ax[1].set_title("Entropy Heatmap Overlay")
        ax[1].axis('off')

        save_path = os.path.join('generated_samples_2', f"img_with_entropy_{i}.png")
        fig.tight_layout() 
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"已保存带熵分析的图至: {save_path}")

generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
)