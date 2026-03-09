import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import torch.nn.functional as F
import PIL.Image

from transformers import DynamicCache

import os
import json
import argparse
from tqdm import tqdm


# ============================================================
# 核心生成函数（不变，保留 TTS 逻辑）
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
    cfg_weight: float = 10.0,
    image_token_num_per_image: int = 576,
    d: int = 8,
    sigma: float = 7.0,
    beam_size: int = 3,
) -> np.ndarray:
    """
    对单个 prompt 生成一张图，返回 np.ndarray (H, W, 3) uint8。
    """
    _device = next(mmgpt.parameters()).device

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).to(_device)

    tokens = torch.stack([input_ids.clone(), input_ids.clone()], dim=0)
    
    # 2. 对 Unconditional 分支 (索引为 1) 进行处理：
    # 保留第一个 Token (BOS) 和最后一个 Token (Image Start Tag)，
    # 将中间的所有描述性文字替换为 pad_id。
    tokens[1, 1:-1] = vl_chat_processor.pad_id 
    
    # 3. 将 Token 转换为 Embedding
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    outputs = mmgpt.language_model.model(
        inputs_embeds=inputs_embeds, use_cache=True, past_key_values=None
    )
    past_key_values = outputs.past_key_values
    last_hidden_state = outputs.last_hidden_state[:, -1, :]

    generated_tokens = []
    token_entropies = []
    history_hidden_states = []

    t_fast = 0
    t_slow = 0

    while len(generated_tokens) < image_token_num_per_image:

        history_hidden_states.append(last_hidden_state)

        logits = mmgpt.gen_head(last_hidden_state)
        logit_cond = logits[0, :]
        logit_uncond = logits[1, :]
        final_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        probs = torch.softmax(final_logits/0.95, dim=-1)
        log_probs = torch.log_softmax(final_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).item()

        next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens.append(next_token.item())
        token_entropies.append(entropy)

        next_token_input = next_token.repeat(2)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token_input)
        inputs_embeds = img_embeds.unsqueeze(1)

        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        last_hidden_state = outputs.last_hidden_state[:, -1, :]

        t_fast += 1
        t_slow += 1

        if t_fast - t_slow < d:
            t_slow -= 1

        trigger_backtrack = False
        if t_fast - t_slow == d:
            current_window_entropy = sum(token_entropies[-d:])
            if current_window_entropy > sigma and len(generated_tokens) > d:
                trigger_backtrack = True

        if trigger_backtrack:
            if len(history_hidden_states) <= d:
                pass  # 历史不足，跳过回溯
            else:
                generated_tokens = generated_tokens[:-d]
                token_entropies = token_entropies[:-d]

                current_len = past_key_values.get_seq_length()
                past_key_values.crop(current_len - d)

                history_hidden_states = history_hidden_states[:-d]
                start_hidden_state = history_hidden_states[-1]

                beam_hidden_state = start_hidden_state.repeat(beam_size, 1)
                past_key_values = expand_cache_for_beam(past_key_values, beam_size)

                beam_scores = torch.full((beam_size,), -1e9, device=_device)
                beam_scores[0] = 0.0 
                beam_seqs = torch.zeros(
                    (beam_size, 0), dtype=torch.long, device=_device
                )
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
                    beam_seqs = torch.cat(
                        [beam_seqs[beam_indices], token_indices.unsqueeze(1)], dim=1
                    )

                    cache_indices = []
                    for b_idx in beam_indices:
                        cache_indices.append(2 * b_idx)
                        cache_indices.append(2 * b_idx + 1)
                    past_key_values.reorder_cache(
                        torch.tensor(cache_indices, device=_device)
                    )

                    next_tokens_input = token_indices.repeat_interleave(2)
                    img_embeds = mmgpt.prepare_gen_img_embeds(next_tokens_input)
                    img_embeds = img_embeds.unsqueeze(1)

                    b_outputs = mmgpt.language_model.model(
                        inputs_embeds=img_embeds,
                        use_cache=True,
                        past_key_values=past_key_values,
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
                    wh = bh[2 * best_idx : 2 * best_idx + 2, :]
                    history_hidden_states.append(wh)

                last_hidden_state = history_hidden_states[-1]
                t_slow = t_fast
                continue

        # 正常推进
        

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
    """将文本 prompt 转为模型输入字符串（含 image_start_tag）。"""
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
    metadata_path: str,       # GenEval metadata JSONL 文件路径
    output_dir: str,          # 输出根目录
    num_images_per_prompt: int = 4,   # 每个 prompt 生成几张图
    cfg_weight: float = 10.0,
    image_token_num_per_image: int = 576,
    d: int = 8,
    sigma: float = 7.0,
    beam_size: int = 3,
    start_idx: int = 0,       # 断点续传：从第几个 prompt 开始
):
    """
    读取 GenEval metadata JSONL，批量生成图像。

    输出目录结构（与 GenEval evaluate.py 兼容）：
        output_dir/
          00000/
            samples_0.jpg
            samples_1.jpg
            ...
            metadata.jsonl   ← 每个子目录复制一份 metadata 方便 eval
          00001/
            ...

    metadata JSONL 每行格式示例：
        {"prompt": "a cat sitting on a mat", "category": "single_object", ...}
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取所有 prompts
    prompts = []
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    print(f"Total prompts: {len(prompts)}, generating {num_images_per_prompt} images each.")
    print(f"Starting from index: {start_idx}")

    for global_idx, meta in enumerate(tqdm(prompts, desc="Prompts")):
        if global_idx < start_idx:
            continue

        prompt_text = meta["prompt"]
        prompt_dir = os.path.join(output_dir, f"{global_idx:05d}")
        os.makedirs(prompt_dir, exist_ok=True)

        # 保存 metadata（GenEval evaluate.py 需要读这个）
        meta_out_path = os.path.join(prompt_dir, "metadata.jsonl")
        with open(meta_out_path, "w") as mf:
            mf.write(json.dumps(meta) + "\n")

        # 编码 prompt（只做一次）
        formatted_prompt = encode_prompt(vl_chat_processor, prompt_text)

        for img_idx in range(num_images_per_prompt):
            save_path = os.path.join(prompt_dir, f"samples_{img_idx}.jpg")

            # 断点续传：已有则跳过
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
                d=d,
                sigma=sigma,
                beam_size=beam_size,
            )

            PIL.Image.fromarray(img_array).save(save_path, quality=95)

    print(f"\nDone. Results saved to: {output_dir}")


# ============================================================
# 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Janus-Pro GenEval Image Generation with TTS")
    parser.add_argument("--model_path", type=str,
                        default="/share/home/u11154/JingyiLiu/MM2026/Janus/model_weights/Janus-Pro-7B")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="/share/home/u11154/JingyiLiu/MM2026/geneval/prompts/evaluation_metadata.jsonl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory")
    parser.add_argument("--num_images_per_prompt", type=int, default=4,
                        help="Number of images to generate per prompt")
    parser.add_argument("--cfg_weight", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=7.0,
                        help="Entropy threshold for TTS backtracking")
    parser.add_argument("--d", type=int, default=8,
                        help="Window size for TTS")
    parser.add_argument("--beam_size", type=int, default=3,
                        help="Beam size for TTS backtracking")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Resume from this prompt index (for checkpoint restart)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading model from: {args.model_path}")
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    print("Model loaded.")

    generate_for_geneval(
        mmgpt=vl_gpt,
        vl_chat_processor=vl_chat_processor,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        num_images_per_prompt=args.num_images_per_prompt,
        cfg_weight=args.cfg_weight,
        image_token_num_per_image=576,
        d=args.d,
        sigma=args.sigma,
        beam_size=args.beam_size,
        start_idx=args.start_idx,
    )
