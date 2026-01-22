import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.tokenizer import generate_text_simple


def generate_and_print_sample(model, tokenizer, device, prompt, max_new_tokens=50,
                           temperature=0.8, repetition_penalty=1.2):
    model.eval()
    input_ids = tokenizer.text_to_token_ids(prompt)
    input_ids = torch.tensor([input_ids], device=device)
    prompt_len = len(input_ids[0])

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]

            # 应用重复惩罚
            newly_generated = input_ids[0][prompt_len:].tolist()
            for token_id in set(newly_generated):
                if token_id < next_token_logits.shape[1]:
                    next_token_logits[0, token_id] /= repetition_penalty

            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # ✅ 修复：简化 EOS 检查条件
            if (hasattr(tokenizer, 'eos_id') and
                next_token.item() == tokenizer.eos_id):
                print(f"Hit EOS at position {input_ids.size(1)}, stopping.")
                break

    output_text = tokenizer.token_ids_to_text(input_ids[0].cpu().tolist())
    print(output_text)


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 循环与之前相同：获取logits，只关注最后的时间步
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 新增：使用top_k采样过滤logits
        if top_k is not None:
          # 只保留top_k个值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')), logits)

        # 新增：应用temperature缩放
        if temperature > 0:
            logits = logits / temperature

        # 应用softmax获取概率
        probs = torch.softmax(logits, dim=-1)

        if temperature > 0:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        # 检查是否达到EOS
        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx