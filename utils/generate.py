import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.tokenizer import generate_text_simple


# 生成并打印样本
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    input_token_ids = tokenizer.text_to_token_ids(start_context)
    input_tensor = torch.tensor(input_token_ids).unsqueeze(0).to(device)

    output_tensor = generate_text_simple(
        model=model, idx=input_tensor,
        max_new_tokens=50, context_size=model.ctx_len
    )

    output_token_ids = output_tensor.squeeze(0).tolist()
    
    decoded_text = tokenizer.token_ids_to_text(output_token_ids)
    print(decoded_text.replace("\n", " "))  # 紧凑的打印格式
    model.train()


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
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)


        # 新增：应用温度缩放
        if temperature > 0.0:
            logits = logits / temperature


            # 新增（不在书中）：为了在mps设备上获得等效结果的数值稳定性提示
            # 在softmax之前减去行最大值
            logits = logits - logits.max(dim=-1, keepdim=True).values
            
            # 应用softmax获取概率
            probs = torch.softmax(logits, dim=-1)  # (批次大小, 上下文长度)


            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (批次大小, 1)


        # 否则与之前相同：获取词汇表中logits值最大的标记的索引
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (批次大小, 1)


        if idx_next == eos_id:  # 如果遇到结束序列标记且指定了eos_id，则提前停止生成
            break


        # 与之前相同：将采样的索引追加到运行序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (批次大小, 标记数+1)


    return idx
