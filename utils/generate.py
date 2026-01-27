"""
=============================================================================
文本生成模块 (generate.py)
=============================================================================

【模块作用】
实现各种文本生成策略

【相关知识】
1. Temperature: 控制生成的随机性，越高越随机
2. Top-K采样: 只从概率最高的K个token中采样
3. 重复惩罚: 减少重复内容的生成
4. 多项式采样: 根据概率分布采样
5. EOS token: 序列结束标记
"""

import torch

def generate_and_print_sample(model, tokenizer, device, prompt, max_new_tokens=50,
                           temperature=0.8, repetition_penalty=1.2):
    """
    生成并打印样本文本

    参数:
        model: 语言模型
        tokenizer: 分词器
        device: 计算设备
        prompt: 提示词
        max_new_tokens: 最大生成token数
        temperature: 温度参数（控制随机性）
        repetition_penalty: 重复惩罚系数
    """
    model.eval()  # 设置为评估模式
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

            # 应用temperature缩放
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)

            # 多项式采样
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接新token
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 检查EOS
            if (hasattr(tokenizer, 'eos_id') and
                next_token.item() == tokenizer.eos_id):
                print(f"Hit EOS at position {input_ids.size(1)}, stopping.")
                break

    output_text = tokenizer.token_ids_to_text(input_ids[0].cpu().tolist())
    print(output_text)

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    高级文本生成函数

    参数:
        model: 语言模型
        idx: 输入序列 (batch_size, seq_len)
        max_new_tokens: 最大生成token数
        context_size: 上下文窗口大小
        temperature: 温度参数（0表示贪婪解码）
        top_k: Top-K采样的K值（None表示不使用）
        eos_id: EOS token ID

    返回:
        生成的完整序列

    相关知识：
    - temperature=0: 贪婪解码（每次选概率最高的）
    - temperature>0: 采样解码（增加多样性）
    - top_k: 只从概率最高的K个token中采样
    """
    for _ in range(max_new_tokens):
        # 裁剪到上下文长度
        idx_cond = idx[:, -context_size:]

        # 模型推理
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        logits = logits[:, -1, :]

        # Top-K采样过滤
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')), logits)

        # Temperature缩放
        if temperature > 0:
            logits = logits / temperature

        # Softmax转概率
        probs = torch.softmax(logits, dim=-1)

        # 选择下一个token
        if temperature > 0:
            # 采样解码
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # 贪婪解码
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        # 检查EOS
        if eos_id is not None and idx_next.item() == eos_id:
            break

        # 拼接新token
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
