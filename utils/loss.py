"""
=============================================================================
损失计算模块 (loss.py)
=============================================================================

【模块作用】
实现各种训练阶段的损失计算函数

【相关知识】
1. 交叉熵损失: 分类任务的标准损失函数
2. flatten: 展平张量以便计算损失
3. ignore_index: 忽略某些标签的损失
4. logprob: 对数概率
5. DPO损失: 直接偏好优化的损失函数
"""

import torch

# ============================================================================
# 预训练损失计算
# ============================================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算批次的交叉熵损失

    参数:
        input_batch: 输入token序列 (batch_size, seq_len)
        target_batch: 目标token序列 (batch_size, seq_len)
        model: 语言模型
        device: 计算设备

    返回:
        损失值（标量）

    相关知识：
    - 交叉熵衡量预测分布和真实分布的差异
    - 展平所有batch和seq维度，计算所有token的平均损失
    """
    # 将数据移动到指定设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # 模型前向传播，获取logits
    # logits形状: (batch_size, seq_len, vocab_size)
    logits = model(input_batch)

    # 展平logits和target
    # logits.flatten(0, 1): (batch_size * seq_len, vocab_size)
    # target_batch.flatten(): (batch_size * seq_len,)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算数据加载器的平均损失"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# ============================================================================
# SFT损失计算
# ============================================================================

def calc_sft_loss_batch(input_batch, target_batch, model, device):
    """
    计算SFT任务的批次损失

    参数:
        input_batch: 输入序列 (batch_size, seq_len)
        target_batch: 标签序列，prompt部分为-100 (batch_size, seq_len)
        model: 语言模型
        device: 计算设备

    返回:
        损失值（标量）

    相关知识：
    - 使用ignore_index=-100忽略prompt部分的损失
    - 只对答案部分计算损失
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # 模型前向传播
    logits = model(input_batch)

    # 展平以计算token级别损失
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = target_batch.reshape(-1)

    # 使用ignore_index=-100自动忽略prompt部分
    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=-100
    )
    return loss

def calc_sft_loss_loader(data_loader, model, device, num_batches=None):
    """计算SFT数据加载器的平均损失"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_sft_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / min(len(data_loader), num_batches) if num_batches else total_loss / len(data_loader)

# ============================================================================
# DPO损失计算
# ============================================================================

def calc_dpo_logprob_batch(prompt_batch, chosen_batch, rejected_batch, model, device, padding_token_id=0):
    """
    计算DPO的log概率

    参数:
        prompt_batch: prompt序列 (batch_size, seq_len)
        chosen_batch: 优选回答序列 (batch_size, seq_len)
        rejected_batch: 拒绝回答序列 (batch_size, seq_len)
        model: 语言模型
        device: 计算设备
        padding_token_id: padding token的ID

    返回:
        chosen_logprobs, rejected_logprobs

    相关知识：
    - DPO需要计算chosen和rejected序列的对数概率
    - 只计算回答部分的概率（忽略padding部分）
    """
    # 将数据移动到设备
    prompt_batch = prompt_batch.to(device)
    chosen_batch = chosen_batch.to(device)
    rejected_batch = rejected_batch.to(device)

    # 拼接prompt和response
    input_ids_chosen = torch.cat([prompt_batch, chosen_batch], dim=1)
    input_ids_rejected = torch.cat([prompt_batch, rejected_batch], dim=1)

    input_ids_chosen = input_ids_chosen.to(device)
    input_ids_rejected = input_ids_rejected.to(device)

    # 模型前向传播
    logits_chosen = model(input_ids_chosen)
    logits_rejected = model(input_ids_rejected)

    # 计算log概率
    logprobs_chosen = torch.log_softmax(logits_chosen, dim=-1)
    logprobs_rejected = torch.log_softmax(logits_rejected, dim=-1)

    # 创建mask（非padding的位置）
    chosen_mask = (chosen_batch != padding_token_id).float()
    rejected_mask = (rejected_batch != padding_token_id).float()

    # 提取每个token的log概率
    # 注意：logprobs_chosen的序列长度包含prompt+chosen，所以索引也应该是完整的input_ids_chosen
    chosen_logprobs_seq = logprobs_chosen.gather(dim=-1, index=input_ids_chosen.unsqueeze(-1)).squeeze(-1)
    rejected_logprobs_seq = logprobs_rejected.gather(dim=-1, index=input_ids_rejected.unsqueeze(-1)).squeeze(-1)

    # 计算总log概率（应用mask）
    # 注意：mask仍然只需要对response部分应用
    chosen_logprobs = (chosen_logprobs_seq[:, -chosen_batch.shape[1]:] * chosen_mask).sum(dim=-1)
    rejected_logprobs = (rejected_logprobs_seq[:, -rejected_batch.shape[1]:] * rejected_mask).sum(dim=-1)
    return chosen_logprobs, rejected_logprobs

def calc_dpo_loss_batch(prompt_batch, chosen_batch, rejected_batch, model, ref_model, device, beta=0.1, padding_token_id=0):
    """
    计算DPO损失

    参数:
        beta: DPO的温度参数

    相关知识：
    - DPO损失鼓励模型生成chosen而非rejected
    - 使用参考模型进行归一化
    """
    # 计算训练模型的logprob
    chosen_logprobs, rejected_logprobs = calc_dpo_logprob_batch(prompt_batch, chosen_batch, rejected_batch, model, device, padding_token_id)
    # 计算参考模型的logprob
    ref_chosen_logprobs, ref_rejected_logprobs = calc_dpo_logprob_batch(prompt_batch, chosen_batch, rejected_batch, ref_model, device, padding_token_id)

    # DPO损失公式
    logits = beta * ((chosen_logprobs - rejected_logprobs) - (ref_chosen_logprobs - ref_rejected_logprobs))

    # softplus(x) = log(1 + exp(x)), 数值稳定的log sigmoid实现
    loss = torch.nn.functional.softplus(-logits).mean()
    return loss

def calc_dpo_loss_loader(data_loader, model, ref_model, device, num_batches=None, beta=0.1, padding_token_id=0):
    """计算DPO数据加载器的平均损失"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (prompt_batch, chosen_batch, rejected_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_dpo_loss_batch(prompt_batch, chosen_batch, rejected_batch, model, ref_model, device, beta, padding_token_id)
            total_loss += loss.item()
        else:
            break
    return total_loss / min(len(data_loader), num_batches) if num_batches else total_loss / len(data_loader)
