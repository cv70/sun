import torch

# 计算批次的损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch) # (batch_size, seq_len, vocab_size)

    # logits.flatten(0, 1) (batch_size * seq_len, vocab_size)
    # target_batch.flatten() (batch_size * seq_len)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# 计算数据加载器的总损失（平均损失）
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果num_batches超过数据加载器中的批次数，则将批次数减少到匹配
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# 计算SFT任务的批次损失
def calc_sft_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)  # (batch_size, seq_len, vocab_size)

    # 展平 logits 和 target 以计算 token 级别 loss
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)        # (batch_size * seq_len, vocab_size)
    targets_flat = target_batch.reshape(-1)             # (batch_size * seq_len,)

    # 使用 ignore_index=-100 自动忽略 prompt 部分
    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=-100
    )
    return loss

# 计算SFT任务的数据加载器的总损失（平均损失）
def calc_sft_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果num_batches超过数据加载器中的批次数，则将批次数减少到匹配
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_sft_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / min(len(data_loader), num_batches) if num_batches else total_loss / len(data_loader)

# 计算DPO任务批次损失
def calc_dpo_logprob_batch(prompt_batch, chosen_batch, rejected_batch, model, device, padding_token_id=0):
    # prompt_batch: (batch_size, seq_len)
    # chosen_batch: (batch_size, seq_len)
    # rejected_batch: (batch_size, seq_len)
    prompt_batch = prompt_batch.to(device)
    chosen_batch = chosen_batch.to(device)
    rejected_batch = rejected_batch.to(device)

    # 拼接 prompt 和 response
    input_ids_chosen = torch.cat([prompt_batch, chosen_batch], dim=1)
    input_ids_rejected = torch.cat([prompt_batch, rejected_batch], dim=1)

    input_ids_chosen = input_ids_chosen.to(device)
    input_ids_rejected = input_ids_rejected.to(device)

    logits_chosen = model(input_ids_chosen) # (batch_size, seq_len, vocab_size)
    logits_rejected = model(input_ids_rejected) # (batch_size, seq_len, vocab_size)

    # 计算每个位置的 logprob
    logprobs_chosen = torch.log_softmax(logits_chosen, dim=-1) # (batch_size, seq_len, vocab_size)
    logprobs_rejected = torch.log_softmax(logits_rejected, dim=-1) # (batch_size, seq_len, vocab_size)

    # 创建 mask：True 表示非 padding
    chosen_mask = (chosen_batch != padding_token_id).float()  # (B, L)
    rejected_mask = (rejected_batch != padding_token_id).float()  # (B, L)

    # 计算最终的logprob（可能需要mask）
    chosen_logprobs_seq = logprobs_chosen.gather(dim=-1, index=chosen_batch.unsqueeze(-1)).squeeze(-1) # (batch_size, seq_len)
    rejected_logprobs_seq = logprobs_rejected.gather(dim=-1, index=rejected_batch.unsqueeze(-1)).squeeze(-1) # (batch_size, seq_len)

    # 计算最终的 logprob
    chosen_logprobs = (chosen_logprobs_seq * chosen_mask).sum(dim=-1)  # (B,)
    rejected_logprobs = (rejected_logprobs_seq * rejected_mask).sum(dim=-1)  # (B,)
    return chosen_logprobs, rejected_logprobs

def calc_dpo_loss_batch(prompt_batch, chosen_batch, rejected_batch, model, ref_model, device, beta=0.1, padding_token_id=0):
    # 计算训练模型的 logprob
    chosen_logprobs, rejected_logprobs = calc_dpo_logprob_batch(prompt_batch, chosen_batch, rejected_batch, model, device, padding_token_id)
    # 计算参考模型的 logprob
    ref_chosen_logprobs, ref_rejected_logprobs = calc_dpo_logprob_batch(prompt_batch, chosen_batch, rejected_batch, ref_model, device, padding_token_id)

    # 计算DPO损失
    # DPO公式：β * [(logπ_θ(y_w|x) - logπ_θ(y_l|x)) - (logπ_ref(y_w|x) - logπ_ref(y_l|x))]
    logits = beta * ((chosen_logprobs - rejected_logprobs) - (ref_chosen_logprobs - ref_rejected_logprobs))
    
    # 防止数值不稳定
    # DPO 损失: -log σ(logits)
    # σ(x) = 1 / (1 + exp(-x))
    # -log σ(x) = log(1 + exp(-x))
    loss = torch.nn.functional.softplus(-logits).mean()
    return loss


# 计算DPO任务数据加载器的总损失（平均损失）
def calc_dpo_loss_loader(data_loader, model, ref_model, device, num_batches=None, beta=0.1, padding_token_id=0):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果num_batches超过数据加载器中的批次数，则将批次数减少到匹配
        num_batches = min(num_batches, len(data_loader))
    for i, (prompt_batch, chosen_batch, rejected_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_dpo_loss_batch(prompt_batch, chosen_batch, rejected_batch, model, ref_model, device, beta, padding_token_id)
            total_loss += loss.item()
        else:
            break
    return total_loss / min(len(data_loader), num_batches) if num_batches else total_loss / len(data_loader)
