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
    # input_batch: (batch_size, seq_len)
    # target_batch: (batch_size)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)  # (batch_size, seq_len, vocab_size)

    # 只计算整个序列预测出来的下一词的损失
    logits = logits[:, -1, :] # (batch_size, vocab_size)
    print(f"logits shape: {logits.shape}")
    print(f"target_batch shape: {target_batch.shape}")

    return torch.nn.functional.cross_entropy(
        logits, 
        target_batch
    )

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
