"""
=============================================================================
预训练模块 (train_pretrain.py)
=============================================================================

【模块作用】
实现大语言模型的预训练流程

【相关知识】
1. 预训练 (Pre-training): 在大规模文本上训练基础语言模型
2. AdamW优化器: 带权重衰减的自适应优化器
3. 学习率 (Learning Rate): 控制参数更新步长的超参数
4. 权重衰减 (Weight Decay): L2正则化，防止过拟合
5. 数据划分: 训练集/验证集分离
6. 梯度清零: optimizer.zero_grad()
7. 反向传播: loss.backward()
8. 参数更新: optimizer.step()
"""

import glob
import os
import sys
import time
import swanlab
import torch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from torch.utils.data import DataLoader, random_split
from model.llm import LLM
from config.config import LLM_CONFIG
from utils.loss import calc_loss_batch
from utils.generate import generate_and_print_sample
from utils.eval import evaluate_model
from dataset.data_loader import create_dataloader_from_txt_file
from tokenizer.tokenizer import Tokenizer

def train_epoch(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    start_context,
    tokenizer,
    eval_freq=max(1, len(train_loader) // 3),  # 评估频率
    eval_iter=5
):
    """
    训练一个epoch

    参数:
        model: 语言模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        start_context: 生成样本的起始文本
        tokenizer: 分词器
        eval_freq: 评估频率（每多少步评估一次）
        eval_iter: 评估时的批次数
    """
    steps = 0
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f" Starting Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")

        start_time = time.time()
        model.train()  # 设置为训练模式

        # 遍历训练数据
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            steps += 1

            # 每10步打印一次损失
            if steps % 10 == 0:
                print(f"[Epoch {epoch+1}] Step {steps:5d} | Loss: {loss.item():.4f}")

            # 定期评估模型
            if steps % eval_freq == 0:
                print(f"\n Evaluating at Step {steps}...")
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                print(f" Eval Result -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")

        # 每个epoch结束后生成样本文本
        print(f"\n Generating sample after epoch {epoch + 1}:")
        generate_and_print_sample(model, tokenizer, device, start_context)
        print(f"\n Epoch {epoch + 1} finished in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    # 加载数据文件
    text_filenames = sorted(glob.glob(os.path.join("../"+LLM_CONFIG['data_dir'], "*.md")))
    tokenizer = Tokenizer("../tokenizer/"+LLM_CONFIG['tokenizer_path'])

    # 创建数据加载器
    txt_dataloader = create_dataloader_from_txt_file(
        tokenizer,
        text_filenames,
        48,  # batch_size
        LLM_CONFIG["context_length"],
        1,  # stride
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    # 划分训练集和验证集
    total_data = len(txt_dataloader.dataset)
    train_data_num = int(total_data * 0.9)
    train_data, val_data = random_split(txt_dataloader.dataset, [train_data_num, total_data - train_data_num])
    train_loader = DataLoader(train_data, batch_size=txt_dataloader.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=txt_dataloader.batch_size, shuffle=False)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型
    model = LLM(LLM_CONFIG).to(device)

    # 创建优化器 (AdamW)
    # lr: 学习率，控制参数更新的步长
    # weight_decay: 权重衰减，相当于L2正则化
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

    # 开始训练
    train_epoch(
        model, train_loader, val_loader, optimizer, device, 2, "国民党反动派", tokenizer
    )

    # 保存模型
    save_path = "../model/sun_base.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
