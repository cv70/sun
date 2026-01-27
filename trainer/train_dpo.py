"""
=============================================================================
DPO训练模块 (train_dpo.py)
=============================================================================

【模块作用】
实现直接偏好优化(DPO)的训练流程

【相关知识】
1. DPO (Direct Preference Optimization): 直接使用偏好数据优化模型
2. 参考模型 (Reference Model): 用于计算基线log概率，不参与训练
3. 梯度裁剪 (Gradient Clipping): 防止梯度爆炸
4. chosen vs rejected: 优选回答 vs 拒绝回答
"""

import glob
import os
import sys
import time
import torch
import swanlab
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import LLM_CONFIG
from utils.generate import generate_and_print_sample
from utils.loss import calc_dpo_loss_batch
from utils.eval import evaluate_dpo_model
from dataset.dpo_data_loader import create_dpo_dataloader_from_file
from model.llm import LLM
from tokenizer.tokenizer import Tokenizer

def train_dpo_epoch(model, ref_model, train_loader, val_loader, optimizer, device, num_epochs, start_context, tokenizer, eval_freq=500, eval_iter=2):
    """
    训练DPO模型

    参数:
        model: 要训练的模型
        ref_model: 参考模型（冻结参数，用于计算baseline）
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        start_context: 生成样本的起始文本
        tokenizer: 分词器
        eval_freq: 评估频率
        eval_iter: 评估批次数

    相关知识：
    - ref_model参数冻结，不参与训练
    - DPO损失鼓励模型生成chosen而非rejected
    """
    steps = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()  # 设置为训练模式

        for prompt_ids, chosen_ids, rejected_ids in train_loader:
            optimizer.zero_grad()  # 梯度清零
            loss = calc_dpo_loss_batch(prompt_ids, chosen_ids, rejected_ids, model, ref_model, device)  # 计算DPO损失
            loss.backward()  # 反向传播

            # 添加梯度裁剪
            # 防止梯度过大导致训练不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # 更新参数
            steps += 1

            if steps % eval_freq == 0:
                train_loss, val_loss = evaluate_dpo_model(model, ref_model, train_loader, val_loader, device, eval_iter)
                print(f"Epoch {epoch + 1} Step {steps}: Train loss {train_loss}, Val loss {val_loss}")

        # 生成样本
        generate_and_print_sample(model, tokenizer, device, start_context)
        print(f"Epoch {epoch + 1}/{num_epochs}, Cost Time: {time.time() - start_time:.2f}s")

def train_dpo():
    """执行DPO训练流程"""
    # 加载数据集
    dpo_filenames = sorted(glob.glob(os.path.join('../'+LLM_CONFIG['data_dir'], "*dpo.json")))

    tokenizer = Tokenizer("../tokenizer/"+LLM_CONFIG['tokenizer_path'])

    # 创建DPO数据加载器
    dpo_dataloader = create_dpo_dataloader_from_file(
        tokenizer,
        dpo_filenames,
        1,  # batch_size
        LLM_CONFIG["context_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    # 划分训练集和验证集
    total_data = len(dpo_dataloader.dataset)
    train_data_num = int(total_data * 0.9)
    train_data, val_data = random_split(dpo_dataloader.dataset, [train_data_num, total_data - train_data_num])
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练模型（SFT后的模型）
    model = LLM(LLM_CONFIG).to(device)
    model.load_state_dict(torch.load("../model/sun_sft.pth"))

    # 创建参考模型（与模型相同参数）
    ref_model = LLM(LLM_CONFIG).to(device)
    ref_model.load_state_dict(torch.load("../model/sun_sft.pth"))
    ref_model.eval()  # 设置为评估模式，冻结参数

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

    # 开始训练
    train_dpo_epoch(
        model,
        ref_model,
        train_loader,
        val_loader,
        optimizer,
        device,
        2,  # num_epochs
        "中农许多都占有土地",
        tokenizer,
        eval_freq= max(1, len(train_loader) // 3),
        eval_iter=2
    )

    # 保存模型
    save_path = "../model/sun_dpo.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_dpo()
