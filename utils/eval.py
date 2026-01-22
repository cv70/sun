"""
=============================================================================
模型评估模块 (eval.py)
=============================================================================

【模块作用】
实现各种训练阶段的模型评估函数

【相关知识】
1. model.eval(): 设置模型为评估模式，关闭dropout等
2. torch.no_grad(): 关闭梯度计算以节省内存
3. 评估损失: 在验证集上计算损失
"""

import torch
from utils.loss import calc_loss_loader, calc_sft_loss_loader, calc_dpo_loss_loader

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    评估预训练模型

    参数:
        model: 语言模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        eval_iter: 评估的批次数

    返回:
        train_loss, val_loss
    """
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 恢复训练模式
    return train_loss, val_loss

def evaluate_sft_model(model, train_loader, val_loader, device, eval_iter):
    """评估SFT模型"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_sft_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_sft_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def evaluate_dpo_model(model, ref_model, train_loader, val_loader, device, eval_iter):
    """评估DPO模型"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_dpo_loss_loader(train_loader, model, ref_model, device, num_batches=eval_iter)
        val_loss = calc_dpo_loss_loader(val_loader, model, ref_model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
