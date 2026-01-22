import glob
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llm import LLM
from config.config import LLM_CONFIG
from utils.loss import calc_sft_loss_batch
from utils.generate import generate_and_print_sample
from utils.eval import evaluate_sft_model
from dataset.sft_data_loader import create_sft_dataloader_from_file
from tokenizer.tokenizer import Tokenizer

def train_sft_epoch(model, train_loader, val_loader, optimizer, device, num_epochs, start_context, tokenizer, eval_freq=1000, eval_iter=5):
    """
    SFT训练一个epoch
    """
    steps = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 梯度清零
            loss = calc_sft_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 反向传播，计算损失梯度
            optimizer.step() # 更新参数
            steps += 1

            if steps % eval_freq == 0:
                train_loss, val_loss = evaluate_sft_model(model, train_loader, val_loader, device, eval_iter)
                print(f"Epoch {epoch + 1} Step {steps}: Train loss {train_loss}, Val loss {val_loss}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
        print(f"Epoch {epoch + 1}/{num_epochs}, Cost Time: {time.time() - start_time:.2f}s")
def collate_fn(batch):
    input_ids, labels = zip(*batch)
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return input_ids_padded, labels_padded

def train_sft():
    """
    执行SFT训练流程
    """
    # 加载数据集
    sft_filenames = sorted(glob.glob(os.path.join('../'+LLM_CONFIG['data_dir'], "*sft.json")))
    
    tokenizer = Tokenizer("../tokenizer/"+LLM_CONFIG['tokenizer_path'])
    
    # 创建SFT数据加载器
    sft_dataloader = create_sft_dataloader_from_file(
        tokenizer,
        sft_filenames,
        1, 
        LLM_CONFIG["context_length"], 
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )

    # 划分训练集和验证集
    total_data = len(sft_dataloader.dataset)
    train_data_num = int(total_data * 0.9)
    train_data, val_data = random_split(sft_dataloader.dataset, [train_data_num, total_data - train_data_num])
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False,collate_fn=collate_fn)

    # 创建模型并移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练模型
    model = LLM(LLM_CONFIG).to(device)
    model.load_state_dict(torch.load("../model/sun_base.pth"))

    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    
    # 开始训练
    print("Starting SFT training...")
    train_sft_epoch(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        10, 
        "在毛泽东同志的带领下", 
        tokenizer,
        eval_freq=500,
        eval_iter=2
    )
    
    # 保存模型
    save_path = "../model/sun_sft.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_sft()
