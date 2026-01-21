import glob
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """
    steps = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        for prompt_ids, chosen_ids, rejected_ids in train_loader:
            optimizer.zero_grad() # 梯度清零
            loss = calc_dpo_loss_batch(prompt_ids, chosen_ids, rejected_ids, model, ref_model, device)
            loss.backward() # 反向传播，计算损失梯度
            
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step() # 更新参数
            steps += 1

            if steps % eval_freq == 0:
                train_loss, val_loss = evaluate_dpo_model(model, ref_model, train_loader, val_loader, device, eval_iter)
                print(f"Epoch {epoch + 1} Step {steps}: Train loss {train_loss}, Val loss {val_loss}")
        
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
        print(f"Epoch {epoch + 1}/{num_epochs}, Cost Time: {time.time() - start_time:.2f}s")


def train_dpo():
    """
    执行DPO训练流程
    """
    # 加载数据集
    dpo_filenames = sorted(glob.glob(os.path.join(LLM_CONFIG['data_dir'], "*dpo.json")))
    
    tokenizer = Tokenizer(LLM_CONFIG['tokenizer_path'])

    # 划分训练集和验证集
    dpo_dataloader = create_dpo_dataloader_from_file(
        tokenizer,
        dpo_filenames,
        1,  # SFT通常使用较小的batch size
        LLM_CONFIG["context_length"], 
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )
    total_data = len(dpo_dataloader.dataset)
    train_data_num = int(total_data * 0.9)
    train_data, val_data = random_split(dpo_dataloader.dataset, [train_data_num, total_data - train_data_num])
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # 创建模型并移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练模型
    model = LLM(LLM_CONFIG).to(device)
    model.load_state_dict(torch.load("model/sun_sft.pth"))

    ref_model = LLM(LLM_CONFIG).to(device)
    ref_model.load_state_dict(torch.load("model/sun_sft.pth"))
    ref_model.eval()  # 固定参考模型参数

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)

    train_dpo_epoch(
        model, 
        ref_model,
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        2, 
        "独孤求败一生无敌，为何从未收徒？", 
        tokenizer,
        eval_freq=500,
        eval_iter=2
    )

    # 保存模型
    save_path = "model/sun_dpo.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_dpo()
