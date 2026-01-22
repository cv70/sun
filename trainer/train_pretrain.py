import glob
import os
import sys

import time
import swanlab
import torch
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
    eval_freq=max(1, len(train_loader) // 3),     # 评估频率，每训练多少个step（batch）进行一次验证评估 
    eval_iter=5
):
    steps = 0
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f" Starting Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        start_time = time.time()
        model.train()
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            steps += 1

            if steps % 10 == 0:
                # 每个batch记录一次
                # swanlab.log({"loss": loss.item(),"step": steps,"epoch": epoch})
                print(f"[Epoch {epoch+1}] Step {steps:5d} | Loss: {loss.item():.4f}")

            if steps % eval_freq == 0:
                print(f"\n Evaluating at Step {steps}...")
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                print(f" Eval Result → Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")

        print(f"\n Generating sample after epoch {epoch + 1}:")
        generate_and_print_sample(model, tokenizer, device, start_context)
        print(f"\n Epoch {epoch + 1} finished in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    
    # swanlab.init(
    #     experiment_name="sun-pretrain",      # 实验名称
    #     description="SUN模型预训练",          # 实验描述
    #     project="sun",                       # 项目名称
    #     config={                             # 超参数配置
    #         "learning_rate": 1e-5,
    #         "batch_size": 48,
    #         "epochs": 10,
    #         "model": "sun-base",
    #     }
    # )

    # # Load configuration
    # with open(args.config, 'r') as f:
    #     config = json.load(f)
    #     GPT_CONFIG_124M.update(config)

    # text_filenames = sorted(glob.glob(os.path.join(LLM_CONFIG['data_dir'], "*pretrain.txt")))
    text_filenames = sorted(glob.glob(os.path.join("../"+LLM_CONFIG['data_dir'], "*.md")))
    tokenizer = Tokenizer("../tokenizer/"+LLM_CONFIG['tokenizer_path'])
    # print(text_filenames)pret
    text_filenames = text_filenames
    txt_dataloader = create_dataloader_from_txt_file(
        tokenizer,
        text_filenames, 48, LLM_CONFIG["context_length"], 1,
        shuffle=True, drop_last=True, num_workers=0
    )

    total_data = len(txt_dataloader.dataset)
    train_data_num = int(total_data * 0.9)
    train_data, val_data = random_split(txt_dataloader.dataset, [train_data_num, total_data - train_data_num])
    train_loader = DataLoader(train_data, batch_size=txt_dataloader.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=txt_dataloader.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LLM(LLM_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    train_epoch(
        model, train_loader, val_loader, optimizer, device, 2, "国民党反动派", tokenizer
    )

    # 保存模型
    save_path = "../model/sun_base.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_pretrain()
