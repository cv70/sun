import glob
import os
import sys

import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, random_split
from model.llm import LLM
from config.config import LLM_CONFIG
from utils.loss import calc_loss_batch
from utils.generate import generate_and_print_sample
from utils.eval import evaluate_model
from dataset.data_loader import create_dataloader_from_txt_file
from tokenizer.tokenizer import Tokenizer

def train_pretrain_epoch(model, train_loader, val_loader, optimizer, device, num_epochs, start_context, tokenizer, eval_freq=1000, eval_iter=5):
    train_losses, val_losses = [], []
    steps = 0
    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 梯度清零
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 反向传播，计算损失梯度
            optimizer.step() # 更新参数
            steps += 1

            if steps % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_losses)
                print(f"Epoch {epoch + 1} Step {steps}: Train loss {train_loss}, Val loss {val_loss}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
        print(f"Epoch {epoch + 1}/{num_epochs}, Cost Time: {time.time() - start_time:.2f}s")


def train_pretrain():
    # # Load configuration
    # with open(args.config, 'r') as f:
    #     config = json.load(f)
    #     GPT_CONFIG_124M.update(config)

    text_filenames = sorted(glob.glob(os.path.join(LLM_CONFIG['data_dir'], "*pretrain.txt")))
    tokenizer = Tokenizer(LLM_CONFIG['tokenizer_path'])
    # print(text_filenames)
    text_filenames = text_filenames
    txt_dataloader = create_dataloader_from_txt_file(
        tokenizer,
        text_filenames, 128, LLM_CONFIG["context_length"], 1,
        shuffle=True, drop_last=True, num_workers=0
    )

    total_data = len(txt_dataloader.dataset)
    train_data_num = int(total_data * 0.9)
    train_data, val_data = random_split(txt_dataloader.dataset, [train_data_num, total_data - train_data_num])
    train_loader = DataLoader(train_data, batch_size=txt_dataloader.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=txt_dataloader.batch_size, shuffle=False)

    # Create model and move it to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LLM(LLM_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    train_pretrain_epoch(
        model, train_loader, val_loader, optimizer, device, 2, "郭靖挥出一拳", tokenizer
    )

    # 保存模型
    save_path = "model/sun_base.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_pretrain()
