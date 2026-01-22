import json
import torch
from torch.utils.data import Dataset, DataLoader

# DPO专用的数据集类，处理用户-助手对话格式
class DPODataset(Dataset):
    def __init__(self, tokenizer, prompts, chosens, rejects, max_length):
        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.max_length = max_length

        for prompt, chosen, rejected in zip(prompts, chosens, rejects):
            # todo: 处理一下question和answer超过max_length的情况

            # 处理超过max_length的情况，进行截断
            prompt_chunk = tokenizer.text_to_token_ids(prompt)
            chosen_chunk = tokenizer.text_to_token_ids(chosen, add_eos=True)
            rejected_chunk = tokenizer.text_to_token_ids(rejected, add_eos=True)
            
            self.prompts.append(torch.tensor(prompt_chunk))
            self.chosens.append(torch.tensor(chosen_chunk))
            self.rejects.append(torch.tensor(rejected_chunk))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.chosens[idx], self.rejects[idx]


def create_dpo_dataloader(tokenizer, prompts, chosens, rejects, batch_size, max_length,
                         shuffle=True, drop_last=True, num_workers=0):
    """
    为DPO任务创建数据加载器
    """
    dataset = DPODataset(tokenizer, prompts, chosens, rejects, max_length)

    print(f"创建了 {len(dataset)} 个DPO样本数据")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    print(f"创建了 {len(dataloader)} 个批次数据")
    return dataloader


def create_dpo_dataloader_from_file(tokenizer, dpo_file_paths, batch_size, max_length,
                         shuffle=True, drop_last=True, num_workers=0):
    """
    从文本文件创建DPO数据加载器
    """
    prompts = []
    chosens = []
    rejects = []

    for dpo_file_path in dpo_file_paths:
        with open(dpo_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                prompts.append(item['prompt'])
                chosens.append(item['chosen'])
                rejects.append(item['rejected'])

    dataloader = create_dpo_dataloader(tokenizer, prompts, chosens, rejects, batch_size, max_length,
                         shuffle, drop_last, num_workers)

    return dataloader
