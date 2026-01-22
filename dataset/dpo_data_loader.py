import json
import torch
from torch.utils.data import Dataset, DataLoader

class DPODataset(Dataset):
    def __init__(self, tokenizer, prompts, chosens, rejects, max_length):
        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.max_length = max_length
        self.tokenizer = tokenizer

        for prompt, chosen, rejected in zip(prompts, chosens, rejects):
            # 编码
            prompt_ids = tokenizer.text_to_token_ids(prompt)
            chosen_ids = tokenizer.text_to_token_ids(chosen)
            rejected_ids = tokenizer.text_to_token_ids(rejected)
            
            # 截断
            prompt_ids = self.truncate_to_max_length(prompt_ids, self.max_length)
            chosen_ids = self.truncate_to_max_length(chosen_ids, self.max_length)
            rejected_ids = self.truncate_to_max_length(rejected_ids, self.max_length)
            
            self.prompts.append(torch.tensor(prompt_ids))
            self.chosens.append(torch.tensor(chosen_ids))
            self.rejects.append(torch.tensor(rejected_ids))
    
    def truncate_to_max_length(self, ids, max_len):
        """截断到最大长度"""
        return ids[:max_len] if len(ids) > max_len else ids
    
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
