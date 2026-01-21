import json
import torch
from torch.utils.data import Dataset, DataLoader

# SFT专用的数据集类，处理用户-助手对话格式
class SFTDataset(Dataset):
    def __init__(self, tokenizer, questions, answers, max_length):
        self.input_ids = []
        self.output_ids = []

        for question, answer in zip(questions, answers):
            # todo: 处理一下question和answer超过max_length的情况

            input_chunk = tokenizer.text_to_token_ids(question)
            output_chunk = tokenizer.text_to_token_ids(answer) # todo: 加上eos_token

            for i in range(len(output_chunk)):
                self.input_ids.append(torch.tensor(input_chunk + output_chunk[:i])) # 输入是前i个token
                self.output_ids.append(torch.tensor(output_chunk[i])) # 目标是第i+1个token

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        target_ids = self.output_ids[idx]

        return input_ids, target_ids


def create_sft_dataloader(tokenizer, questions, answers, batch_size, max_length,
                         shuffle=True, drop_last=True, num_workers=0):
    """
    为SFT任务创建数据加载器
    """
    dataset = SFTDataset(tokenizer, questions, answers, max_length)

    print(f"创建了 {len(dataset)} 个SFT样本数据")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    print(f"创建了 {len(dataloader)} 个批次数据")
    return dataloader


def create_sft_dataloader_from_file(tokenizer, sft_file_paths, batch_size, max_length,
                         shuffle=True, drop_last=True, num_workers=0):
    """
    从文本文件创建SFT数据加载器
    """
    questions = []
    answers = []

    for sft_file_path in sft_file_paths:
        # 读取json
        with open(sft_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                questions.append(item['question'])
                answers.append(item['answer'])

    dataloader = create_sft_dataloader(tokenizer, questions, answers, batch_size, max_length,
                         shuffle, drop_last, num_workers)

    return dataloader
