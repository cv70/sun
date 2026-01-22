import json
import torch
from torch.utils.data import Dataset, DataLoader

class SFTDataset(Dataset):
    def __init__(self, tokenizer, questions, answers, max_length):
        self.input_ids = []
        self.labels = []

        if hasattr(tokenizer, "eos_id"):
            eos_id = tokenizer.eos_id
        else:
            try:
                eos_id = tokenizer.sp_bce.eos_id()
            except:
                eos_id = 1

        for question, answer in zip(questions, answers):
            question_ids = tokenizer.text_to_token_ids(question)
            answer_ids = tokenizer.text_to_token_ids(answer)

            # 添加 EOS
            answer_ids = answer_ids + [eos_id]

            full_input = question_ids + answer_ids

            if len(full_input) > max_length:
                full_input = full_input[:max_length-1] + [eos_id]
            if len(question_ids) >= max_length:
                question_ids = question_ids[:max_length - 1]
                full_input = question_ids + [eos_id]

            labels = [-100] * len(question_ids) + answer_ids
            if len(labels) > max_length:
                labels = labels[:max_length]

            self.input_ids.append(torch.tensor(full_input, dtype=torch.long))
            self.labels.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


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


def create_sft_dataloader_from_file(tokenizer, filenames, batch_size, max_length,
                                 shuffle=True, drop_last=True, num_workers=0):
    """
    从JSON文件创建SFT数据加载器
    """
    all_questions = []
    all_answers = []

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                all_questions.append(item['question'])
                all_answers.append(item['answer'])

    return create_sft_dataloader(tokenizer, all_questions, all_answers,
                              batch_size, max_length, shuffle, drop_last, num_workers)