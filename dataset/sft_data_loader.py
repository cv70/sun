"""
=============================================================================
数据加载模块 - 监督微调 (sft_data_loader.py)
=============================================================================

【模块作用】
这个文件实现了监督微调(SFT)阶段的数据加载器：
1. 处理问答对数据
2. 使用-100忽略prompt部分的损失
3. 在答案末尾添加EOS token

【相关知识】
1. SFT (Supervised Fine-tuning): 使用标注数据微调预训练模型
2. 忽略索引(-100): PyTorch中交叉熵损失的默认忽略值
3. EOS token: 序列结束标记
4. padding: 将不同长度的序列填充到相同长度
"""

# ============================================================================
# 导入必要的库
# ============================================================================
import json  # JSON文件处理
import torch  # PyTorch框架
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器


# ============================================================================
# SFTDataset类：监督微调数据集
# ============================================================================
class SFTDataset(Dataset):
    """
    监督微调数据集类

    相关知识：
    - SFT使用问答对数据进行训练
    - 只对答案部分计算损失，使用-100忽略prompt部分
    """

    def __init__(self, tokenizer, questions, answers, max_length):
        """
        初始化SFT数据集

        参数:
            tokenizer: 分词器对象
            questions: 问题列表
            answers: 答案列表
            max_length: 最大序列长度
        """
        self.input_ids = []  # 输入token序列
        self.labels = []  # 标签序列（prompt部分为-100）

        # ========================================================================
        # 获取EOS token ID
        # ========================================================================
        if hasattr(tokenizer, "eos_id"):
            eos_id = tokenizer.eos_id
        else:
            try:
                eos_id = tokenizer.sp_bpe.eos_id()
            except:
                eos_id = 1  # 默认EOS token ID

        # 遍历所有问答对
        for question, answer in zip(questions, answers):
            # 将问题和答案编码为token ID
            question_ids = tokenizer.text_to_token_ids(question)
            answer_ids = tokenizer.text_to_token_ids(answer)

            # 在答案末尾添加EOS token
            answer_ids = answer_ids + [eos_id]

            # 拼接问题和答案
            full_input = question_ids + answer_ids

            # 如果总长度超过max_length，进行截断
            if len(full_input) > max_length:
                # 截断到max_length-1，保留位置给EOS
                full_input = full_input[:max_length-1] + [eos_id]

            # 如果问题本身就很长，只保留问题+EOS
            if len(question_ids) >= max_length:
                question_ids = question_ids[:max_length - 1]
                full_input = question_ids + [eos_id]

            # ====================================================================
            # 创建标签：问题部分为-100，答案部分为对应的token ID
            # ====================================================================
            # -100是PyTorch交叉熵损失的默认忽略值
            # 这样模型只学习生成答案，不学习prompt
            labels = [-100] * len(question_ids) + answer_ids

            if len(labels) > max_length:
                labels = labels[:max_length]

            self.input_ids.append(torch.tensor(full_input, dtype=torch.long))
            self.labels.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        """返回数据集大小"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        return self.input_ids[idx], self.labels[idx]


# ============================================================================
# SFT数据加载器创建函数
# ============================================================================
def create_sft_dataloader(tokenizer, questions, answers, batch_size, max_length,
                       shuffle=True, drop_last=True, num_workers=0):
    """
    为SFT任务创建数据加载器

    参数:
        tokenizer: 分词器
        questions: 问题列表
        answers: 答案列表
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱数据
        drop_last: 是否丢弃最后不完整的批次
        num_workers: 数据加载的进程数

    返回:
        DataLoader对象
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

    参数:
        tokenizer: 分词器
        filenames: JSON文件路径列表
        batch_size: 批次大小
        max_length: 最大序列长度

    返回:
        DataLoader对象
    """
    all_questions = []
    all_answers = []

    # 从JSON文件读取数据
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                all_questions.append(item['question'])
                all_answers.append(item['answer'])

    return create_sft_dataloader(tokenizer, all_questions, all_answers,
                              batch_size, max_length, shuffle, drop_last, num_workers)
