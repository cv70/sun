import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import LLM_CONFIG
from tokenizer.tokenizer import Tokenizer

# DatasetV1 将文本转换为输入和输出的 token_ids 序列
class DatasetV1(Dataset):
    def __init__(self, tokenizer, txts, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        for txt in txts:
            token_ids = tokenizer.text_to_token_ids(txt)
            print(len(token_ids))

            # 滑动窗口构建样本数据
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1: i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(tokenizer, txts, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):

    dataset = DatasetV1(tokenizer, txts, max_length, stride)

    print(f"创建了 {len(dataset)} 个样本数据")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    print(f"创建了 {len(dataloader)} 个批次数据")
    return dataloader

def create_dataloader_from_txt_file(tokenizer, file_paths, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    raw_texts = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            raw_texts.append(raw_text)

    dataloader = create_dataloader_v1(tokenizer, raw_texts, batch_size, max_length, stride,
                         shuffle, drop_last, num_workers)

    return dataloader

if __name__ == "__main__":
    # 写一段文本
    raw_text = """各位来宾、老师、同学们：

早上好。

十年前，当我们第一次看到AI画出一幅水墨山水，有人惊叹：“这不过是像素的堆砌。”
五年前，当AI写出了第一篇高考作文，有人嘲笑：“它不懂情感，只是在模仿。”
三年前，当AI在医疗影像中发现医生漏诊的早期肿瘤，有人沉默了——因为那不是概率，那是生命。

今天，AI 已经不再是一个“工具”——它是一面镜子，映照出我们对智慧、创造力与人性的全部想象与恐惧。

它能写诗，但不懂离别；
它能作曲，但不知思念；
它能诊断疾病，却无法握住病人的手说一句“别怕”。

我们正站在一个前所未有的转折点上：人类历史上第一次，我们正在创造一种比我们更擅长“思考”的存在，却无法教会它“感受”。

这不是科幻小说的桥段，这是正在发生的现实。
你刷到的每一条短视频推荐，背后是数亿次行为的建模；
你孩子作业本上的批注，可能来自一个从未上过学的AI系统；
你深夜加班时收到的自动回复，也许出自一个没有睡眠、没有疲惫、没有情绪的“数字员工”。

我们惊叹于它的效率，却开始焦虑它的边界。
我们依赖它的判断，却不敢信任它的选择。
我们用它来提升生产力，却害怕它剥夺我们的意义。

但我想问：
当机器比我们更懂语言，我们是否还懂得如何真诚地说话？
当算法比我们更会选答案，我们是否还敢提出那个“错误却勇敢”的问题？
当AI能替我们写信、作画、做决策——我们，还愿意亲自去爱、去错、去挣扎吗？

人工智能的终极挑战，从来不是“它能不能”，而是“它该不该”。
不是“它有多聪明”，而是“我们，是否还配得上这份聪明”。

我们不需要一个更强大的AI。
我们需要一个更清醒的人类。

一个知道何时该放手、何时该守护的人类；
一个懂得技术是翅膀、而非枷锁的人类；
一个在算法洪流中，依然坚持为孤独者留一盏灯、为沉默者留一句话、为不确定留一点空间的人类。

今天的演讲，不是关于Transformer、不是关于参数量、也不是关于算力竞赛。
它是关于——
当世界越来越被机器理解，我们，该如何重新学会理解彼此？

谢谢大家。
现在，请允许我，和你们一起，走进这场关于“人”的重新发现。"""

    tokenizer = Tokenizer(LLM_CONFIG['tokenizer_path'])
    vocab_size = LLM_CONFIG['vocab_size']
    emb_dim = LLM_CONFIG['emb_dim']
    context_length = LLM_CONFIG['context_length']


    token_embedding_layer = torch.nn.Embedding(vocab_size, emb_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, emb_dim)

    batch_size = 8
    dataloader = create_dataloader_v1(
        tokenizer,
        [raw_text],
        batch_size=batch_size,
        max_length=context_length,
        stride=1
    )

    for batch in dataloader:
        x, y = batch

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(context_length))

        input_embeddings = token_embeddings + pos_embeddings

        break

    print(input_embeddings.shape)
