"""
=============================================================================
数据加载模块 - 预训练 (data_loader.py)
=============================================================================

【模块作用】
这个文件实现了预训练阶段的数据加载器，用于：
1. 将文本转换为token序列
2. 使用滑动窗口构建训练样本
3. 批量加载数据供模型训练

【相关知识】
1. Dataset类：PyTorch的数据集抽象，需要实现__len__和__getitem__
2. DataLoader：批量加载数据，支持shuffle、多进程等
3. 滑动窗口：将长文本切分成多个固定长度的训练样本
4. 语言模型训练：预测序列中的下一个token
"""

# ============================================================================
# 导入必要的库
# ============================================================================
import os  # 操作系统接口
import sys  # 系统相关功能
import torch  # PyTorch框架
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置和分词器
from config.config import LLM_CONFIG
from tokenizer.tokenizer import Tokenizer


# ============================================================================
# DatasetV1类：预训练数据集
# ============================================================================
class DatasetV1(Dataset):
    """
    预训练数据集类

    相关知识：
    - 继承torch.utils.data.Dataset类
    - 必须实现__len__和__getitem__方法
    - 使用滑动窗口将长文本切分为固定长度的训练样本
    """

    def __init__(self, tokenizer, txts, max_length, stride):
        """
        初始化数据集

        参数:
            tokenizer: 分词器对象
            txts: 文本列表
            max_length: 每个样本的最大长度
            stride: 滑动窗口的步长
        """
        self.input_ids = []  # 存储输入token序列
        self.target_ids = []  # 存储目标token序列

        # 遍历每段文本
        for txt in txts:
            # 将文本编码为token ID列表
            token_ids = tokenizer.text_to_token_ids(txt)
            print(len(token_ids))

            # 使用滑动窗口构建样本
            # range(0, len - max_length, stride) 生成起始位置索引
            for i in range(0, len(token_ids) - max_length, stride):
                # 输入：从i到i+max_length的token
                input_chunk = token_ids[i:i + max_length]
                # 目标：从i+1到i+max_length+1的token（下一个token预测）
                target_chunk = token_ids[i + 1: i + max_length + 1]

                # 转换为tensor并存储
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集大小"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        return self.input_ids[idx], self.target_ids[idx]


# ============================================================================
# 数据加载器创建函数
# ============================================================================
def create_dataloader_v1(tokenizer, txts, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    """
    创建数据加载器

    参数:
        tokenizer: 分词器
        txts: 文本列表
        batch_size: 批次大小
        max_length: 最大序列长度
        stride: 滑动窗口步长
        shuffle: 是否打乱数据
        drop_last: 是否丢弃最后不完整的批次
        num_workers: 数据加载的进程数

    返回:
        DataLoader对象
    """
    # 创建数据集
    dataset = DatasetV1(tokenizer, txts, max_length, stride)
    print(f"创建了 {len(dataset)} 个样本数据")

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    print(f"创建了 {len(dataloader)} 个批次数据")
    return dataloader


def create_dataloader_from_txt_file(tokenizer, file_paths, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    """从文本文件创建数据加载器"""
    raw_texts = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            raw_texts.append(raw_text)

    dataloader = create_dataloader_v1(tokenizer, raw_texts, batch_size, max_length, stride,
                         shuffle, drop_last, num_workers)
    return dataloader


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    raw_text = """各位来宾、老师、同学们：

早上好。

十年前，当我们第一次看到AI画出一幅水墨山水，有人惊叹："这不过是像素的堆砌。"
五年前，当AI写出了第一篇高考作文，有人嘲笑："它不懂情感，只是在模仿。"
三年前，当AI在医疗影像中发现医生漏诊的早期肿瘤，有人沉默了——因为那不是概率，那是生命。

今天，AI 已经不再是一个"工具"——它是一面镜子，映照出我们对智慧、创造力与人性的全部想象与恐惧。

它能写诗，但不懂离别；
它能作曲，但不知思念；
它能诊断疾病，却无法握住病人的手说一句"别怕"。

我们正站在一个前所未有的转折点上：人类历史上第一次，我们正在创造一种比我们更擅长"思考"的存在，却无法教会它"感受"。

这不是科幻小说的桥段，这是正在发生的现实。
你刷到的每一条短视频推荐，背后是数亿次行为的建模；
你孩子作业本上的批注，可能来自一个从未上过学的AI系统；
你深夜加班时收到的自动回复，也许出自一个没有睡眠、没有疲惫、没有情绪的"数字员工"。

我们惊叹于它的效率，却开始焦虑它的边界。
我们依赖它的判断，却不敢信任它的选择。
我们用它来提升生产力，却害怕它剥夺我们的意义。

但我想问：
当机器比我们更懂语言，我们是否还懂得如何真诚地说话？
当算法比我们更会选答案，我们是否还敢提出那个"错误却勇敢"的问题？
当AI能替我们写信、作画、做决策——我们，还愿意亲自去爱、去错、去挣扎吗？

人工智能的终极挑战，从来不是"它能不能"，而是"它该不该"。
不是"它有多聪明"，而是"我们，是否还配得上这份聪明"。

我们不需要一个更强大的AI。
我们需要一个更清醒的人类。

一个知道何时该放手、何时该守护的人类；
一个懂得技术是翅膀、而非枷锁的人类；
一个在算法洪流中，依然坚持为孤独者留一盏灯、为沉默者留一句话、为不确定留一点空间的人类。

今天的演讲，不是关于Transformer、不是关于参数量、也不是关于算力竞赛。
它是关于——
当世界越来越被机器理解，我们，该如何重新学会理解彼此？

谢谢大家。
现在，请允许我，和你们一起，走进这场关于"人"的重新发现。"""

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
