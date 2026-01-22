"""
=============================================================================
分词器模块 (tokenizer.py)
=============================================================================

【模块作用】
这个模块实现了文本与数字ID之间的双向转换功能。
分词器是连接人类语言（文本）和机器语言（数字）的桥梁。

【相关知识】
1. Tokenization（分词）: 将文本切分成模型能处理的最小单位（token）
2. SentencePiece: Google开源的通用的分词工具，支持多种语言
3. BPE (Byte Pair Encoding): 一种子词分词算法，能处理未知词
4. EOS token (End of Sequence): 序列结束标记，告诉模型停止生成
5. 交叉熵损失 (Cross Entropy): 语言模型训练常用的损失函数
6. Softmax: 将模型输出转换为概率分布的函数
7. Temperature sampling: 控制生成随机性的采样技术
"""

# ============================================================================
# 导入必要的库和模块
# ============================================================================
import os  # 操作系统接口模块，用于文件路径操作
import sys  # 系统相关的参数和函数，用于修改Python路径
import sentencepiece as spm  # SentencePiece库，用于分词
import torch  # PyTorch深度学习框架

# 将项目根目录添加到Python搜索路径中
# 这样才能导入config和model模块
# os.path.dirname: 获取父目录
# os.path.abspath: 获取绝对路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目配置
from config.config import LLM_CONFIG

# 导入LLM模型类
from model.llm import LLM

# 导入生成函数（虽然在本文件中定义了generate_text_simple，但这里导入了utils中的版本）
from utils.generate import generate


# ============================================================================
# Tokenizer类：文本与ID的转换器
# ============================================================================
class Tokenizer():
    """
    分词器类，用于文本和token ID之间的相互转换

    相关知识：
    - 类 (Class): 面向对象编程的基本概念，封装数据和方法
    - __init__: 构造函数，创建对象时自动调用
    - self: 表示对象自身，用于访问对象的属性和方法
    """

    def __init__(self, spm_model_path):
        """
        初始化分词器

        参数:
            spm_model_path: SentencePiece模型文件的路径

        相关知识：
        - SentencePieceProcessor: SentencePiece的分词处理器类
        - .load(): 加载预训练的模型文件
        - eos_id: end-of-sequence token的ID，表示序列结束
        """
        # 创建SentencePiece处理器对象
        # spm.SentencePieceProcessor() 创建一个未加载模型的对象
        self.sp_bpe = spm.SentencePieceProcessor()

        # 加载预训练的BPE模型文件
        # self.sp_bpe.load() 从文件中加载分词模型
        self.sp_bpe.load(spm_model_path)

        # 获取并保存EOS token的ID
        # eos_id()方法返回序列结束标记的ID
        # 添加eos_id属性，方便后续使用
        self.eos_id = self.sp_bpe.eos_id()

    def text_to_token_ids(self, text, add_eos=False):
        """
        将文本转换为token ID列表

        参数:
            text: 输入的文本字符串
            add_eos: 是否在末尾添加EOS token（默认False）

        返回:
            token ID的列表，例如: [123, 456, 789]

        相关知识：
        - EncodeAsIds: SentencePiece的方法，将文本编码为ID序列
        - add_eos参数控制是否添加结束标记，生成时常用
        """
        # 调用SentencePiece的编码方法
        # EncodeAsIds将文本字符串转换为整数ID列表
        encoded = self.sp_bpe.EncodeAsIds(text, add_eos=add_eos)

        # 返回编码后的ID列表
        return encoded

    def token_ids_to_text(self, token_ids):
        """
        将token ID列表转换为文本

        参数:
            token_ids: token ID的列表，例如: [123, 456, 789]

        返回:
            解码后的文本字符串

        相关知识：
        - DecodeIds: SentencePiece的方法，将ID序列解码为文本
        - 这是text_to_token_ids的逆操作
        """
        # 调用SentencePiece的解码方法
        # DecodeIds将整数ID列表转换为文本字符串
        decoded = self.sp_bpe.DecodeIds(token_ids)

        # 返回解码后的文本
        return decoded

    def vocab_size(self):
        """
        获取词表大小

        返回:
            词表中token的总数

        相关知识：
        - vocab_size: 词表大小，模型输出层的维度
        - 这个值应该与配置中的vocab_size一致
        """
        # 返回词表中的token数量
        return self.sp_bpe.vocab_size()


# ============================================================================
# 简单的文本生成函数
# ============================================================================
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    简单的文本生成函数，使用贪婪解码策略

    参数:
        model: 训练好的语言模型
        idx: 当前上下文的token索引，形状为(batch_size, n_tokens)
        max_new_tokens: 要生成的新token数量
        context_size: 模型支持的最大上下文长度

    返回:
        生成的完整token序列，包含输入和新生成的部分

    相关知识：
    - 贪婪解码 (Greedy Decoding): 每次选择概率最高的token
    - 自回归生成 (Autoregressive): 基于已生成的序列继续生成
    - 上下文窗口 (Context Window): 模型能处理的最大序列长度
    """
    # idx是当前上下文中的索引数组，形状为(batch_size, n_tokens)
    # 例如: [[1, 2, 3]] 表示一个包含3个token的序列

    # 循环生成max_new_tokens个新token
    for _ in range(max_new_tokens):
        # ====================================================================
        # 步骤1: 裁剪上下文到模型支持的最大长度
        # ====================================================================
        # 如果当前上下文超出模型支持的上下文大小，则裁剪
        # 只保留最后context_size个token
        # 例如，如果模型只支持256个token，而序列有300个
        # 那么只使用最后256个token作为上下文
        idx_cond = idx[:, -context_size:]  # 形状: (batch_size, context_size)

        # ====================================================================
        # 步骤2: 模型推理，获取logits
        # ====================================================================
        # 使用torch.no_grad()关闭梯度计算
        # 推理时不需要计算梯度，可以节省内存
        with torch.no_grad():
            # 将裁剪后的序列输入模型，获取输出logits
            # logits形状: (batch_size, seq_len, vocab_size)
            # vocab_size是词表大小，每个位置对应一个token的预测分数
            logits = model(idx_cond)

        # ====================================================================
        # 步骤3: 只关注最后一个时间步的预测
        # ====================================================================
        # 我们只需要预测下一个token，所以只取最后一个位置的输出
        # logits[:, -1, :] 提取最后一个时间步的所有词的logits
        # 形状从 (batch_size, seq_len, vocab_size) 变为 (batch_size, vocab_size)
        logits = logits[:, -1, :]

        # ====================================================================
        # 步骤4: 将logits转换为概率分布
        # ====================================================================
        # Softmax将任意实数转换为[0, 1]区间的概率值
        # dim=-1表示在最后一个维度（vocab_size维度）上进行softmax
        # 结果: 每个token对应一个概率值，所有概率之和为1
        probas = torch.softmax(logits, dim=-1)  # 形状: (batch_size, vocab_size)

        # ====================================================================
        # 步骤5: 选择概率最高的token（贪婪解码）
        # ====================================================================
        # argmax返回最大值的索引
        # dim=-1表示在vocab_size维度上找最大值
        # keepdim=True保持维度不变，方便后续拼接
        # 形状: (batch_size, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # ====================================================================
        # 步骤6: 将新token拼接到序列末尾
        # ====================================================================
        # torch.cat在指定维度上拼接张量
        # dim=1表示在序列长度维度上拼接
        # 形状从 (batch_size, n_tokens) 变为 (batch_size, n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=1)

    # 返回生成的完整序列（包含输入和新生成的部分）
    return idx


# ============================================================================
# 模块初始化代码
# ============================================================================
import os  # 重新导入os模块（虽然前面已经导入过）
import sys  # 重新导入sys模块
# 将父目录添加到Python搜索路径
# sys.path.insert(0, path) 将路径插入到搜索路径的最前面
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# 主程序：测试分词器和生成功能
# ============================================================================
if __name__ == "__main__":
    """
    主程序入口
    当直接运行这个文件时，会执行以下测试代码

    相关知识：
    - __name__: Python内置变量，当文件被直接运行时为"__main__"
    - if __name__ == "__main__": 这是一个常见的Python模式
    - 这样可以避免在被导入时执行测试代码
    """
    # 导入配置和模型
    from config.config import LLM_CONFIG  # 导入模型配置
    from model.llm import LLM  # 导入LLM模型类

    # ========================================================================
    # 步骤1: 设置输入文本（提示词）
    # ========================================================================
    # 这是模型的起始文本，模型将基于这个文本继续生成
    start_context = "国民党出兵攻打"

    # ========================================================================
    # 步骤2: 创建分词器对象
    # ========================================================================
    # 从配置中获取分词器模型路径
    # Tokenizer类会加载SentencePiece模型文件
    tokenizer = Tokenizer(LLM_CONFIG['tokenizer_path'])

    # ========================================================================
    # 步骤3: 创建模型对象
    # ========================================================================
    # 使用配置字典初始化LLM模型
    model = LLM(LLM_CONFIG)

    # 设置模型为评估模式
    # model.eval()会关闭dropout和batch normalization等训练专用行为
    model.eval()

    # ========================================================================
    # 步骤4: 设置随机种子
    # ========================================================================
    # 设置随机种子使得结果可复现
    # 每次运行使用相同的种子会得到相同的输出
    torch.manual_seed(123)

    # ========================================================================
    # 步骤5: 将文本转换为token ID
    # ========================================================================
    # 使用分词器将起始文本转换为ID列表
    token_ids_list = tokenizer.text_to_token_ids(start_context)

    # ========================================================================
    # 步骤6: 将ID列表转换为PyTorch张量
    # ========================================================================
    # 创建一个二维张量，形状为(1, seq_len)
    # dtype=torch.long指定数据类型为64位整数
    idx = torch.tensor([token_ids_list], dtype=torch.long)

    # ========================================================================
    # 步骤7: 生成文本
    # ========================================================================
    # 调用生成函数，生成15个新token
    generation_output = generate_text_simple(
        model=model,  # 训练好的模型
        idx=idx,  # 输入序列
        max_new_tokens=15,  # 生成15个新token
        context_size=LLM_CONFIG["context_length"]  # 上下文长度
    )

    # ========================================================================
    # 步骤8: 将生成的token ID转换回文本
    # ========================================================================
    # .tolist()将张量转换为Python列表
    # token_ids_to_text将ID列表解码为文本
    decoded_text = tokenizer.token_ids_to_text(generation_output[0].tolist())

    # ========================================================================
    # 步骤9: 打印结果
    # ========================================================================
    # 打印输入文本
    print("输入文本:", start_context)

    # 打印模型生成的完整文本（包含输入）
    print("输出文本:", decoded_text)

    # 打印生成的总token数（输入+输出）
    print(f"输出token数: {generation_output.shape[1]}")
