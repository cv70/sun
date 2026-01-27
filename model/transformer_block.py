"""
=============================================================================
Transformer块模块 (transformer_block.py)
=============================================================================

【模块作用】
这个文件实现了Transformer架构的基本构建块：TransformerBlock。
一个Transformer块由多头注意力和前馈神经网络组成，通过残差连接和层归一化连接。

【相关知识】
1. Transformer块: Transformer的基本单元，包含注意力和FFN两个子层
2. 残差连接 (Residual Connection): 将输入加到输出上，防止梯度消失
3. Pre-LN vs Post-LN: 归一化在子层之前或之后的两种架构选择
4. 深层网络: 多个Transformer块堆叠形成深度网络
5. GPT架构: 使用带因果掩码的自注意力Transformer
"""

# ============================================================================
# 导入必要的库
# ============================================================================
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块

# 导入自定义模块
from model.attention import MultiHeadAttention  # 多头注意力机制
from model.feed_forward import FeedForward, LayerNorm  # 前馈网络和层归一化


# ============================================================================
# TransformerBlock类：Transformer的基本块
# ============================================================================
class TransformerBlock(nn.Module):
    """
    Transformer块：Transformer架构的基本构建单元

    相关知识：
    - Transformer块由两个子层组成：多头注意力和前馈神经网络
    - 每个子层都使用残差连接和层归一化
    - 这种结构称为"Pre-LN"架构（归一化在子层之前）
    - GPT-2、GPT-3等都使用类似的架构
    - 典型的Transformer模型包含多个堆叠的Transformer块
    """

    def __init__(self, cfg):
        """
        初始化Transformer块

        参数:
            cfg: 配置字典，包含模型的所有超参数

        相关知识：
        - Transformer块需要知道嵌入维度、头数、上下文长度等配置
        - 这些配置被传递给注意力和前馈网络子层
        """
        # 调用父类构造函数
        super().__init__()

        # ========================================================================
        # 创建多头注意力子层
        # ========================================================================
        # MultiHeadAttention实现自注意力机制
        # 允许每个token关注序列中的所有其他token
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],  # 输入维度（嵌入维度）
            d_out=cfg["emb_dim"],  # 输出维度（与输入相同，方便残差连接）
            context_length=cfg["context_length"],  # 最大序列长度
            num_heads=cfg["n_heads"],  # 注意力头数
            dropout=cfg["drop_rate"],  # Dropout率
            qkv_bias=cfg["qkv_bias"]  # 是否在QKV变换中使用偏置
        )

        # ========================================================================
        # 创建前馈神经网络子层
        # ========================================================================
        # FeedForward对每个token独立应用非线性变换
        self.ff = FeedForward(cfg)

        # ========================================================================
        # 创建层归一化层
        # ========================================================================
        # norm1: 用于注意力子层之前的归一化
        # norm2: 用于前馈网络子层之前的归一化
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # ========================================================================
        # 创建Dropout层（用于残差连接）
        # ========================================================================
        # drop_shortcut: 应用在残差连接上的dropout
        # 这是另一种正则化手段
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        前向传播：执行Transformer块的计算

        参数:
            x: 输入张量，形状为(batch_size, num_tokens, emb_dim)
               batch_size: 批次大小
               num_tokens: 序列长度（token数量）
               emb_dim: 嵌入维度

        返回:
            输出张量，形状与输入相同 (batch_size, num_tokens, emb_dim)

        相关知识：
        - 前向传播执行两个子层的计算
        - 每个子层都遵循: Norm -> SubLayer -> Dropout -> Residual的流程
        - 这种架构称为"Pre-LN"（Layer Normalization在子层之前）
        """
        # ========================================================================
        # 子层1: 多头注意力 + 残差连接
        # ========================================================================

        # 保存输入用于残差连接
        # shortcut变量保存原始输入，稍后要加回到输出上
        # 这是残差连接的核心：output = layer(input) + input
        shortcut = x

        # 应用层归一化（Pre-LN架构：归一化在子层之前）
        # 归一化可以稳定训练，加速收敛
        x = self.norm1(x)

        # 应用多头注意力
        # 每个token可以关注序列中的所有其他token（包括自己）
        # 形状: (batch_size, num_tokens, emb_dim)
        x = self.att(x)

        # 应用dropout正则化
        # 随机将一部分神经元输出置为0
        x = self.drop_shortcut(x)

        # 添加残差连接
        # 将原始输入加到注意力输出上
        # 这样可以让梯度更容易流过深层网络
        # 这是ResNet的核心思想，被Transformer采用
        x = x + shortcut

        # ========================================================================
        # 子层2: 前馈神经网络 + 残差连接
        # ========================================================================

        # 再次保存输入用于残差连接
        # 这次是注意力子层的输出
        shortcut = x

        # 应用层归一化
        # 同样是Pre-LN架构
        x = self.norm2(x)

        # 应用前馈神经网络
        # FFN对每个token独立应用非线性变换
        # 扩展-压缩结构：Linear -> GELU -> Linear
        x = self.ff(x)

        # 应用dropout正则化
        x = self.drop_shortcut(x)

        # 添加残差连接
        # 将FFN的输入加到输出上
        x = x + shortcut

        # 返回Transformer块的输出
        # 输出形状与输入相同: (batch_size, num_tokens, emb_dim)
        return x
