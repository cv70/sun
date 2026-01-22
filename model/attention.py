"""
=============================================================================
多头注意力模块 (attention.py)
=============================================================================

【模块作用】
这个文件实现了Transformer架构中的核心组件：多头注意力机制（Multi-Head Attention）。
注意力机制允许模型在处理序列时，动态地关注不同位置的信息，
是现代大语言模型能够理解和生成复杂语言的关键。

【相关知识】
1. 自注意力 (Self-Attention): 序列中每个元素与其他所有元素计算相关性
2. 多头注意力 (Multi-Head Attention): 并行计算多组注意力，捕获不同的特征
3. Query、Key、Value: 注意力机制的三个核心概念，源自信息检索系统
   - Query: 查询向量，表示当前要关注的内容
   - Key: 键向量，用于与Query匹配
   - Value: 值向量，包含实际的上下文信息
4. 因果掩码 (Causal Mask): 防止模型看到未来的信息（自回归生成必需）
5. 缩放点积注意力 (Scaled Dot-Product Attention): 标准的注意力计算方式
6. 残差连接 (Residual Connection): 将输入直接加到输出上，防止梯度消失
"""

# ============================================================================
# 导入必要的库
# ============================================================================
import torch  # PyTorch深度学习框架
from torch import nn  # PyTorch神经网络模块，包含各种层结构


# ============================================================================
# MultiHeadAttention类：多头注意力机制
# ============================================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制类

    相关知识：
    - nn.Module: PyTorch中所有神经网络模块的基类
    - 继承nn.Module后，可以使用PyTorch的自动微分、参数管理等功能
    - 多头注意力的核心思想：将注意力计算分成多个"头"，每个头学习不同的特征
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        初始化多头注意力层

        参数:
            d_in: 输入的维度大小（嵌入维度）
            d_out: 输出的维度大小
            context_length: 最大序列长度（用于创建因果掩码）
            dropout: Dropout概率，用于正则化
            num_heads: 注意力头的数量
            qkv_bias: 是否在Q、K、V变换中使用偏置

        相关知识：
        - 构造函数定义层的结构和参数
        - 注意力头数必须能整除输出维度
        """
        # 调用父类(nn.Module)的构造函数
        # 这是PyTorch神经网络模块的标准初始化方式
        super().__init__()

        # ========================================================================
        # 断言检查：确保输出维度能被注意力头数整除
        # ========================================================================
        # assert是Python的断言语句，如果条件为False会抛出异常
        # 这是因为每个头需要分配相等数量的维度
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        # 保存参数到实例变量，方便在forward方法中使用
        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 注意力头数

        # 计算每个头的维度
        # 例如：d_out=256, num_heads=8, 则head_dim=32
        # 每个头独立处理head_dim维度的信息
        self.head_dim = d_out // num_heads

        # 保存上下文长度，用于后续验证
        self.context_length = context_length

        # ========================================================================
        # 创建Q、K、V的线性变换层
        # ========================================================================
        # 这三个线性层将输入向量变换为Query、Key、Value向量
        # nn.Linear(in_features, out_features, bias)
        # 每个层都是可学习的参数矩阵

        # Query变换层：将输入映射到Query空间
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Key变换层：将输入映射到Key空间
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Value变换层：将输入映射到Value空间
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # ========================================================================
        # 输出投影层
        # ========================================================================
        # 将多头注意力输出投影回原始维度
        # 这是对多头结果进行融合的步骤
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout层：用于正则化，防止过拟合
        self.dropout = nn.Dropout(dropout)

        # ========================================================================
        # 创建因果掩码（Causal Mask / Attention Mask）
        # ========================================================================
        # register_buffer将张量注册为模型的缓冲区
        # 缓冲区不是可训练参数，但会随模型一起移动（GPU/CPU）和保存
        # torch.triu创建上三角矩阵，diagonal=1表示不包含对角线
        # 结果是一个只有上三角（不含对角线）为1，其余为0的矩阵
        # 例如，对于context_length=4:
        # [[0, 1, 1, 1],
        #  [0, 0, 1, 1],
        #  [0, 0, 0, 1],
        #  [0, 0, 0, 0]]
        # 这个掩码用于实现因果注意力：每个token只能关注它自己和之前的token
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        """
        前向传播：计算多头注意力

        参数:
            x: 输入张量，形状为(batch_size, num_tokens, d_in)
               batch_size: 批次大小
               num_tokens: 序列中的token数量
               d_in: 输入特征维度

        返回:
            context_vec: 上下文向量，形状为(batch_size, num_tokens, d_out)

        相关知识：
        - forward方法定义了模块的前向计算逻辑
        - 当调用模块实例时会自动执行forward
        """
        # ========================================================================
        # 步骤1: 解析输入形状
        # ========================================================================
        # 获取输入张量的三个维度
        b, num_tokens, d_in = x.shape
        # b: batch_size，批次大小（一次处理多少个序列）
        # num_tokens: 序列长度（每个序列有多少个token）
        # d_in: 输入维度（每个token的向量维度）

        # ========================================================================
        # 步骤2: 验证序列长度不超过模型支持的最大长度
        # ========================================================================
        # 确保输入序列长度不超过模型设计的最大上下文长度
        # 如果超过会报错，因为注意力掩码无法处理更长的序列
        assert num_tokens <= self.context_length, \
            f"num_tokens ({num_tokens}) must be less than or equal to context_length ({self.context_length})"

        # ========================================================================
        # 步骤3: 计算Query、Key、Value
        # ========================================================================
        # 将输入x分别通过三个线性层，得到Q、K、V
        # 每个输出的形状都是 (batch_size, num_tokens, d_out)
        keys = self.W_key(x)  # Key向量
        queries = self.W_query(x)  # Query向量
        values = self.W_value(x)  # Value向量

        # ========================================================================
        # 步骤4: 重排张量以分离多头
        # ========================================================================
        # 将最后一个维度(d_out)拆分为(num_heads, head_dim)
        # 通过增加一个维度来显式地分离不同的注意力头
        # .view()改变张量的形状而不改变数据
        # 形状变化: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # ========================================================================
        # 步骤5: 转置以使num_heads在batch_size之后
        # ========================================================================
        # 交换维度1和2，使头维度在token维度之前
        # 这样方便后续对每个头独立计算注意力
        # .transpose()交换两个维度
        # 形状变化: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # ========================================================================
        # 步骤6: 计算缩放点积注意力分数
        # ========================================================================
        # @ 是矩阵乘法运算符
        # queries @ keys.transpose(2, 3) 计算Query和Key的点积
        # keys.transpose(2, 3) 将最后两个维度转置: (b, n_heads, seq, head_dim) -> (b, n_heads, head_dim, seq)
        # 结果形状: (b, num_heads, num_tokens, num_tokens)
        # 这个张量表示每个token对其他所有token的注意力分数
        attn_scores = queries @ keys.transpose(2, 3)

        # ========================================================================
        # 步骤7: 应用因果掩码
        # ========================================================================
        # 将预计算的掩码裁剪到当前序列长度
        # [:num_tokens, :num_tokens] 切片操作，只取需要的部分
        # .bool()将张量转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 将掩码位置的注意力分数设为负无穷大
        # masked_fill_是原地操作，下划线表示修改原张量
        # 这是因为softmax之后，负无穷大会变成0概率
        # 这样实现了"未来的token不可见"的效果
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # ========================================================================
        # 步骤8: 计算注意力权重
        # ========================================================================
        # 缩放注意力分数以防止梯度消失
        # 除以head_dim的平方根（标准做法）
        # 然后应用softmax将分数转换为概率分布
        # dim=-1表示在最后一个维度上进行softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # 应用dropout进行正则化
        attn_weights = self.dropout(attn_weights)

        # ========================================================================
        # 步骤9: 计算上下文向量
        # ========================================================================
        # 将注意力权重与Value向量相乘
        # 每个token的输出是所有token value的加权求和
        # 形状: (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # ========================================================================
        # 步骤10: 合并多头
        # ========================================================================
        # 将头维度和head_dim维度合并
        # 转置回 (b, num_tokens, num_heads, head_dim)
        # 然后.contiguous()确保内存连续
        # .view()将 (b, num_tokens, num_heads, head_dim) 合并为 (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # ========================================================================
        # 步骤11: 应用输出投影
        # ========================================================================
        # 通过最终的线性层融合多头信息
        context_vec = self.out_proj(context_vec)

        # 返回上下文向量
        return context_vec
