import math
import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock
from model.feed_forward import LayerNorm

class RotaryPE(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """ 旋转位置编码
            - dim (int): 旋转嵌入的维度大小。
            - max_position_embeddings (int): 预计算的最大位置嵌入数，默认为2048。
            - base (int): 用于计算逆频率的基本频率，默认为10000。
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算逆频率值，并将其注册为模型的缓冲区
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了支持`torch.jit.trace`功能，立即计算预存储的余弦和正弦缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """ 预计算的余弦和正弦缓存
        """
        self.max_seq_len_cached = seq_len
        # 创建一个从0到最大序列长度-1的整数张量，与 inv_freq 具有相同的设备和数据类型
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算每个位置与每个维度的频率，形成频谱矩阵
        freqs = torch.outer(t, self.inv_freq)
        
        # 不同于论文中的实现，这里采用了不同的排列方式以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class SinusoidPE(nn.Module):
    def __init__(self, ctx_len, emb_dim):
        super().__init__()
        # 创建位置编码矩阵 [c, d]
        pe = torch.zeros(ctx_len, emb_dim)
        position = torch.arange(0, ctx_len).unsqueeze(1)
        
        # 计算频率除数: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * 
                           -(math.log(10000.0) / emb_dim))
        
        # 应用正弦余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引: 正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引: 余弦

        # 增加批次维度 [c, d]
        pe.unsqueeze(0)
        
        # 注册为缓冲区 (不参与训练)
        self.register_buffer('pe', pe)


class LLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ctx_len = cfg["context_length"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.pe = SinusoidPE(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

        # 使用bfloat16精度
        self.bfloat16()

    def forward(self, in_idx):
        # in_idx: (batch_size, seq_len)
        b, s = in_idx.shape

        # (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(s, device=in_idx.device))
        x = tok_embeds + pos_embeds
        # x = tok_embeds + self.pe.pe[:s, :]  # (batch_size, seq_len, emb_dim)
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x) # (batch_size, seq_len, vocab_size)
        return logits
