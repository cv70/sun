"""
=============================================================================
分词器训练模块 (train_sp.py)
=============================================================================

【模块作用】
训练SentencePiece分词器

【相关知识】
1. SentencePiece: Google开源的通用分词工具
2. BPE (Byte Pair Encoding): 子词分词算法
3. vocab_size: 词表大小
4. character_coverage: 字符覆盖率，决定有多少字符会被分配token
"""

import sentencepiece as spm
import os
import glob

def train_chinese_spm(input_txt_dir, vocab_size, output_dir="."):
    """
    训练中文SentencePiece模型

    参数:
        input_txt_dir: 输入文本目录
        vocab_size: 词表大小
        output_dir: 输出目录

    相关知识：
    - SentencePiece会自动学习最优的分词方式
    - BPE算法从字符级开始，逐步合并最常见的字节对
    """
    # 保存的模型名称前缀
    prefix = os.path.join(output_dir, "spm")

    # 获取所有文本文件
    text_filenames = sorted(glob.glob(os.path.join(input_txt_dir, "*.md")))
    print("file list: ", text_filenames)

    # 训练SentencePiece模型
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(
        input=text_filenames,  # 输入文件列表
        model_prefix=prefix,  # 输出文件前缀
        model_type="bpe",  # 使用BPE算法
        vocab_size=vocab_size,  # 词表大小
        self_test_sample_size=1,
        input_format="text",
        character_coverage=0.9995,  # 字符覆盖率
        num_threads=os.cpu_count(),  # 使用所有CPU核心
        split_digits=True,  # 将数字划分为单个token
        allow_whitespace_only_pieces=True,
        byte_fallback=True,  # 字节回退，处理未知字符
        unk_surface=r" \342\201\207 ",  # 未知token的表示
        max_sentence_length=24000  # 最大句子长度
    )

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")

def test_chinese_spm(spm_model_path):
    """
    测试SentencePiece模型

    参数:
        spm_model_path: 模型文件路径
    """
    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load(spm_model_path)

    input_txt = '为了中华民族的伟大复兴而奋斗'

    # 编码为pieces
    pieces = sp_bpe.EncodeAsPieces(input_txt)
    print("Pieces:", pieces)

    # 编码为IDs
    token_ids = sp_bpe.EncodeAsIds(input_txt)
    print("Token IDs:", token_ids)

    # 解码为文本
    parsed_text = sp_bpe.DecodeIds(token_ids)
    print("Decoded text:", parsed_text)

if __name__ == "__main__":
    input_txt_dir = "../dataset"
    vocab_size = 16384
    output_dir = "../tokenizer"
    train_chinese_spm(input_txt_dir, vocab_size, output_dir)
    test_chinese_spm(f"{output_dir}/spm_16384.model")
