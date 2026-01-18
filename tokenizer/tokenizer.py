import sentencepiece as spm
import tiktoken
import torch

class Tokenizer():
    def __init__(self, spm_model_path):
        self.sp_bpe = spm.SentencePieceProcessor()
        self.sp_bpe.load(spm_model_path)

    def text_to_token_ids(self, text):
        encoded = self.sp_bpe.EncodeAsIds(text)
        return encoded
        encoded_tensor = torch.tensor(encoded)#.unsqueeze(0) # 添加批次维度
        return encoded_tensor

    def token_ids_to_text(self, token_ids):
        decoded = self.sp_bpe.DecodeIds(token_ids)
        return decoded
        flat = token_ids.squeeze(0) # 移除批次维度
        return self.sp_bpe.DecodeIds(flat.tolist())
    
    def vocab_size(self):
        return self.sp_bpe.vocab_size()


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文中的索引数组(b, n_tokens)
    for _ in range(max_new_tokens):
        
        # 如果当前上下文超出支持的上下文大小，则裁剪当前上下文
        # 例如，如果LLM只支持5个token，而上下文大小为10
        # 那么只有最后5个token被用作上下文
        idx_cond = idx[:, -context_size:]
        # print(idx_cond)
        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 只关注最后一个时间步
        # (b, n_tokens, vocab_size) 变为 (b, vocab_size)
        logits = logits[:, -1, :]  

        # 应用softmax获取概率
        probas = torch.softmax(logits, dim=-1)  # (b, vocab_size)

        # 获取具有最高概率值的词汇条目索引
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (b, 1)

        # 将采样的索引追加到运行序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (b, n_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={''})
    encoded_tensor = torch.tensor(encoded)#.unsqueeze(0) # 添加批次维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # 移除批次维度
    return tokenizer.decode(flat.tolist())


if __name__ == "__main__":
    from config.config import LLM_CONFIG
    from model.llm import LLM

    start_context = "郭靖挥出一拳"
    tokenizer = Tokenizer(LLM_CONFIG['tokenizer_path'])

    model = LLM(LLM_CONFIG)
    model.eval() # 设置为评估模式

    torch.manual_seed(123)
    token_ids = generate_text_simple(
        model=model,
        idx=tokenizer.text_to_token_ids(start_context),
        max_new_tokens=10,
        context_size=LLM_CONFIG["context_length"]
    )
    text = tokenizer.token_ids_to_text(token_ids)
    print(text)
