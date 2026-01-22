import os
import sys
import sentencepiece as spm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import LLM_CONFIG
from model.llm import LLM
from utils.generate import generate

class Tokenizer():
    def __init__(self, spm_model_path):
        self.sp_bpe = spm.SentencePieceProcessor()
        self.sp_bpe.load(spm_model_path)

    def text_to_token_ids(self, text, add_eos=False):
        encoded = self.sp_bpe.EncodeAsIds(text, add_eos=add_eos)
        return encoded

    def token_ids_to_text(self, token_ids):
        decoded = self.sp_bpe.DecodeIds(token_ids)
        return decoded
    
    def vocab_size(self):
        return self.sp_bpe.vocab_size()
    
    def eos_id(self):
        return self.sp_bpe.eos_id()


if __name__ == "__main__":
    start_context = "郭靖挥出一拳</s>"
    tokenizer = Tokenizer(LLM_CONFIG['tokenizer_path'])

    model = LLM(LLM_CONFIG)
    model.eval() # 设置为评估模式

    torch.manual_seed(123)
    start_context = tokenizer.text_to_token_ids(start_context, add_eos=False)
    print(start_context)
    start_context = torch.tensor([start_context])
    token_ids = generate(
        model=model,
        idx=start_context,
        max_new_tokens=16,
        context_size=LLM_CONFIG["context_length"],
        eos_id=tokenizer.eos_id(),
    )
    text = tokenizer.token_ids_to_text(token_ids.tolist())
    print(text)
