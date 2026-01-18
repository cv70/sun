import torch
from torch.utils.data import Dataset, DataLoader
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
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = Tokenizer
    vocab_size = 50257
    output_dim = 256
    context_length = 1024


    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    batch_size = 8
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length
    )

    for batch in dataloader:
        x, y = batch

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings = token_embeddings + pos_embeddings

        break

    print(input_embeddings.shape)
