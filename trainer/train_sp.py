import sentencepiece as spm
import os
import glob

def tain_chinese_spm(input_txt_dir, vocab_size, output_dir="."):
    # 保存的模型名称
    prefix = os.path.join(output_dir, "spm")

    text_filenames = sorted(glob.glob(os.path.join(input_txt_dir, "*.md")))
    print("file list: ", text_filenames)

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(input=text_filenames,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=1,
                                   input_format="text",
                                   character_coverage=0.9995,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,       # 是否将数字划分为单个 token, 在 llama 中是这么做的
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   max_sentence_length=24000)


    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")

def test_chinese_spm(spm_model_path):
    sp_bpe = spm.SentencePieceProcessor() 
    sp_bpe.load(spm_model_path)

    input_txt = '为了中华民族的伟大复兴而奋斗'

    pieces = sp_bpe.EncodeAsPieces(input_txt)
    print(pieces)

    token_ids = sp_bpe.EncodeAsIds(input_txt)
    print(token_ids)

    parsed_pieces = sp_bpe.IdToPiece(token_ids)
    print(parsed_pieces)

    parsed_text = sp_bpe.DecodeIds(token_ids)
    print(parsed_text)

    parsed_ids = sp_bpe.PieceToId(pieces)
    print(parsed_ids)

if __name__ == "__main__":
    input_txt_dir = "../dataset"
    vocab_size = 16384
    output_dir = "../tokenizer"
    tain_chinese_spm(input_txt_dir, vocab_size, output_dir)
    test_chinese_spm(f"{output_dir}/spm_16384.model")
