LLM_CONFIG= {
    "data_dir": "dataset",
    "tokenizer_path": "spm_16384.model",
    "vocab_size": 16384,    # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 256,         # Embedding dimension
    "n_heads": 8,           # Number of attention heads
    "n_layers": 8,          # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
