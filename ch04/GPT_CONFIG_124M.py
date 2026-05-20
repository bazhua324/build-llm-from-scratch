# Configuration dictionary for the GPT-2 124M parameter model, defining vocab size,
# context length, embedding dimensions, number of attention heads, transformer layers,
# dropout rate, and whether to use bias in QKV projections.
GPT_CONFIG_124M = {
    "vocab_size": 50527,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}