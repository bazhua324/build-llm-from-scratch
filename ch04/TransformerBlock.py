"""
Implements the TransformerBlock, the core repeating unit of a GPT-style language model.
"""
from torch import nn
from torch.nn.functional import dropout

from ch03.MultiHeadAttention import MultiHeadAttention
from ch04.FeedForward import FeedForward
from ch04.LayerNorm import LayerNorm


class TransformerBlock(nn.Module):
    """
    A single transformer block combining multi-head self-attention and a feed-forward network.

    Applies pre-layer normalization, multi-head attention with a residual shortcut, then
    pre-layer normalization again followed by a feed-forward network with another residual
    shortcut. Dropout is applied after each sub-layer before the residual addition.
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
            dropout=cfg["drop_rate"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.dropout_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x

        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)

        x = shortcut + x

        x = self.norm2(x)
        x = self.ff(x) # Liner layer -> GELU activation -> linear layer
        x = self.dropout_shortcut(x)

        x = x + shortcut

        return x

