import torch
from torch import nn

class LayerNorm(nn.Module):
    """
    Layer Normalization
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # a smaller value (eps)
        # a scale and a shift are two trainable parameters that LLM automatically adjusts during training if it is determined that doing so could improve the model's performance on its training task .
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # Add a smaller value (eps) before computing the square root of the variance; this i s to avoid division-by-zeros if the variance is 0 .
        return self.scale * norm_x + self.shift

