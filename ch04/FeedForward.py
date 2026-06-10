import torch
from torch import nn
from ch04.GPT_CONFIG_124M import GPT_CONFIG_124M
from GELU import GELU


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used inside each transformer block.

    Expands the embedding dimension by a factor of 4 via a linear layer, applies GELU
    activation, then projects back down to the original embedding dimension.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x) # self.layers is an instance of nn.Sequential

print(GPT_CONFIG_124M["emb_dim"])
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)