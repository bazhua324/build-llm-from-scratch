# Improved self-attention (v2) that replaces raw nn.Parameter weights with
# nn.Linear layers for better weight initialization and training stability.
import torch
from torch import nn

class SelfAttention_v2(nn.Module):
    """
    A Self-Attention Module using PyTorch's Linear layer. nn.Linear over manula nn.Parameter approach is that nn.Linear has a preffered weight initilization schema, which leads to more stable model training.
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # nn.Linear has a preferred weight initialization scheme, which leads to more stable model training
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v2 = SelfAttention_v2(3, 3)
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)
print(sa_v2(inputs))
# tensor([[ 0.2633,  0.4277, -0.1353],
#         [ 0.2641,  0.4296, -0.1350],
#         [ 0.2641,  0.4296, -0.1350],
#         [ 0.2647,  0.4316, -0.1381],
#         [ 0.2642,  0.4303, -0.1373],
#         [ 0.2648,  0.4316, -0.1375]], grad_fn=<MmBackward0>)