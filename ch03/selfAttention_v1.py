import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    """
    A compact Self-Attention Module
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        # Self-Attention
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec  = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(3, 3)
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)
print(sa_v1(inputs))
# tensor([[0.6692, 1.0276, 1.1106],
#         [0.6864, 1.0577, 1.1389],
#         [0.6860, 1.0570, 1.1383],
#         [0.6738, 1.0361, 1.1180],
#         [0.6711, 1.0307, 1.1139],
#         [0.6783, 1.0441, 1.1252]], grad_fn=<MmBackward0>)
