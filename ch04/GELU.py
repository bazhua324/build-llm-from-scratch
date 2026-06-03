import torch
import torch.nn as nn

class GELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) activation function.
    Used in GPT instead of ReLU because it provides a smoother transition
    around zero, which helps training. Negative values are not hard-cut to 0
    like ReLU, but smoothly approach 0.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


