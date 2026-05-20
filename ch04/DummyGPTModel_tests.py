import tiktoken
import torch
from ch04.GPT_CONFIG_124M import GPT_CONFIG_124M
from ch04.DummyGPTModel import DummyGPTModel
def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []
    txt1 = "Every effort moves you"

    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    print(batch)
    # [tensor([6109, 3626, 6100,  345]), tensor([6109, 1110, 6622,  257])]

    batch = torch.stack(batch, dim=0)
    print(batch)
    # tensor([[6109, 3626, 6100,  345],
    #         [6109, 1110, 6622,  257]])

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)

    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)

    # Output shape: torch.Size([2, 4, 50527])
    # tensor([[[-0.4811,  0.4325, -0.6385,  ..., -0.9398, -0.1178,  0.1945],
    #          [-0.0843,  0.4519,  0.2060,  ..., -0.5766, -0.5318, -1.4576],
    #          [ 0.7199,  0.0575,  0.2597,  ..., -1.0028, -1.9158, -0.5157],
    #          [ 1.8422,  0.0252,  0.9573,  ..., -0.0315, -0.2424, -0.2487]],
    #
    #         [[-0.9559,  1.3580, -0.4468,  ..., -0.5588,  0.4615, -0.3940],
    #          [-0.5088,  0.2881,  0.5815,  ..., -0.0353,  0.1133, -0.5654],
    #          [ 0.0683, -1.4644,  1.1000,  ..., -1.2006, -0.6936, -0.2516],
    #          [ 0.6819, -0.8203,  1.2607,  ...,  1.1441, -0.7271, -2.8555]]],
    #        grad_fn=<UnsafeViewBackward0>)

if __name__ == '__main__':
    main()

