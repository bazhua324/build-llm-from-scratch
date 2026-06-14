import torch

from ch04.GPTModel import GPTModel
from ch04.GPT_CONFIG_124M import GPT_CONFIG_124M


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :] # only cares about the predictions of the last token
        probas = torch.softmax(logits, dim=-1)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=1)

    return idx