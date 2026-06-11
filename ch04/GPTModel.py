import tiktoken
import torch
from torch import nn

from ch04.LayerNorm import LayerNorm
from ch04.TransformerBlock import TransformerBlock

from ch04.GPT_CONFIG_124M import GPT_CONFIG_124M


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(in_idx.shape[1], device=tok_embeds.device)
                         ) # Allow training model on CPU/GPU, depending on where the inputs are

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

# Create batch
tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

out = model(batch)
print(f"Input batch: \n {batch}")
print(f"\nOutput shape: \n {out}")


