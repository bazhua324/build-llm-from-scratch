import torch

from ch04.GPTModel import GPTModel
from ch05.GPT_CONFIG_124M import GPT_CONFIG_124M

model = GPTModel(GPT_CONFIG_124M)

torch.save(model.state_dict(), 'model.pth')

# Save adaptive optimizers
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
},
"model_and_optimizer.pth"
)

# Initialize weights and optimizer
checkpoint = torch.load('model_and_optimizer.pth', weights_only=True)

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.train()