import tiktoken
import torch
from torch.utils.data import DataLoader

from ch06.spam_dataset import SpamDataset


def create_dataloaders(batch_size: int = 8, num_workers: int = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    train_dataset = SpamDataset("train.csv", tokenizer, max_length=None)
    val_dataset = SpamDataset("validation.csv", tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset("test.csv", tokenizer, max_length=train_dataset.max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader, tokenizer