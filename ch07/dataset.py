import torch
from torch.utils.data import Dataset
from ch07.data_prep import format_input


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = [tokenizer.encode(format_input(entry)) for entry in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

def custom_collate_fn(batch,
                      pad_token_id=50256,
                      ignore_index=-100,
                      allow_max_length=None):

    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1]) # Truncate the last token for inputs
        targets = torch.tensor(padded[1:]) # Shift +1 to the right for targets

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allow_max_length is not None:
            inputs = inputs[:allow_max_length]
            targets = targets[:allow_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Stack into (batch_size, seq_len) tensors and move to the target device
    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)

    return inputs_tensor, targets_tensor

if __name__ == "__main__":
    batch = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]
    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)