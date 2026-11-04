import time

import tiktoken
from torch.utils.data import DataLoader
import torch

from ch05.load_pretrained_weights_from_openai import load_pretrained_model
from ch05.pretraining_on_unlabeled_data import calc_loss_loader, train_model_simple
from ch07.data_prep import download_and_load_file, split_data, format_input
from ch07.dataset import InstructionDataset, custom_collate_fn

MODEL_NAME = "gpt2-small (124M)"
NUM_EPOCHES = 10
BATCH_SIZE = 8
NUM_WORKERS = 0
DEBUG = True

def main():
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    data = download_and_load_file()
    train_data, val_data, test_data = split_data(data)

    if DEBUG:
        # Small subset for a quick run
        train_data, val_data = train_data[:80], val_data[:16]

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = InstructionDataset(train_data, tokenizer)

    # DataLoader
    train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=custom_collate_fn,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=NUM_WORKERS)

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                collate_fn=custom_collate_fn,
                                shuffle=False,
                                drop_last=False,
                                num_workers=NUM_WORKERS)

    # Model
    model, config = load_pretrained_model(model_name=MODEL_NAME, device=device)

    # loss before training
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print("Before finetuning - Train loss: ", train_loss)
    print("Before finetuning - Val loss: ", val_loss)

    # Finetune
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=NUM_EPOCHES,
        eval_iter=5,
        eval_freq=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )
    minutes = (time.time() - start_time) / 60
    print(f"Training completed in {minutes} minutes.")

if __name__ == "__main__":
    main()