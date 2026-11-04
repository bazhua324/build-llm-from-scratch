from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
import torch

from ch02.dataloader import create_dataloader_v1
from ch04.GPTModel import GPTModel
from ch04.generate_text_simple import generate_text_simple
from ch05.GPT_CONFIG_124M import GPT_CONFIG_124M


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded = token_ids_to_text(token_ids, tokenizer)
    print(decoded.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path=None):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def use_gpt_to_generate_text(model, tokenizer, device):
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def calculating_the_text_generation_loss(model, tokenizer, device):
    inputs = torch.tensor([[16833, 3626, 6100],
                           [40, 1107, 588]])
    targets = torch.tensor([[3626, 6100, 345],
                            [1107, 588, 11311]])
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1)
    print("Probas shape:", probas.shape)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)
    print("Targets batch 1:", token_ids_to_text(targets[0], tokenizer))
    print("Outputs batch 1:", token_ids_to_text(token_ids[0].flatten(), tokenizer))

    target_probas_1 = probas[0, [0, 1, 2], targets[0]]
    target_probas_2 = probas[1, [0, 1, 2], targets[1]]
    print("Text 1:", target_probas_1)
    print("Text 2:", target_probas_2)

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    avg_log_probas = torch.mean(log_probas)
    print("Average log probability:", avg_log_probas)
    print("Negative average log probability:", -avg_log_probas)

    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits shape:", logits_flat.shape)
    print("Flattened targets shape:", targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("Loss:", loss)
    print("Perplexity:", torch.exp(loss))


def calculate_the_training_and_validation_set_loss(model, tokenizer, device):
    file_path = Path(__file__).resolve().parent.parent / "ch02" / "the-verdict.txt"
    text_data = file_path.read_text(encoding="utf-8")

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Total characters:", total_characters)
    print("Total tokens:", total_tokens)

    train_ratio = 0.90
    split_idx = int(total_characters * train_ratio)
    train_data, val_data = text_data[:split_idx], text_data[split_idx:]

    ctx = GPT_CONFIG_124M["context_length"]

    if total_tokens * train_ratio < ctx:
        print("Not enough tokens for the training loader.")
    if total_tokens * (1 - train_ratio) < ctx:
        print("Not enough tokens for the validation loader.")

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data, batch_size=2, max_length=ctx, stride=ctx,
        drop_last=True, shuffle=True, num_workers=0)
    val_loader = create_dataloader_v1(
        val_data, batch_size=2, max_length=ctx, stride=ctx,
        drop_last=False, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer,
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    calculate_the_training_and_validation_set_loss(model, tokenizer, device)