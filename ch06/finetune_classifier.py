"""Fine-tune GPT-2 for spam classification (Chapter 6).

Run: uv run python ch06/finetune_classifier.py
"""
import time

import torch

from ch06.data_loaders import create_dataloaders
from ch06.evaluation import calc_accuracy_loader, calc_loss_batch, calc_loss_loader, evaluate_model
from ch06.model_setup import load_pretrained_model, prepare_for_classification
from ch06.plotting import plot_values


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            return torch.device("mps")
    return torch.device("cpu")


def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):
    """
    Train the classifier and return the tracked metrics. Plotting, final evaluation and saving are the caller's job.
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Quick accuracy check at the end of each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | "
              f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        # Checkpoint after every epoch so a crash doesn't cost the whole run
        torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")

    return train_losses, val_losses, train_accs, val_accs, examples_seen


if __name__ == "__main__":
    # --- Setup ---
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders()
    model, config = load_pretrained_model()
    model = prepare_for_classification(model, config)

    device = get_device()
    print("Device:", device)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.1%})")

    # --- Baseline before training ---
    torch.manual_seed(123)
    print("\n=== Baseline (before fine-tuning) ===")
    with torch.no_grad():
        print(f"Train loss: {calc_loss_loader(train_loader, model, device, num_batches=5):.3f}")
        print(f"Val loss:   {calc_loss_loader(val_loader, model, device, num_batches=5):.3f}")
    print(f"Train acc:  {calc_accuracy_loader(train_loader, model, device, num_batches=5)*100:.2f}%")
    print(f"Val acc:    {calc_accuracy_loader(val_loader, model, device, num_batches=5)*100:.2f}%")

    # --- Training ---
    print("\n=== Fine-tuning ===")
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    # --- Save the fine-tuned model ---
    torch.save(model.state_dict(), "review_classifier.pth")
    print("Model saved to review_classifier.pth")

    # --- Plots ---
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_tensor, train_losses, val_losses, label="loss")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_tensor, train_accs, val_accs, label="accuracy")

    # --- Final evaluation on the full datasets ---
    print("\n=== Final results (full datasets) ===")
    print(f"Training accuracy:   {calc_accuracy_loader(train_loader, model, device)*100:.2f}%")
    print(f"Validation accuracy: {calc_accuracy_loader(val_loader, model, device)*100:.2f}%")
    print(f"Test accuracy:       {calc_accuracy_loader(test_loader, model, device)*100:.2f}%")