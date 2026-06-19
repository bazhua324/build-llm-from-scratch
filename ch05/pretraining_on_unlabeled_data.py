import tiktoken
import torch

from ch02.dataloader import create_dataloader_v1
from ch04.GPTModel import GPTModel
from ch04.generate_text_simple import generate_text_simple

from ch05.GPT_CONFIG_124M import GPT_CONFIG_124M

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval() # Disable dropout during inference

tokenizer = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dim
    print(encoded_tensor)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def use_gpt_to_generate_text():
    start_context = "effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text=start_context, tokenizer=tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print(f"Output text: \n", token_ids_to_text(token_ids, tokenizer=tokenizer))

def calculating_the_text_generation_loss():
    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                           [40,    1107, 588]])   #  "I really like"]

    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]

    with torch.no_grad():
        logits = model(inputs) # Raw scores for each vocabulary token at every position

    probas = torch.softmax(logits, dim=-1) # Softmax converts logits into probabilities that sum to 1 .
    print(f"Probas Shape: ", {probas.shape})

    token_ids = torch.argmax(probas, dim=-1, keepdim=True) # Pick the token with the highest probability at each position
    print(f"Token IDs:\n", {token_ids})

    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer=tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer=tokenizer)}")

    # The token probabilities corresponding to the target indices
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print(f"Text 1: {target_probas_1}")

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print(f"Text 2: {target_probas_2}")

    # Compute Log of all token probabilities
    # The probability < 1, log is negative
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)
    # tensor([-11.1986, -10.1733,  -9.6486, -11.4122, -10.1179, -10.3542])

    # Calculate the average probabilities for each token
    avg_log_probas = torch.mean(log_probas)
    print(f"Average log probability: {avg_log_probas}")
    # -10.484132766723633

    neg_avg_log_probas = avg_log_probas * -1    # cross_entropy loss
    print(f"Negative log probability: {neg_avg_log_probas}")
    # 10.484132766723633

    # Logits have shape (batch_size, num_tokens, vocab_size)
    print(f"Logtis shape: ", logits.shape)

    # Targets have shape (batch_size, num_tokens)
    print(f"Targets shape:", targets.shape)

    # Flatten Logits tensor and Targets tensor for the cross_entropy function
    logits_flatten = logits.flatten(0, 1)
    targets_flatten = targets.flatten()

    print(f"Flattened logits shape: {logits_flatten.shape}")
    print(f"Flattened targets shape: {targets_flatten.shape}")

    loss = torch.nn.functional.cross_entropy(logits_flatten, targets_flatten)
    print(f"Loss: {loss}") # entropy loss is equal to negative log probability

    perplexity = torch.exp(loss) # The perplexity is often considered more interpretable because it can be understood as the effective vocabulary size that the model is uncertain about at each step (in the example, that'd be 35743 words or tokens)
    print(f"Perplexity: {perplexity}")
    # 35743.7890625

def calculate_the_training_and_validation_set_loss():
    file_path = "../ch02/the-verdict.txt"
    with open(file_path, "r") as f:
        text_data = f.read()
    print(text_data[:99]) # First 99 characters
    print(text_data[-99:]) # Last 99 characters

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print("total_characters:", total_characters)
    print("total_token:", total_tokens)

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(total_characters * train_ratio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    # Sanity check
    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
              "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
              "decrease the `training_ratio`")

    def calc_loss_batch(input_batch, output_batch, model, device):
        input_batch, output_batch = input_batch.to(device) , output_batch.to(device)
        logits = model(input_batch)

        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            output_batch.flatten(),
        )
        return loss

    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0

        if num_batches is None:
            num_batches = len(data_loader)

        for i, (input_batch, output_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, output_batch, model, device)
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batches

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

        print(f"Train loss: {train_loss}")
        print(f"Val loss: {val_loss}")












if __name__ == "__main__":
    # use_gpt_to_generate_text()
    # calculating_the_text_generation_loss()
    calculate_the_training_and_validation_set_loss()
