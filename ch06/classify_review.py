import tiktoken
import torch

from ch06.model_setup import load_pretrained_model, prepare_for_classification


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_length = model.pos_emb.weight.shape[0]
    if max_length is None:
        max_length = supported_length
    max_length = min(max_length, supported_length)
    input_ids = input_ids[:max_length]                                  # truncate if too long
    input_ids += [pad_token_id] * (max_length - len(input_ids))        # pad to max_length

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]                          # last token only
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cpu")

    model, config = load_pretrained_model()
    model = prepare_for_classification(model, config)
    model.load_state_dict(torch.load("review_classifier.pth", map_location=device))
    model.eval()

    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    for text in (text_1, text_2):
        result = classify_review(text, model, tokenizer, device, max_length=120)
        print(f"{result:9} | {text[:60]}...")





