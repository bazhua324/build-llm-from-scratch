import json
import os
import re

import tiktoken
import torch
from tqdm import tqdm

from ch04.GPTModel import GPTModel
from ch04.generate_text_simple import generate
from ch05.load_pretrained_weights_from_openai import get_config
from ch05.pretraining_on_unlabeled_data import text_to_token_ids, token_ids_to_text
from ch07.data_prep import download_and_load_file, split_data, format_input

MODEL_NAME = "gpt2-small (124M)"
MODEL_PATH = f"models/{re.sub(r'[ ()]', '', MODEL_NAME)}-sft.pth"
DEBUG = False
OUTPUT_PATH = "data/instruction-data-with-response.json"

def generate_response(model, tokenizer, entry, config, device, max_new_tokens=256):
    input_text = format_input(entry)
    token_ids = generate(
        model,
        idx = text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=config["context_length"],
        eos_id = 50256
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)

    return generated_text[len(input_text):].replace("### Response:", "").strip()

def load_finetuned_model(device):
    config = get_config(MODEL_NAME)
    model = GPTModel(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, config

def main():
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    # Data
    data = download_and_load_file()
    train_data, val_data, test_data = split_data(data)

    if DEBUG:
        test_data = test_data[:5]

    model, config = load_finetuned_model(device)

    # generate for the whole test set set
    for entry in tqdm(test_data, desc="Generating"):
        entry["model_response"] = generate_response(model, tokenizer, entry, config, device)

    # Quick look at the first 3
    for entry in test_data[:3]:
        print(format_input(entry))
        print(f"Correct response: {entry['output']}")
        print(f"Model response: {entry['model_response']}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)
    print(f"Saved {len(test_data)} entries to {OUTPUT_PATH}")



if __name__ == "__main__":
    main()