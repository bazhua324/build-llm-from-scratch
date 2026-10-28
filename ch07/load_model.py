import tiktoken
import torch

from ch04.generate_text_simple import generate
from ch05.pretraining_on_unlabeled_data import text_to_token_ids, token_ids_to_text
from ch06.model_setup import load_pretrained_model
from ch07.data_prep import download_and_load_file, split_data, format_input

if __name__ == "__main__":
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")
    model, config = load_pretrained_model()

    data = download_and_load_file()
    train_data, val_data, test_data = split_data(data)

    input_text = format_input(val_data[0])
    print(input_text)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=config["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].strip()
    print(response_text)

