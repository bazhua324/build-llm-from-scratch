import json
import os
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
DATA_PATH = "instruction-data.json"


def download_and_load_file(file_path=DATA_PATH,url=DATA_URL):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def split_data(data, train_frac=0.85, test_ratio=0.1):
    train_end = int(len(data) * train_frac)
    test_end = train_end + int(len(data) * test_ratio)

    train_data = data[:train_end]
    test_data = data[train_end:test_end]
    val_data = data[test_end:]
    return train_data, val_data, test_data

def format_input(entry):
    instruction_text = (
                        f"Below is an instruction that describes a task. "
                        f"Write a response that appropriately completes the request."
                        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

def format_full(entry):
    return format_input(entry) + f"\n\n### Response:\n{entry['output']}"

if __name__ == "__main__":
    data = download_and_load_file()

    print("Number of entries:", len(data))
    print("Example entry:\n", data[50])
    print("Another example entry:\n", data[999])

    train_data, val_data, test_data = split_data(data)
    print(f"\ntrain: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"

    print(model_input + desired_response)
