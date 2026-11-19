import json
import urllib

import psutil
from tqdm import tqdm

from ch07.data_prep import format_input

INPUT_PATH = "data/instruction-data-with-response.json"
OLLAMA_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "llama3"

def check_if_running(process_name):
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            return True
    return False

def query_model(prompt, model=JUDGE_MODEL, url=OLLAMA_URL):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "seed": 123,        # fixed seed for reproducible scores
            "temperature": 0,   # deterministic output
            "num_ctx": 2048,
        },
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Ollama streams the answer line by line as JSON
    response_text = ""
    with urllib.request.urlopen(request) as response:
        for line in response:
            line = line.decode("utf-8")
            if not line.strip():
                continue
            response_json = json.loads(line)
            response_text += response_json["message"]["content"]
    return response_text

def generate_model_scores(json_data, json_key="model_response", model=JUDGE_MODEL):
    scores = []
    for entry in tqdm(json_data, desc="Scoring"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
    return scores

def main():
    if not check_if_running("ollama"):
        raise RuntimeError("Ollama is not running")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # look at a few judgments with explanation first
    for entry in test_data[:3]:
        prompt = (
            f"Given the input '{format_input(entry)}' "
            f"and correct output '{entry['output']}', "
            f"score the model response '{entry['model_response']}'"
            f" on a scale from 0 to 100, where 100 is the best score.")

        print("\nDataset response:\n>>", entry["output"])
        print("\nModel response:\n>>", entry["model_response"])
        print("\nScore:\n>>", query_model(prompt))
        print("-" * 60)

    scores = generate_model_scores(test_data)
    print(f"\nNumber of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    main()

