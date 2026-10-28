import tiktoken
import torch

from ch04.GPTModel import GPTModel
from ch04.generate_text_simple import generate
from ch04.GPT_CONFIG_124M import GPT_CONFIG_124M
from ch05.pretraining_on_unlabeled_data import text_to_token_ids, token_ids_to_text

# file_name = "gpt2-small-124M.pth"
#
# url =  f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"
#
# if not os.path.exists(file_name):
#     urllib.request.urlretrieve(url, file_name)
#     print(f"Downloaded to {file_name}")
#
# gpt = GPTModel(BASE_CONFIG)
# gpt.load_state_dict(torch.load(file_name, weights_only=True))
# gpt.eval()
#
# torch.manual_seed(123)
# tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate(
#     model=gpt,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=25,
#     context_size=NEW_CONFIG["context_size"],
#     top_k=50,
#     temperature=1.5,
# )
#
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


from ch05.gpt_download import download_and_load_gpt2

# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

def get_config(model_name = "gpt2-small (124M)", drop_rate=0.0):
    config = GPT_CONFIG_124M.copy()
    config.update(model_configs[model_name])
    config.update({
        "context_length": 1024,
        "qkv_bias": True,
        "drop_rate": drop_rate
                   })

    return config

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError("Shape mismatch: {} vs {}".format(left.shape, right.shape))

    return torch.nn.Parameter(torch.tensor(right))

import numpy as np
def load_weights_into_gpt(model, params):
    model.pos_emb.weight = assign(model.pos_emb.weight, params['wpe'])
    model.tok_emb.weight = assign(model.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        model.trf_blocks[b].att.W_query.weight = assign(
            model.trf_blocks[b].att.W_query.weight, q_w.T)
        model.trf_blocks[b].att.W_key.weight = assign(
            model.trf_blocks[b].att.W_key.weight, k_w.T)
        model.trf_blocks[b].att.W_value.weight = assign(
            model.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        model.trf_blocks[b].att.W_query.bias = assign(
            model.trf_blocks[b].att.W_query.bias, q_b)
        model.trf_blocks[b].att.W_key.bias = assign(
            model.trf_blocks[b].att.W_key.bias, k_b)
        model.trf_blocks[b].att.W_value.bias = assign(
            model.trf_blocks[b].att.W_value.bias, v_b)

        model.trf_blocks[b].att.out_proj.weight = assign(
            model.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        model.trf_blocks[b].att.out_proj.bias = assign(
            model.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        model.trf_blocks[b].ff.layers[0].weight = assign(
            model.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        model.trf_blocks[b].ff.layers[0].bias = assign(
            model.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        model.trf_blocks[b].ff.layers[2].weight = assign(
            model.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        model.trf_blocks[b].ff.layers[2].bias = assign(
            model.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        model.trf_blocks[b].norm1.scale = assign(
            model.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        model.trf_blocks[b].norm1.shift = assign(
            model.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        model.trf_blocks[b].norm2.scale = assign(
            model.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        model.trf_blocks[b].norm2.shift = assign(
            model.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    model.final_norm.scale = assign(model.final_norm.scale, params["g"])
    model.final_norm.shift = assign(model.final_norm.shift, params["b"])
    model.out_head.weight = assign(model.out_head.weight, params["wte"])

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            return torch.device("mps")

    return torch.device("cpu")

def load_pretrained_model(model_name="gpt2-small (124M)", models_dir="models/gpt2", device=None):

    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

    config = get_config(model_name)
    model = GPTModel(config)
    load_weights_into_gpt(model, params)

    if device is not None:
        model.to(device)
    model.eval()

    return model, config

if __name__ == "__main__":
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")

    device = get_device()
    print("device:", device)

    model, config = load_pretrained_model(device=device)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=config["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))








