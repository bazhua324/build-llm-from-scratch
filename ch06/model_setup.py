import torch

from ch04.GPTModel import GPTModel
from ch05.gpt_download import download_and_load_gpt2
from ch05.load_pretrained_weights_from_openai import load_weights_into_gpt
from ch06.config import get_config


def load_pretrained_model(choose_model: str = "gpt2-small (124M)"):
    config = get_config(choose_model)
    model_size = choose_model.split(" ")[-1].strip("()")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.eval()
    return model, config


def prepare_for_classification(model, config, num_classes: int = 2):
    """Freeze everything, swap the output head, then unfreeze the parts we train."""
    for param in model.parameters():
        param.requires_grad = False

    model.out_head = torch.nn.Linear(config["emb_dim"], num_classes)

    # Only the last transformer block and the final norm are trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    return model