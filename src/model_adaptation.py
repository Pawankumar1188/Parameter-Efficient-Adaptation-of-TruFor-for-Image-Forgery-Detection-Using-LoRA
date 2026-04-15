"""Model adaptation logic for TruFor + LoRA."""

import torch


def build_model():
    # TODO: implement TruFor loading and LoRA adapter injection
    raise NotImplementedError("Model construction is not implemented yet.")


def freeze_base_parameters(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if "lora" not in name and "decode_head" not in name:
            param.requires_grad = False
