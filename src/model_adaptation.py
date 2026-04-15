import os
import torch
from types import SimpleNamespace
from peft import LoraConfig, get_peft_model


def load_trufor_model(repo_root: str, ckpt_path: str, device: str = "cuda"):
    import sys
    sys.path.insert(0, repo_root)

    from lib.config import config, update_config
    from lib.utils import get_model

    args = SimpleNamespace(
        experiment="trufor_ph3",
        gpu=[0],
        opts=[]
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.gpu = range(len(args.gpu))

    update_config(config, args)

    model = get_model(config).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.eval()
    return model


def apply_lora_to_trufor(model):
    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "kv", "proj"],
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    for name, p in model.named_parameters():
        if "lora_" in name or "decode_head" in name or "decode_head_conf" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"Percent trainable: {100 * trainable / total:.4f}%")

    return model


def build_optimizer(model, lr: float = 1e-4, weight_decay: float = 1e-4):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )
