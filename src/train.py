"""Training loop for LoRA-adapted TruFor."""

import torch


def train(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
