import time
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


scaler = torch.amp.GradScaler("cuda")


def extract_logits(outputs):
    if isinstance(outputs, dict):
        for k in ["pred", "anomaly", "mask", "out", "logits"]:
            if k in outputs:
                return outputs[k]
        return list(outputs.values())[0]

    if isinstance(outputs, (list, tuple)):
        # For your trained TruFor-LoRA pipeline, outputs[1] was the useful trainable map
        if len(outputs) > 1 and hasattr(outputs[1], "shape"):
            return outputs[1]
        return outputs[0]

    return outputs


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def seg_loss_fn(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dsc = dice_loss(logits, targets)
    return 0.7 * bce + 0.3 * dsc


def train_one_epoch(model, loader, optimizer, device="cuda", accum_steps=4, epoch=1, num_epochs=1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Train {epoch}/{num_epochs}", leave=True)
    start_time = time.time()

    for step, batch in enumerate(pbar):
        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            logits = extract_logits(outputs)

            if logits.ndim == 3:
                logits = logits.unsqueeze(1)

            if logits.shape[-2:] != masks.shape[-2:]:
                masks = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")

            loss = seg_loss_fn(logits, masks)
            loss_to_backprop = loss / accum_steps

        scaler.scale(loss_to_backprop).backward()

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        avg_loss = running_loss / (step + 1)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{avg_loss:.4f}",
            "gpu_mem_gb": f"{torch.cuda.memory_allocated() / 1024**3:.2f}"
        })

    if len(loader) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    epoch_time = time.time() - start_time
    return running_loss / len(loader), epoch_time


@torch.no_grad()
def validate_one_epoch(model, loader, device="cuda", epoch=1, num_epochs=1):
    model.eval()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Valid {epoch}/{num_epochs}", leave=True)
    start_time = time.time()

    for step, batch in enumerate(pbar):
        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            logits = extract_logits(outputs)

            if logits.ndim == 3:
                logits = logits.unsqueeze(1)

            if logits.shape[-2:] != masks.shape[-2:]:
                masks = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")

            loss = seg_loss_fn(logits, masks)

        running_loss += loss.item()
        avg_loss = running_loss / (step + 1)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{avg_loss:.4f}"
        })

    epoch_time = time.time() - start_time
    return running_loss / len(loader), epoch_time


def fit(model, train_loader, valid_loader, optimizer, device="cuda", num_epochs=8, save_path="best_model.pth"):
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_time = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            accum_steps=4,
            epoch=epoch,
            num_epochs=num_epochs
        )

        val_loss, val_time = validate_one_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs
        )

        print(
            f"\nEpoch {epoch}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"train_time={train_time/60:.2f} min | "
            f"val_time={val_time/60:.2f} min"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path
            )
            print("Saved best model\n")
