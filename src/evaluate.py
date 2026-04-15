import numpy as np
import pandas as pd
import torch
from PIL import Image


def load_image_tensor(image_path, image_size=512):
    img_pil = Image.open(image_path).convert("RGB")
    img_resized = img_pil.resize((image_size, image_size))

    x = np.array(img_resized).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    return img_pil, x


def load_mask_tensor(mask_path, image_size=512):
    m = Image.open(mask_path).convert("L").resize((image_size, image_size))
    m = (np.array(m) > 127).astype(np.float32)
    return m


@torch.no_grad()
def infer_map_from_model(model_obj, image_tensor, device="cuda", use_lora_branch=False):
    model_obj.eval()

    with torch.amp.autocast("cuda"):
        outputs = model_obj(image_tensor.to(device))

        if isinstance(outputs, (list, tuple)):
            if use_lora_branch:
                if len(outputs) > 1 and hasattr(outputs[1], "shape"):
                    pred = outputs[1]
                else:
                    pred = outputs[0]
            else:
                pred = outputs[0]

        elif isinstance(outputs, dict):
            for k in ["pred", "anomaly", "mask", "out", "logits"]:
                if k in outputs:
                    pred = outputs[k]
                    break
            else:
                pred = list(outputs.values())[0]
        else:
            pred = outputs

        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        if pred.shape[1] == 2:
            pred = torch.softmax(pred, dim=1)[:, 1:2, :, :]
        else:
            pred = torch.sigmoid(pred)

        pred = pred[0, 0].detach().cpu().numpy()

    return pred


def compute_metrics(pred_bin, gt_bin, eps=1e-7):
    pred_bin = pred_bin.astype(np.uint8)
    gt_bin = gt_bin.astype(np.uint8)

    tp = np.logical_and(pred_bin == 1, gt_bin == 1).sum()
    fp = np.logical_and(pred_bin == 1, gt_bin == 0).sum()
    fn = np.logical_and(pred_bin == 0, gt_bin == 1).sum()
    tn = np.logical_and(pred_bin == 0, gt_bin == 0).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(acc),
    }


def evaluate_models_on_fake_samples(df_eval, baseline_model, trained_model, device="cuda", image_size=512, threshold=0.1):
    rows = []

    for _, row in df_eval.iterrows():
        image_path = row["imagepath"]
        mask_path = row["maskpath"]

        _, x = load_image_tensor(image_path, image_size=image_size)
        gt = load_mask_tensor(mask_path, image_size=image_size)

        base_map = infer_map_from_model(baseline_model, x, device=device, use_lora_branch=False)
        lora_map = infer_map_from_model(trained_model, x, device=device, use_lora_branch=True)

        base_bin = (base_map > threshold).astype(np.uint8)
        lora_bin = (lora_map > threshold).astype(np.uint8)
        gt_bin = gt.astype(np.uint8)

        m_base = compute_metrics(base_bin, gt_bin)
        m_lora = compute_metrics(lora_bin, gt_bin)

        rows.append({
            "imagepath": image_path,
            "baseline_dice": m_base["dice"],
            "baseline_iou": m_base["iou"],
            "baseline_precision": m_base["precision"],
            "baseline_recall": m_base["recall"],
            "baseline_accuracy": m_base["accuracy"],
            "lora_dice": m_lora["dice"],
            "lora_iou": m_lora["iou"],
            "lora_precision": m_lora["precision"],
            "lora_recall": m_lora["recall"],
            "lora_accuracy": m_lora["accuracy"],
        })

    return pd.DataFrame(rows)


def evaluate_models_on_real_samples(df_eval, baseline_model, trained_model, device="cuda", image_size=512, threshold=0.1):
    rows = []

    for _, row in df_eval.iterrows():
        image_path = row["imagepath"]

        _, x = load_image_tensor(image_path, image_size=image_size)

        base_map = infer_map_from_model(baseline_model, x, device=device, use_lora_branch=False)
        lora_map = infer_map_from_model(trained_model, x, device=device, use_lora_branch=True)

        base_bin = (base_map > threshold).astype(np.uint8)
        lora_bin = (lora_map > threshold).astype(np.uint8)

        rows.append({
            "imagepath": image_path,
            "baseline_predicted_forged_ratio": float(base_bin.mean()),
            "lora_predicted_forged_ratio": float(lora_bin.mean()),
            "baseline_predicted_forged_pixels": int(base_bin.sum()),
            "lora_predicted_forged_pixels": int(lora_bin.sum()),
        })

    return pd.DataFrame(rows)
