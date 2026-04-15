"""Evaluation utilities for forgery localization metrics."""

import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection + eps) / (union + eps)
