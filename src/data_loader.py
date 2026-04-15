"""Data loading utilities for TruFor forgery detection experiments."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ForgeryDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        image_size: Tuple[int, int] = (512, 512),
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = cv2.imread(str(self.image_paths[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0

        mask = cv2.imread(str(self.mask_paths[index]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_size)
        mask = (mask > 128).astype(np.float32)

        tensor_image = torch.from_numpy(image).permute(2, 0, 1)
        tensor_mask = torch.from_numpy(mask).unsqueeze(0)
        return tensor_image, tensor_mask
