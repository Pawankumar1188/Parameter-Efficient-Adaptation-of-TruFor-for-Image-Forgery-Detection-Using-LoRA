import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from sklearn.model_selection import train_test_split


class CocoGlideTruForDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 512):
        self.df = df.reset_index(drop=True)
        self.transform = A.Compose([
            A.Resize(image_size, image_size)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row["imagepath"])
        if image is None:
            raise FileNotFoundError(f"Image not found: {row['imagepath']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if row["label"] == 1:
            mask = cv2.imread(row["maskpath"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {row['maskpath']}")
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        out = self.transform(image=image, mask=mask)
        image = out["image"].astype(np.float32) / 255.0
        mask = (out["mask"] > 127).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))   # HWC -> CHW
        mask = np.expand_dims(mask, axis=0)       # HW -> 1HW

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )


def build_cocoglide_dataframe(data_root: str, test_size: float = 0.15, random_state: int = 42):
    table_path = os.path.join(data_root, "table.csv")
    df_table = pd.read_csv(table_path)

    df_fake = pd.DataFrame({
        "imagepath": df_table["fake"].apply(lambda x: os.path.join(data_root, str(x))),
        "maskpath": df_table["mask"].apply(lambda x: os.path.join(data_root, str(x))),
        "label": 1
    })

    df_real = pd.DataFrame({
        "imagepath": df_table["real"].apply(lambda x: os.path.join(data_root, str(x))),
        "maskpath": None,
        "label": 0
    })

    df_all = pd.concat([df_fake, df_real], ignore_index=True)

    df_all = df_all[df_all["imagepath"].apply(os.path.exists)].reset_index(drop=True)
    df_all = df_all[
        ~(
            (df_all["label"] == 1) &
            (~df_all["maskpath"].apply(lambda x: os.path.exists(x) if isinstance(x, str) else False))
        )
    ].reset_index(drop=True)

    train_df, valid_df = train_test_split(
        df_all,
        test_size=test_size,
        random_state=random_state,
        stratify=df_all["label"]
    )

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def make_subset(df: pd.DataFrame, per_class: int, random_state: int = 42):
    return df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(min(len(x), per_class), random_state=random_state)
    ).reset_index(drop=True)


def build_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    image_size: int = 512,
    batch_size: int = 1,
    num_workers: int = 0
):
    train_dataset = CocoGlideTruForDataset(train_df, image_size=image_size)
    valid_dataset = CocoGlideTruForDataset(valid_df, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader
