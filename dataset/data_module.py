import os
import pyarrow.parquet as pq
import torch
from torch.utils.data import random_split
import pytorch_lightning as pl
from .custom_dataset import CustomDataset


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=32, transform=None, val_split_ratio=0.2):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.transform = transform
        self.val_split_ratio = val_split_ratio

    def setup(self, stage: str):
        if stage == 'fit':
            train_dir = os.path.join(self.data_root, 'train')
            full_train_dataset = CustomDataset(
                train_dir, transform=self.transform)

            train_length = int(len(full_train_dataset) *
                               (1 - self.val_split_ratio))
            val_length = len(full_train_dataset) - train_length

            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, [train_length, val_length])

        if stage == 'test':
            test_dir = os.path.join(self.data_root, 'test')
            self.test_dataset = CustomDataset(
                test_dir, split="test", transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
