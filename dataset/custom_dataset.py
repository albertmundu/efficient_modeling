import os
import pyarrow as pa
import torch
from torch.utils.data import Dataset, random_split
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
import io


class CustomDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.transform = transform

        self.files = []
        self.data = os.path.join(data_dir, f'{split}.arrow')

        self.table = pa.ipc.RecordBatchFileReader(
            pa.memory_map(self.data)).read_all()

        self.class_names = {"dog": 0, "cat": 1}
        self.split = split

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        img = self.get_raw_image(idx)
        if self.transform:
            image = self.transform(img)
        if self.split == "train":
            label = self.table["label"][idx].as_py()
            label = torch.tensor(self.class_names[label], dtype=torch.long)
            return image, label
        else:
            img_id = self.table["img_id"][idx].as_py()
            img_id = torch.tensor(img_id, dtype=torch.long)
            return image, img_id

    def get_raw_image(self, idx):
        img = self.table["image"][idx].as_py()

        image_bytes = io.BytesIO(img)
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")
