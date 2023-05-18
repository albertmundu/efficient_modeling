from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torchvision import models, transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from dataset.data_module import CustomDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import csv


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, save_period, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_period = save_period

    def on_epoch_end(self, trainer, pl_module):
        print(trainer.current_epoch)
        if (trainer.current_epoch + 1) % self.save_period == 0:
            super().on_epoch_end(trainer, pl_module)


class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.model = models.resnet50(weights="IMAGENET1K_V2")
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, img_id = batch
        logits = self(x)
        pred = torch.argmax(logits, dim=1)
        logs = {"img_id": img_id, "pred": pred}
        # self.log_dict(logs, on_step=False, on_epoch=True,
        #               prog_bar=True, logger=False, sync_dist=True)

        self._write_csv_logs(logs)

        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _write_csv_logs(self, logs):
        # Check if file exists and write header row if not
        file_exists = os.path.isfile("test_logs.csv")
        with open("test_logs.csv", "a", newline='') as file:
            fieldnames = ["id", "label"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            # Write each prediction to file
            for i, pred in enumerate(logs["pred"]):
                writer.writerow({"id": logs["img_id"][i].cpu().item(),
                                "label": f'{pred}'})


def main():
    # Change these paths to match your dataset locations
    data_root = 'data_parquet'
    pl.seed_everything(42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.PILToTensor(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    data_module = CustomDataModule(
        data_root, batch_size=32, transform=transform)

    model = ImageClassifier()

    # Define checkpoint callback
    checkpoint_callback = CustomModelCheckpoint(
        save_period=5,
        monitor="val_loss",
        dirpath="checkpoints",
        filename="image-classifier-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,  # Save all checkpoints
        save_last=True,  # Save last checkpoint
    )

    trainer = Trainer(devices=2, accelerator="gpu", strategy="ddp", max_epochs=10, precision="16-mixed",
                      callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
