#!/usr/bin/env python3

"""
Goal: Is it possible to use separate source of randomness for:
- weight initialization
- data shuffling
- augmentation (data loader)
"""

import os

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ConstantMultiply(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # should initialize this
        self.K = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.K * x

    def _step(self, phase, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = F.mse_loss(yhat, y)
        self.log(f"{phase}/my_loss", loss)
        return loss

    def on_before_batch_transfer(self, batch, dataloader_idx):
        rz_print(f"\n{self.trainer.training} {self.trainer.validating}\n", flush=True)
        return batch

    def on_validation_epoch_end(self):
        breakpoint()

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer


@torch.no_grad()
def create_datasets(K, count):
    xs = [torch.tensor([i]) for i in range(count * 2)]
    ys = [K * x for x in xs]
    dataset = [(x, y) for x, y in zip(xs, ys)]
    dataset_train = dataset[:count]
    dataset_val = dataset[count:]
    return dataset_train, dataset_val


@rank_zero_only
def rz_print(*args, **kwargs):
    print(*args, **kwargs)


def worker_init_fn():
    ...


def main():
    with torch.no_grad():
        K_gt = torch.tensor(2.0, requires_grad=False)
    model = ConstantMultiply()

    count = 8
    dataset_train, dataset_val = create_datasets(K_gt, count)
    batch_size = 2
    dataloader_train = DataLoader(
        dataset_train,
        num_workers=2,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
    )
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    logger = pl.loggers.WandbLogger(
        name="test-run",
        project="test-project",
        mode="disabled",
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=5,
        accelerator="cpu",
        # strategy="ddp",
        # devices=2,
    )
    trainer.fit(model, dataloader_train, dataloader_val)

    with torch.no_grad():
        rz_print(f"Final error for K: {model.K - K_gt}", flush=True)


if __name__ == "__main__":
    main()
