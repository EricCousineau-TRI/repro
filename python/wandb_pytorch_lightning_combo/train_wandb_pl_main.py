#!/usr/bin/env python3

import os
import sys
sys.dont_write_bytecode = True

import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

import wandb_pytorch_lightning_combo.do_pystuck


class ConstantMultiply(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.K = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.K * x

    def _step(self, phase, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = F.mse_loss(yhat, y)
        self.log(f"{phase}/my_loss", loss)
        return loss

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


def main():
    with torch.no_grad():
        K_gt = torch.tensor(2.0, requires_grad=False)
    model = ConstantMultiply()

    count = 8
    dataset_train, dataset_val = create_datasets(K_gt, count)
    batch_size = 4
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    logger = pl.loggers.WandbLogger(
        name="test-run",
        project="uncategorized",
    )

    # # Uncomment this to make all cases work.
    # logger = None

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=5,
        num_processes=2,

        # Freezes.
        accelerator="ddp_cpu",

        # # Freezes.
        # accelerator="ddp_spawn",
        # gpus=[0, 1],

        # # Works.
        # accelerator="ddp",
        # gpus=[0, 1],
    )
    trainer.fit(model, dataloader_train, dataloader_val)

    with torch.no_grad():
        print(f"Final error for K: {model.K - K_gt}", flush=True)


if __name__ == "__main__":
    main()
