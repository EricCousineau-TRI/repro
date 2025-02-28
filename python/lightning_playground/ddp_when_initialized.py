#!/usr/bin/env python3

import argparse
import logging
import os
import warnings

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader


def rank():
    return rank_zero_only.rank


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

    def on_fit_start(self):
        print_dist_is_initialized("on_fit_start")

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


def print_dist_is_initialized(note):
    print(f"rank {rank()} {note}: dist.is_initialized() = {dist.is_initialized()}")


def main():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    with torch.no_grad():
        K_gt = torch.tensor(2.0, requires_grad=False)
    model = ConstantMultiply()

    count = 8
    dataset_train, dataset_val = create_datasets(K_gt, count)
    batch_size = 4
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    print_dist_is_initialized("before trainer")
    trainer = pl.Trainer(
        max_epochs=1,
        strategy="ddp",
        devices=2,
        enable_progress_bar=False,
    )
    print_dist_is_initialized("before fit")
    trainer.fit(model, dataloader_train, dataloader_val)

    with torch.no_grad():
        rz_print(f"Final error for K: {model.K - K_gt}", flush=True)


if __name__ == "__main__":
    main()
