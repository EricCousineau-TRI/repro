#!/usr/bin/env python3

import argparse
import logging
import os
import warnings

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def rank():
    return rank_zero_only.rank


class ConstantMultiply(pl.LightningModule):
    def __init__(self, change_after_fit_called):
        super().__init__()
        self.K = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("param", torch.zeros((1,), requires_grad=False))
        self.change_after_fit_called = change_after_fit_called
        if not self.change_after_fit_called:
            self.change_non_rank0_values()

    def forward(self, x):
        return self.K * x

    def _step(self, phase, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = F.mse_loss(yhat, y)
        self.log(f"{phase}/my_loss", loss)
        return loss

    def _print_values(self, note):
        print(
            f"rank {rank()}: {note}\n"
            f"  param={self.param.cpu().item()}"
            f"  K={self.K.cpu().item()}",
            flush=True,
        )

    def change_non_rank0_values(self):
        # WARNING: If we have different resutls here, the buffers are snc'd
        if rank() != 0:
            print(f"!!! rank {rank()}: HACK! Making separate param !!!")
            with torch.no_grad():
                self.param[:] = rank_zero_only.rank * 10
                self.K[:] = rank_zero_only.rank * 20

    def on_fit_start(self):
        if self.change_after_fit_called:
            self.change_non_rank0_values()
        self._print_values("on_fit_start")
        print()

    def on_fit_end(self):
        print()
        self._print_values("on_fit_end")

    def on_train_epoch_end(self):
        self._print_values("on_train_epoch_end")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--change-after-fit-called", action="store_true")
    args = parser.parse_args()

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    with torch.no_grad():
        K_gt = torch.tensor(2.0, requires_grad=False)
    model = ConstantMultiply(change_after_fit_called=args.change_after_fit_called)

    count = 8
    dataset_train, dataset_val = create_datasets(K_gt, count)
    batch_size = 4
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    logger = pl.loggers.WandbLogger(
        name="test-run",
        project="test-project",
        mode="disabled",
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=1,
        # accelerator="cpu",
        strategy="ddp",
        devices=2,
        enable_progress_bar=False,
    )
    trainer.fit(model, dataloader_train, dataloader_val)

    with torch.no_grad():
        rz_print(f"Final error for K: {model.K - K_gt}", flush=True)


if __name__ == "__main__":
    main()
