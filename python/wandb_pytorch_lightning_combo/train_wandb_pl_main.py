#!/usr/bin/env python3

import os

import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

# begin: https://stackoverflow.com/a/45690594/7829525
import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
# end

_port = find_free_port()
print(f"""
    pystuck port: {_port}
""", flush=True)
import pystuck; pystuck.run_server(port=_port)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_wandb_sweep", action="store_true")
    parser.add_argument("--wandb_sweep_json", type=str, default=None)
    args = parser.parse_args()

    if args.is_wandb_sweep:
        assert args.wandb_sweep_json is not None
        assert args.wandb_sweep_json != ""

    torch.random.manual_seed(0)

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
        log_model=True,
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=5,
        num_processes=2,
        accelerator="ddp_cpu",
        gpus=None,
    )
    trainer.fit(model, dataloader_train, dataloader_val)

    with torch.no_grad():
        print(f"Final error for K: {model.K - K_gt}", flush=True)


if __name__ == "__main__":
    main()
