#!/usr/bin/env python3

"""
Seems fine in current setup:

    val: N=2, batch_idx=0, K_pre=1.0
    val: N=2, batch_idx=1, K_pre=1.0
    train: N=2, batch_idx=0, K_pre=1.0
    train: N=2, batch_idx=1, K_pre=2.0
    val: N=2, batch_idx=0, K_pre=3.0
    val: N=2, batch_idx=1, K_pre=3.0
    train: N=2, batch_idx=0, K_pre=3.0
    train: N=2, batch_idx=1, K_pre=4.0
    val: N=2, batch_idx=0, K_pre=5.0
    val: N=2, batch_idx=1, K_pre=5.0
"""

import warnings

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class ConstantMultiply(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.K = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return -self.K

    def _step(self, phase, batch, batch_idx):
        K_pre, = self.K.detach().cpu().numpy()
        N = len(batch)
        print(f"{phase}: N={N}, batch_idx={batch_idx}, K_pre={K_pre}")
        # Goal here is to get parameter to increment once per batch.
        loss = self(batch).mean()
        self.log(f"{phase}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx)

    def configure_optimizers(self):
        # Using naive SGD so that we can see parameter increment, hehehhehehehe.
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0)
        return optimizer


@torch.no_grad()
def create_dataset(count):
    # These values don't actually value.
    xs = [torch.tensor([0.0]) for i in range(count)]
    return xs


def main():
    count = 4
    N = 2
    dataset = create_dataset(count)
    dataloader = DataLoader(dataset, batch_size=N, shuffle=False)
    dataloader_val = DataLoader(dataset, batch_size=N, shuffle=False)

    model = ConstantMultiply()
    warnings.simplefilter("ignore", UserWarning)  # Denoise pl
    trainer = pl.Trainer(
        max_epochs=2,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(
        model,
        train_dataloader=dataloader, 
        val_dataloaders=dataloader,
    )


if __name__ == "__main__":
    main()
