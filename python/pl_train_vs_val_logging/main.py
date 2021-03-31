#!/usr/bin/env python3

"""
Logging behavior is still confusing:

mode=Mode.CustomLog
train: N=2, batch_idx=0, batch=[0.0, 1.0], p=1.0
  log[step=0]: {'train/loss': -1.0}
train: N=2, batch_idx=1, batch=[2.0, 3.0], p=2.0
  log[step=1]: {'train/loss': -2.0}
val: N=2, batch_idx=0, batch=[0.0, 1.0], p=3.0
  log[step=1]: {'val/loss': -3.0}
val: N=2, batch_idx=1, batch=[2.0, 3.0], p=3.0
  log[step=1]: {'val/loss': -3.0}

mode=Mode.LogDefault
train: N=2, batch_idx=0, batch=[0.0, 1.0], p=1.0
train: N=2, batch_idx=1, batch=[2.0, 3.0], p=2.0
val: N=2, batch_idx=0, batch=[0.0, 1.0], p=3.0
val: N=2, batch_idx=1, batch=[2.0, 3.0], p=3.0
  log[step=1]: {'val/loss': -3.0, 'epoch': 0}

mode=Mode.LogOnStep
train: N=2, batch_idx=0, batch=[0.0, 1.0], p=1.0
train: N=2, batch_idx=1, batch=[2.0, 3.0], p=2.0
val: N=2, batch_idx=0, batch=[0.0, 1.0], p=3.0
  log[step=0]: {'val/loss/epoch_0': -3.0}
val: N=2, batch_idx=1, batch=[2.0, 3.0], p=3.0
  log[step=1]: {'val/loss/epoch_0': -3.0}

mode=Mode.LogOnEpoch
train: N=2, batch_idx=0, batch=[0.0, 1.0], p=1.0
train: N=2, batch_idx=1, batch=[2.0, 3.0], p=2.0
  log[step=1]: {'train/loss': -1.5, 'epoch': 0}
val: N=2, batch_idx=0, batch=[0.0, 1.0], p=3.0
val: N=2, batch_idx=1, batch=[2.0, 3.0], p=3.0
  log[step=1]: {'val/loss': -3.0, 'epoch': 0}

mode=Mode.LogOnBoth
train: N=2, batch_idx=0, batch=[0.0, 1.0], p=1.0
train: N=2, batch_idx=1, batch=[2.0, 3.0], p=2.0
  log[step=1]: {'train/loss_epoch': -1.5, 'epoch': 0}
val: N=2, batch_idx=0, batch=[0.0, 1.0], p=3.0
  log[step=0]: {'val/loss_step/epoch_0': -3.0}
val: N=2, batch_idx=1, batch=[2.0, 3.0], p=3.0
  log[step=1]: {'val/loss_step/epoch_0': -3.0}
  log[step=1]: {'val/loss_epoch': -3.0, 'epoch': 0}
"""

from enum import Enum
import logging
import warnings

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase
import torch
from torch.utils.data import DataLoader


class Mode(Enum):
    CustomLog = 1
    LogDefault = 2
    LogOnStep = 3
    LogOnEpoch = 4
    LogOnBoth = 5


class ConstantMultiply(pl.LightningModule):
    def __init__(self, mode):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor([1.0]))
        assert mode in Mode
        self.mode = mode

    def forward(self, x):
        return -self.p

    def _step(self, phase, batch, batch_idx):
        N = len(batch)
        p, = self.p.detach().cpu().numpy()
        batch_print = batch.flatten().cpu().numpy().tolist()
        print(f"{phase}: N={N}, batch_idx={batch_idx}, batch={batch_print}, p={p}")
        # Goal here is to get parameter to increment once per batch.
        loss = self(batch).mean()

        key = f"{phase}/loss"
        if self.mode == Mode.CustomLog:
            self.logger.log_metrics(
                {key: loss.detach().cpu().item()},
                step=self.global_step,
            )
        elif self.mode == Mode.LogDefault:
            self.log(key, loss)
        elif self.mode == Mode.LogOnStep:
            self.log(key, loss, on_step=True, on_epoch=False, logger=True)
        elif self.mode == Mode.LogOnEpoch:
            self.log(key, loss, on_step=False, on_epoch=True, logger=True)
        elif self.mode == Mode.LogOnBoth:
            self.log(key, loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx)

    def configure_optimizers(self):
        # Using naive SGD so that we can see parameter increment, hehehhehehehe.
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0)
        return optimizer


class PrintLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()

    @property
    def experiment(self):
        return "lightning_logs"

    def log_metrics(self, metrics, step):
        print(f"  log[step={step}]: {metrics}")

    def log_hyperparams(self, params):
        pass

    @property
    def name(self):
        return self.experiment

    @property
    def version(self):
        return 0


@torch.no_grad()
def create_dataset(count):
    # These values don't actually matter.
    xs = [torch.tensor([float(i)]) for i in range(count)]
    return xs


def ignore_pl_warnings():
    warnings.filterwarnings(
        "ignore", r".*GPU available but not used.*", UserWarning
    )
    warnings.filterwarnings(
        "ignore", r".*does not have many workers.*", UserWarning
    )


def disable_pl_info():
    # Meh.
    pl._logger.setLevel(logging.CRITICAL)


def main():
    count = 4
    dataset = create_dataset(count)
    N = 2
    dataloader = DataLoader(dataset, batch_size=N, shuffle=False)
    dataloader_val = DataLoader(dataset, batch_size=N, shuffle=False)
    num_batches = len(dataloader)

    disable_pl_info()
    ignore_pl_warnings()

    for mode in Mode:
        logger = PrintLogger()
        # Recreate trainer each time.
        trainer = pl.Trainer(
            max_epochs=1,
            progress_bar_refresh_rate=0,
            logger=logger,
            flush_logs_every_n_steps=num_batches,
            weights_summary=None,
            num_sanity_val_steps=0,
        )
        print(f"mode={mode}")
        model = ConstantMultiply(mode)
        trainer.fit(
            model,
            train_dataloader=dataloader, 
            val_dataloaders=dataloader,
        )
        print()


if __name__ == "__main__":
    main()
