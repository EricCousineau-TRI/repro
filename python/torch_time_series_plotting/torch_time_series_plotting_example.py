#!/usr/bin/env python3

"""
This is a minimalistic example showing a fitting a function against timesieres
data and plotting stuff against it.

This is namely to show:

- Using matplotlib to log timeseries ground-truth and estimates along epochs.
- Using Tensorboard.
- Using PytorchLightning (CPU only, though).
- Using WandB.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb
import wandb

# Parameters.
AMPLITUDE = 5.0
PERIOD_SEC = 2.0
SHIFT_SEC = 0.75
NUM_PERIODS = 1.0
DT = 0.01
VAL_SPLIT_RATIO = 0.2


def sinusoid(period_sec, amplitude, shift_sec, ts):
    omega = 2 * np.pi / period_sec
    return amplitude * torch.sin(omega * (ts + shift_sec))


class Sinusoid(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Set naive initial parameter values.
        self.period_sec = Parameter(torch.tensor(1.0))
        self.amplitude = Parameter(torch.tensor(1.0))
        self.shift_sec = Parameter(torch.tensor(0.0))

    def forward(self, ts):
        return sinusoid(self.period_sec, self.amplitude, self.shift_sec, ts)

    def _step(self, phase, batch, batch_idx):
        ts, ys = batch
        ys_hat = self(ts)
        loss = F.mse_loss(ys_hat, ys)
        self.log(f"{phase}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx)

    def configure_optimizers(self):
        # TODO(eric.cousineau): I was fumbling around. There's prolly a better
        # optimizer.
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer


class PlotTrajectoryComparisonCallback(pl.Callback):
    def __init__(self, writer, datasets, period_in_epochs):
        assert isinstance(writer, tb.SummaryWriter), type(writer)
        assert isinstance(datasets, dict), type(datasets)
        self._writer = writer
        self._datasets = datasets
        self._period_in_epochs = period_in_epochs

    @torch.no_grad()
    def on_epoch_end(self, trainer, model):
        if trainer.current_epoch % self._period_in_epochs != 0:
            return
        was_training = model.training  # Need more elegant way.
        model.train(False)
        step = trainer.global_step
        # Reevaluate
        for mode, dataset in self._datasets.items():
            ts, xs_gt = zip(*dataset)
            ts = torch.tensor(ts).unsqueeze(-1)  # Need to remove this.
            xs_gt = torch.tensor(xs_gt).unsqueeze(-1)  # Need to remove this.
            xs = model(ts)
            fig = plot_trajectory_comparison(ts, xs, xs_gt)
            self._writer.add_figure(f"{mode}/trajectory", fig, step)
        model.train(was_training)


def plot_trajectory_comparison(ts, xs, xs_gt):
    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(9, 3))
    plt.subplots_adjust(wspace=0.3)
    xs_all = np.concatenate((xs, xs_gt))
    x_min = xs_all.min()
    x_max = xs_all.max()

    plt.sca(axs[0])
    plt.plot(ts, xs)
    plt.xlabel("time (s)")
    plt.ylabel("x")
    plt.ylim([x_min, x_max])
    plt.title(f"actual")

    plt.sca(axs[1])
    plt.plot(ts, xs_gt)
    plt.xlabel("time (s)")
    plt.ylabel("x")
    plt.ylim([x_min, x_max])
    plt.title(f"ground-truth")

    plt.sca(axs[2])
    plt.plot(ts, xs - xs_gt)
    plt.xlabel("time (s)")
    plt.ylabel(f"error xs")
    plt.title(f"error (act. - gt.)")
    plt.tight_layout()
    return fig


@torch.no_grad()
def create_datasets():
    # Creates train+val datasets.
    count_per_period = int(np.ceil(PERIOD_SEC / DT))
    count = NUM_PERIODS * count_per_period
    ts = torch.arange(count) * DT
    ts.unsqueeze_(-1)  # Make it batch-friendly.
    ys = sinusoid(PERIOD_SEC, AMPLITUDE, SHIFT_SEC, ts)
    assert ts.shape == ys.shape
    count_val = int(np.ceil(count * VAL_SPLIT_RATIO))
    dataset = [(t, y) for t, y in zip(ts, ys)]
    dataset_train = dataset[:-count_val]
    dataset_val = dataset[-count_val:]
    return dataset_train, dataset_val


def main():
    # N.B. When using `logger.add_figure` / `figure_to_image`, tkinter will
    # complain about not being in the main thread depending on how workers
    # get forked. To simplify, we switch the backend.
    # https://github.com/r9y9/deepvoice3_pytorch/issues/5
    plt.switch_backend("Agg")

    # Just for example.
    out_dir = "/tmp/time_series_example"
    os.environ["WANDB_MODE"] = "dryrun"

    # N.B. wandb.init complains if `{dir}/wandb` doesn't exist, and it goes to
    # a system tmpdir.
    os.makedirs(os.path.join(out_dir, "wandb"), exist_ok=True)
    # Initialize.
    wandb.init(
        project="test-public",
        sync_tensorboard=True,
        dir=out_dir,
    )

    # Create model.
    model = Sinusoid()
    print(model)

    # Create datasets.
    dataset_train, dataset_val = create_datasets()
    # Create loaders.
    batch_size = 64
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=False
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False
    )

    # N.B. Rather than use the WandbLogger (via pl or wandb directly), use
    # tensorboard so that it's easier to view metrics locally. This helps in
    # cases where upload speeds are very slow; it comes at the cost of losing
    # some WandB-specific features.
    log_dir = os.path.join(out_dir, "log")
    logger = pl.loggers.TensorBoardLogger(log_dir)
    # Make simple callback.
    callbacks = [
        PlotTrajectoryComparisonCallback(
            writer=logger.experiment,
            datasets=dict(train=dataset_train, val=dataset_val),
            period_in_epochs=250,
        ),
    ]

    print()
    print(f"To view tensorboard:")
    print(f"  tensorboard --logdir={log_dir}")
    print()

    # Train.
    # N.B. This will be on the CPU.
    trainer = pl.Trainer(max_epochs=1000, callbacks=callbacks, logger=logger)
    trainer.fit(model, dataloader_train, dataloader_val)


if __name__ == "__main__":
    main()
