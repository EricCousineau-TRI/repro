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

import pprint as pp
from textwrap import indent
import warnings

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
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
    # These values don't actually matter.
    xs = [torch.tensor([0.0]) for i in range(count)]
    return xs


def main():
    count = 4
    N = 2
    dataset = create_dataset(count)
    dataloader = DataLoader(dataset, batch_size=N, shuffle=False)
    dataloader_val = DataLoader(dataset, batch_size=N, shuffle=False)
    num_batches = len(dataloader)

    log_dir = "/tmp/pl_train_vs_val_logging"
    print(f"log_dir: {log_dir}")

    model = ConstantMultiply()
    logger = TensorBoardLogger(log_dir)

    # Denoise pl
    warnings.filterwarnings(
        "ignore", r".*GPU available but not used.*", UserWarning
    )
    warnings.filterwarnings(
        "ignore", r".*does not have many workers.*", UserWarning
    )
    trainer = pl.Trainer(
        max_epochs=2,
        progress_bar_refresh_rate=0,
        logger=logger,
        flush_logs_every_n_steps=num_batches,
    )

    # TODO(eric.cousineau): This causes training to break on pl==1.2.0, but not
    # 1.2.6.
    #   File ".../pytorch_lightning/core/optimizer.py", line 100, in _to_lightning_optimizer
    #     optimizer = trainer.lightning_optimizers[opt_idx]
    # KeyError: 0
    # pprint_trainer_args(trainer)

    trainer.fit(
        model,
        train_dataloader=dataloader, 
        val_dataloaders=dataloader,
    )


def pprint_trainer_args(trainer):
    ignore = {"progress_bar_dict"}
    out = {}
    for attr in dir(trainer):
        if attr.startswith("_") or attr in ignore:
            continue
        try:
            v = getattr(trainer, attr)
        except AttributeError as e:
            raise RuntimeError((e, attr))
        if is_primitive(v):
            out[attr] = v
    print(pformat(out))


def is_primitive(v):
    if isinstance(v, (bool, int, float, str)):
        return True
    elif isinstance(v, (list, tuple)):
        for vi in v:
            if not is_primitive(vi):
                return False
        return True
    elif isinstance(v, dict):
        for ki, vi in v.items():
            if not is_primitive(ki) or not is_primitive(vi):
                return False
        return True
    else:
        return False
    assert False



def pformat(obj, incr="  "):
    """
    Pretty formatting for values with more vertical whitespace, less hanging
    indents.
    """
    def sub_pformat(obj):
        txt = pformat(obj, incr=incr)
        return indent(txt, incr)
    # Try short version.
    short_len = 60
    maybe_short = pp.pformat(obj)
    if "\n" not in maybe_short and len(maybe_short) <= short_len:
        return maybe_short

    if isinstance(obj, list):
        out = f"[\n"
        for obj_i in obj:
            out += sub_pformat(obj_i) + ",\n"
        out += f"]"
        return out
    elif isinstance(obj, dict):
        out = f"{{\n"
        for k_i, obj_i in obj.items():
            txt = sub_pformat(obj_i)
            out += f"{incr}{repr(k_i)}: {txt.strip()},\n"
        out += f"}}"
        return out
    else:
        return indent(pp.pformat(obj), incr)


if __name__ == "__main__":
    main()
