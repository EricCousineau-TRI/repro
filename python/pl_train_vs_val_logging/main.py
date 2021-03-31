import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class ConstantMultiply(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.K = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.K * x

    def _step(self, phase, batch, batch_idx):
        K_pre, = self.K.detach().cpu().numpy()
        print(f"{phase}: N={N}, batch_idx={batch_idx}, K_pre={K_pre}")
        N = len(batch)
        x = batch
        yhat = self(x)
        # Goal here is to get parameter to increment once per batch.
        loss = -yhat.mean()
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
def create_datasets(count):
    # These values don't actually value.
    xs = [torch.tensor([0.0]) for i in range(count)]
    dataset = [(x, y) for x, y in zip(xs, ys)]
    return dataset


def main():
    count = 4
    N = 2
    dataset = create_datasets(count)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_val = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ConstantMultiply()
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_dataloader=dataloader, val_dataloaders=dataloader)


if __name__ == "__main__":
    main()
