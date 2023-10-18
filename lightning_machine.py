from lightning.pytorch import LightningModule
import torch
from torch import optim


class LightningMachine(LightningModule):
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.reshape(-1)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer