import torch
from torch import nn, utils
from sklearn.metrics import r2_score
import lightning.pytorch as pl
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class MyDataset(Dataset):
    def __init__(self, is_train):
        self.X = torch.rand((1000,2))
        x1 = self.X[:,0]
        x2 = self.X[:,1]
        self.y = x1 * x2

        self.test_indices = [i for i in range(1000) if i%10 == 0]
        self.train_indices = [i for i in range(1000) if i % 10 != 0]

        indices = self.test_indices
        if is_train:
            indices = self.train_indices

        self.X = self.X[indices]
        self.y = self.y[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MyMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2,5),
            nn.ReLU(),
            nn.Linear(5,1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.machine = MyMachine()
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.machine(x)
        y_hat = y_hat.reshape(-1)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.machine(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.machine(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = LitAutoEncoder()

train_dataset = MyDataset(is_train=True)
test_dataset = MyDataset(is_train=False)

train_set_size = int(len(train_dataset) * 0.8)
valid_set_size = len(train_dataset) - train_set_size

seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)

train_loader = utils.data.DataLoader(train_set)
test_loader = utils.data.DataLoader(test_dataset)
valid_loader = utils.data.DataLoader(valid_set)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=100, callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
trainer.test(model=model, dataloaders=test_loader)

# checkpoint = "lightning_logs/version_0/checkpoints/epoch=9-step=1000.ckpt"
# model = LitAutoEncoder.load_from_checkpoint(checkpoint)
# model.machine.eval()
# trainer = pl.Trainer(limit_train_batches=100, max_epochs=3)
# trainer.test(model=model, dataloaders=test_loader)
