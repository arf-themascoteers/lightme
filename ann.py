import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning_machine import LightningMachine


class ANN:
    def __init__(self, model, train_dataset, test_dataset, validation_dataset):
        self.model = LightningMachine(model)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset

        self.train_loader = DataLoader(train_dataset)
        self.test_loader = DataLoader(test_dataset)
        self.valid_loader = DataLoader(validation_dataset)

        self.es_callback = EarlyStopping(monitor="val_loss", mode="min")

        self.mc_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename='best_model',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=True
        )

        self.trainer = pl.Trainer(limit_train_batches=100000, max_epochs=3000, callbacks=[self.mc_callback, self.es_callback])

    def train(self):
        self.trainer.fit(model=self.model, train_dataloaders=self.train_loader, val_dataloaders=self.valid_loader)

    def test(self):
        best_checkpoint_path = self.mc_callback.best_model_path
        best_model = self.model.load_from_checkpoint(best_checkpoint_path)
        self.trainer.test(model=best_model, dataloaders=self.test_loader)
