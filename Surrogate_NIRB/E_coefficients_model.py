import lightning as L
from sympy import im
import torch
from torch import nn, optim, Tensor, utils
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
import numpy as np
import os
from torchmetrics import (MeanSquaredError ,
                          R2Score,
                          MeanAbsoluteError)
from lightning.pytorch.callbacks import RichProgressBar, EarlyStopping
from torchmetrics.regression import MeanSquaredError as MSE
import time
from helpers import load_pint_data
from pathlib import Path


class BuildingBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(BuildingBlock, self).__init__()

        self.linear = nn.Linear(in_features=in_channels,
                                out_features=out_channels)
        self.activation = nn.ReLU()# nn.ReLU()
        self.dropout = nn.Dropout(0.)

    def forward(self, x):
        return self.dropout(self.activation(self.linear(x)))


class NIRB_NN(nn.Module):
    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        
        self.n_input = n_input
        self.n_output = n_output
        
        
        self.input = nn.Linear(n_input, 8)
        
        sizes = [
                #  (2,  8),
                 (8,  16),
                 (16, 32),
                #  (32, 64),
                #  (64, 32),
                 (32, 16),
                 ]

        self.sequential = nn.Sequential(*[BuildingBlock(in_, out_) \
                                          for in_, out_ in sizes])
        
        self.output = nn.Linear(16, self.n_output)


    def forward(self, x):
        # Flatten all dimensions except the batch dimension
        x  = torch.flatten(x, start_dim=1)
        x = self.input(x)
        return self.output(self.sequential(x))


class NirbModule(L.LightningModule):
    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()
        self.save_hyperparameters()
        

        self.model : torch.nn.Module = NIRB_NN(n_inputs, n_outputs)
        self.learning_rate = 1e-3
        self.loss = nn.MSELoss()
        
        self.msa_metric = MeanAbsoluteError()
        
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)  # loss(input, target)
        metrics = {"train_loss": loss,
                   "train_msa" : self.msa_metric(y_hat, y),
                   }
        self.log_dict(metrics, 
                      prog_bar=True,
                      on_epoch=True,
                      logger=True,
                      sync_dist=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        metrics = {"test_loss": loss,
                   "test_msa" : self.msa_metric(y_hat, y),
                    }
        self.log_dict(metrics)

        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


def create__train_dataloaders(param_path, basis_func_path, snapshot_path) -> DataLoader:

    print(f"Available CPUs: {os.cpu_count()}")  # This will print the number of CPU cores available
    
    n_training = 20
    
    basis_functions         = np.load(basis_func_path)
    training_snapshots      = np.load(snapshot_path)
    training_parameters     = load_pint_data(param_path, is_numpy = True)
    
    # Prepare data
    training_snapshots = training_snapshots[:n_training, -1, :] # last time step
    training_parameters = training_parameters[:n_training, 2:]
    
    
    # Calculate mean/var from training snapshots
    mean = np.mean(training_parameters, axis=0)
    var = np.var(training_parameters, axis=0)
    
    
    # Standardization
    training_parameters    = (training_parameters - mean)/np.sqrt(var)
    
    # Calculate y in NN for training
    training_coeff = np.matmul(basis_functions, training_snapshots.T)
    
    # Check shapes
    print(f"{basis_functions.shape=}")         
    print(f"{training_snapshots.shape=}")      
    print(f"{training_parameters.shape=}")     
    print(f"{training_coeff.shape=}")
    
    dataset_train = TensorDataset(torch.from_numpy(training_parameters.astype(np.float32)),
                                  torch.from_numpy(training_coeff.T.astype(np.float32)))
    dataloader_train = DataLoader(dataset_train,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                # pin_memory=True,
                                # num_workers=N_WORKERS,
                                # persistent_workers=True,
                                )
    return dataloader_train


if __name__ == "__main__":
    seed_everything(42) 
    BATCH_SIZE = 10
    LR = 1e-4
    N_EPOCHS = 20_000
    N_WORKERS = 0
    ROOT = Path.cwd() / "Snapshots" / "01"
    
    param_path = ROOT / "training_samples.csv"
    basis_func_path = ROOT / "BasisFunctions" / "basis_fts_matrix.npy"
    snapshots_path = ROOT / "Exports" / "temperatures.npy"
    
    dataloader_train = create__train_dataloaders(param_path=param_path,
                                                basis_func_path=basis_func_path,
                                                snapshot_path=snapshots_path)
    
    for x, y in dataloader_train:
        n_inputs = x.shape[1]
        n_outputs = y.shape[1]
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        break  # only check the first batch
    
    model = NirbModule(n_inputs, n_outputs)
    model.learning_rate = LR
    
    summary(model.model, 
            input_size=(1, n_inputs),
            col_names=["input_size",
                       "output_size",
                       "num_params"],)
    
    
    early_stop = EarlyStopping(
        monitor="train_loss",        # or "val_loss"
        stopping_threshold=1e-8,      # 💥 stop when loss drops below this
        mode="min",                  # we're minimizing loss
        verbose=True,
        patience=int(0.25*N_EPOCHS),
        check_on_train_epoch_end=True
        )

    logger = TensorBoardLogger(ROOT, name="nn_logs")
    trainer = L.Trainer(max_epochs=N_EPOCHS,
                        min_epochs=int(0.5*N_EPOCHS),
                        logger=logger,
                        log_every_n_steps=BATCH_SIZE*10,  # Reduce logging frequency
                        # enable_checkpointing=True,
                        # callbacks=[early_stop], #, RichProgressBar(refresh_rate=BATCH_SIZE, leave=False)],
                        # precision=16,
                        # max_time={"days": 0, "hours": 0, "minutes": 25},
                        strategy='ddp',
                        enable_progress_bar=False,
                        profiler="simple",
                        devices=3,
                        accelerator= "cpu" #'mps',
                        )
    
    start_time = time.time()
    chkt_path = "/Users/thomassimader/Documents/NIRB/Snapshots/01/nn_logs/version_0/checkpoints/epoch=49999-step=50000.ckpt"
    trainer.fit(model=model,
                train_dataloaders=dataloader_train,
                ckpt_path = None)
    end_time = time.time() 
    print(f"Total training time = {end_time-start_time:.2f} s")
    trainer.save_checkpoint(ROOT / "latest.ckpt")