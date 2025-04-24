import lightning as L
import torch
from torch import nn
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
from helpers import load_pint_data, min_max_scaler, standardize
from pathlib import Path
import pickle
from typing import List



class NIRB_NN(nn.Module):
    def __init__(self, n_inputs: int,
                 hidden_units: List[int],
                 n_outputs: int,
                 activation = nn.Sigmoid()):
        super().__init__()
        
        all_layers = []
        for hidden_unit in hidden_units:
            layer = nn.Linear(n_inputs, hidden_unit)
            all_layers.append(layer)
            all_layers.append(activation)
            n_inputs = hidden_unit
    
        output_layer = nn.Linear(hidden_units[-1], n_outputs)
        all_layers.append(output_layer) 
        
        self.layers = nn.Sequential(*all_layers)        


    def forward(self, x):
        # Flatten all dimensions except the batch dimension
        x  = torch.flatten(x, start_dim=1)
        logits = self.layers(x)
        return logits


class NirbDataModule(L.LightningDataModule):
    def __init__(self,
                 basis_func_mtrx : np.ndarray,
                 training_snaps: np.ndarray,
                 training_param: np.ndarray,
                 test_snaps: np.ndarray = None,
                 test_param: np.ndarray = None,
                 batch_size: int = 20):
        super().__init__()
        self.basis_func_mtrx = basis_func_mtrx 
        self.training_snaps = training_snaps 
        self.training_param = training_param 
        self.test_snaps = test_snaps 
        self.test_param = test_param 
        self.batch_size = batch_size 
        
        self.mean = np.mean(self.training_param, axis=0)
        self.var = np.var(self.training_param, axis=0)

    def prepare_data(self) -> None:
        self.training_param = standardize(self.training_param, self.mean, self.var)
        self.training_snaps = min_max_scaler(self.training_snaps)
        self.training_coeff = np.matmul(self.basis_func_mtrx, self.training_snaps.T)
        if self.test_param is not None:
            self.test_param = standardize(self.test_param, self.mean, self.var)
        if self.test_snaps is not None:
            self.test_snaps = min_max_scaler(self.test_snaps)
            self.test_coeff = np.matmul(self.basis_func_mtrx, self.test_snaps.T)

    def setup(self, stage:str):
        if stage == "fit":
            self.dataset_train = TensorDataset(torch.from_numpy(self.training_param.astype(np.float32)),
                                               torch.from_numpy(self.training_coeff.T.astype(np.float32)))
            
        if stage == "test":
            self.dataset_test = TensorDataset(torch.from_numpy(self.test_param.astype(np.float32)),
                                               torch.from_numpy(self.test_coeff.T.astype(np.float32)))
            

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          shuffle=True)



class NirbModule(L.LightningModule):
    def __init__(self, n_inputs: int,
                 hidden_units: List[int],
                 n_outputs: int,
                 activation = nn.Sigmoid()):
        super().__init__()
    
        self.learning_rate = 1e-3
        self.loss = nn.MSELoss()
        self.activation = activation
        self.model = NIRB_NN(n_inputs, hidden_units, n_outputs, self.activation)
        
        self.msa_metric = MeanAbsoluteError()
        self.save_hyperparameters()
        
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



if __name__ == "__main__":
    seed_everything(42) 
    BATCH_SIZE = 20
    LR = 5e-4
    N_EPOCHS = 300_000
    N_WORKERS = 0
    ROOT = Path.cwd() / "Snapshots" / "01"
    # DATA_TYPE = "Trai"
    

    basis_func_path = ROOT / "BasisFunctions" / "basis_fts_matrix.npy"
    train_snapshots_path = ROOT / "Exports" / "Training_temperatures.npy"
    test_snapshots_path = ROOT / "Exports" / "Test_temperatures.npy"
    train_param_path = ROOT / "training_samples.csv"
    test_param_path = ROOT / "test_samples.csv"
    
    
    basis_functions         = np.load(basis_func_path)
    training_snapshots      = np.load(train_snapshots_path)
    training_parameters     = load_pint_data(train_param_path, is_numpy = True)
    test_snapshots          = np.load(test_snapshots_path)
    test_parameters         = load_pint_data(test_param_path, is_numpy = True)
    
    # Prepare data
    training_snapshots = training_snapshots[:, -1, :] # last time step
    training_parameters = training_parameters[:, 2:] 
    test_snapshots = test_snapshots[:, -1, :] # last time step
    test_parameters = test_parameters[:, 2:] 
     
     
    dm = NirbDataModule(
        basis_func_mtrx=basis_functions,
        training_snaps=training_snapshots,
        training_param=training_parameters,
        test_param=training_parameters,
        test_snaps=training_snapshots,
        batch_size=20,
    )
    
    # dataloader_train = create__train_dataloaders(param_path=param_path,
    #                                             basis_func_path=basis_func_path,
    #                                             snapshot_path=snapshots_path)
    

    n_inputs = training_parameters.shape[1]
    n_outputs = basis_functions.shape[0]

    
    model = NirbModule(n_inputs, [6, 16, 32, 64, 32], n_outputs)
    model.learning_rate = LR
    
    summary(model.model, 
            input_size=training_parameters.shape,
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
                        logger=None,
                        log_every_n_steps=BATCH_SIZE*10,  # Reduce logging frequency
                        # enable_checkpointing=True,
                        # callbacks=[early_stop], #, RichProgressBar(refresh_rate=BATCH_SIZE, leave=False)],
                        # precision=16,
                        # max_time={"days": 0, "hours": 0, "minutes": 25},
                        strategy='ddp',
                        enable_progress_bar=False,
                        profiler="simple",
                        devices=1,
                        accelerator= "cpu" #'mps',
                        )
    
    
    start_time = time.time()
    ckpt_path = "/Users/thomassimader/Documents/NIRB/Snapshots/01/nn_logs/version_27/checkpoints/epoch=99999-step=200000.ckpt"
    trainer.fit(model=model,
                datamodule=dm,
                ckpt_path = None)
    trainer.test(model=model,
                 datamodule=dm)
    
    
    # end_time = time.time() 
    # print(f"Total training time = {end_time-start_time:.2f} s")
    # with open(Path(logger.log_dir) / "LightningModule.pkl", "wb") as f:
    #     pickle.dump(model, f)
    # with open(Path(logger.log_dir) / "NNModel.pkl", "wb") as f:
    #     pickle.dump(NIRB_NN(n_inputs, n_outputs), f)
    # trainer.save_checkpoint(ROOT / "latest.ckpt")