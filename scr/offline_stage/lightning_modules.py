import lightning as L
from lightning.pytorch.callbacks import Callback
from typing import List
from torch import nn
import torch
from .neural_network import NIRB_NN
from scr.utils.helpers import R2_metric, Q2_metric
import numpy as np
from torchmetrics import MeanAbsoluteError
import optuna
import warnings

class NirbModule(L.LightningModule):
    def __init__(self, n_inputs: int,
                 hidden_units: List[int],
                 n_outputs: int,
                 activation = nn.Sigmoid(),
                 learning_rate : float = 1e-3):
        super().__init__()
    
        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()
        self.activation = activation
        self.model = NIRB_NN(n_inputs, hidden_units, n_outputs, self.activation)
        
        self.msa_metric = MeanAbsoluteError()
        self.save_hyperparameters()
        self.test_snaps_scaled : np.ndarray = None
        self.basis_functions : np.ndarray = None
        
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)  # loss(input, target)
        metrics = {"train_loss": loss,
                #    "train_msa" : self.msa_metric(y_hat, y),
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
        
        if self.test_snaps_scaled is not None and self.basis_functions is not None:
            full_solution_test = np.matmul(y_hat.detach().numpy(), self.basis_functions)
            q2_metric = Q2_metric(self.test_snaps_scaled, full_solution_test)
            metrics["Q2"] = q2_metric
        
        self.log_dict(metrics)

        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


class ComputeR2OnTrainEnd(Callback):
    """
    Callback to compute and log the R2 score on the training set at the end of training.

    Args:
        - train_parameters_scaled (np.ndarray): 
        - train_snapshots_scaled (np.ndarray): 
        - basis_func_mtrx (np.ndarray): 

    Methods:
        on_train_end(trainer, pl_module): 
            Called at the end of training. Evaluates the model on the training data,
            reconstructs the full field using the basis matrix, computes the R2 metric,
            and logs it using the Lightning's logger.
    """
    def __init__(self,
                 train_parameters_scaled : np.ndarray,
                 train_snapshots_scaled : np.ndarray,
                 basis_func_mtrx: np.ndarray):
        super().__init__()
        
        self.train_parameters_scaled = torch.from_numpy(train_parameters_scaled.astype(np.float32)) 
        self.train_snapshots_scaled = train_snapshots_scaled 
        self.basis_func_mtrx = basis_func_mtrx
        
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        pl_module.eval()
        device = pl_module.device # Get the device (MPS or CPU)
        with torch.no_grad():
            x = self.train_parameters_scaled.to(device)
            y_hat = pl_module(x)
            
        full_solution_train = np.matmul(y_hat.detach().cpu().numpy(), self.basis_func_mtrx)
        r2_metric = R2_metric(full_solution_train, self.train_snapshots_scaled)
        
        if pl_module.logger is not None:
            pl_module.logger.experiment.add_scalar("R2", r2_metric, global_step=trainer.current_epoch)



class MyEarlyStopping(L.pytorch.callbacks.early_stopping.EarlyStopping):
    
    def __init__(self,
                 trial: optuna.Trial,
                 termination_threshold=5.0,
                 min_epochs=5000,
                 **kwargs):
        super().__init__(**kwargs)
        self.termination_threshold = termination_threshold
        self.min_epochs = min_epochs
        self._trial = trial
    
    
    def _process(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        
        current_epoch = pl_module.current_epoch
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return
        
        self._trial.report(current_score, step=current_epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(current_epoch)
            raise optuna.TrialPruned(message)    
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # Check only at the end of training
        return self._process(trainer, pl_module)


    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass    