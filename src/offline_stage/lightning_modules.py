import lightning as L
from lightning.pytorch.callbacks import Callback
from typing import List
from torch import nn
import torch
from .neural_network import NIRB_NN
from src.utils.helpers import R2_metric, Q2_metric
import numpy as np
from torchmetrics import MeanAbsoluteError
import optuna
import warnings

class NirbModule(L.LightningModule):
    def __init__(self, n_inputs: int,
                 hidden_units: List[int],
                 n_outputs: int,
                 activation = nn.Sigmoid(),
                 learning_rate : float = 1e-3,
                 batch_size : int = 20):
        super().__init__()
        self.save_hyperparameters(ignore=['activation'])
        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()
        self.activation = activation
        self.model = NIRB_NN(n_inputs, hidden_units, n_outputs, self.activation)
        self.batch_size = batch_size
        self.msa_metric = MeanAbsoluteError()
        self.test_snaps_scaled : np.ndarray = None
        self.val_snaps_scaled : np.ndarray = None
        self.basis_functions : np.ndarray = None
        self.train_loss : float = None
        self.log_kwargs = {}
        
    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        metrics = {"train_loss": loss}
        self.log_dict(metrics, **self.log_kwargs)
        self.train_loss = loss
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        
        metrics = {"test_loss": loss,
                   "test_msa" : self.msa_metric(y_hat, y),
                    }
        
        if self.test_snaps_scaled is not None and self.basis_functions is not None:
            try:
                full_solution_test = np.matmul(y_hat.detach().numpy(), self.basis_functions)
                q2_metric = Q2_metric(self.test_snaps_scaled, full_solution_test)
                metrics["Q2"] = q2_metric
                metrics["hp_metric"] = q2_metric
            except Exception:
                metrics["Q2"] = -1
        
        self.log_dict(metrics, **self.log_kwargs)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)        
        y_hat_cpu = y_hat.detach().cpu().numpy()
        loss = self.loss(y_hat, y)  # loss(input, target)
        metrics = {"val_loss": loss}
        if self.basis_functions is not None:
            if y_hat_cpu.shape[0] == self.val_snaps_scaled.shape[0]: ## Only compute Q2 if batch size and len validation snaps are the same (only when devices cpu = 1, otherwise it results in an error)
                full_solution_test = np.matmul(y_hat_cpu, self.basis_functions)
                q2_metric = Q2_metric(self.val_snaps_scaled, full_solution_test)
                metrics["Q2_val"] = q2_metric
            else:
                full_solution_test = np.matmul(y_hat_cpu, self.basis_functions)
                validation_snaps_truncated = np.matmul(y.detach().cpu().numpy(), self.basis_functions)
                q2_metric = Q2_metric(validation_snaps_truncated, full_solution_test)
                metrics["Q2_val_trunc"] = q2_metric
        self.log_dict(metrics, **self.log_kwargs)
    
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     return self(batch)
    
    


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



class OptunaPruning(L.pytorch.callbacks.early_stopping.EarlyStopping):
    
    def __init__(self,
                 trial: optuna.Trial,
                 check_val_every_n_steps : int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self._trial = trial
        self.check_val_every_n_steps = check_val_every_n_steps
    
    def _process(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Enable Optuna pruning if one trial seems not promising.

        Args:
            trainer (L.Trainer): _description_
            pl_module (L.LightningModule): _description_

        Raises:
            optuna.TrialPruned: This error tells a trainer that the current ~optuna.trial.Trial was pruned. 
        """        
        current_step = pl_module.global_step
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return
        
        self._trial.report(current_score, step=current_step)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(current_step)
            raise optuna.TrialPruned(message)    
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Only call pruning check when validation occurs
        current_step = pl_module.global_step
        if current_step % self.check_val_every_n_steps == 0:  # Check only on validation epochs
            self._process(trainer, pl_module)


    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass    