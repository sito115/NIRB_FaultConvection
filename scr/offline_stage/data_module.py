import numpy as np
from scr.utils.helpers import standardize, min_max_scaler
from torch.utils.data import TensorDataset, DataLoader
import torch

class NirbDataModule():
    def __init__(self,
                 basis_func_mtrx : np.ndarray,
                 training_snaps: np.ndarray,
                 training_param: np.ndarray,
                 test_snaps: np.ndarray = None,
                 test_param: np.ndarray = None,
                 batch_size: int = 20):
        """Data Module for NIRB.

        Args:
            basis_func_mtrx (np.ndarray): Basis Functions (n_basis x n_points)
            training_snaps (np.ndarray): Training Snapshots (n_snaps x n_points)
            training_param (np.ndarray): Training Parameters (n_snaps x n_parameters)
            test_snaps (np.ndarray, optional): _description_. Defaults to None. (n_snaps x n_points)
            test_param (np.ndarray, optional): _description_. Defaults to None. (n_snaps x n_parameters)
            batch_size (int, optional): _description_. Defaults to 20.
        """
        self.basis_func_mtrx = basis_func_mtrx 
        self.training_snaps = training_snaps 
        self.training_param = training_param 
        self.test_snaps = test_snaps 
        self.test_param = test_param 
        self.batch_size = batch_size 
        
        self.mean = np.mean(self.training_param, axis=0)
        self.var = np.var(self.training_param, axis=0)
        self.compute_coefficients()
        self.setup()

    def compute_coefficients(self) -> None:
        """Calculcates the coefficients (output of NN) for training and test.
        """        
        self.training_param_scaled = standardize(self.training_param, self.mean, self.var)
        self.training_snaps_scaled = min_max_scaler(self.training_snaps)
        self.training_coeff = np.matmul(self.basis_func_mtrx, self.training_snaps_scaled.T).T
        if self.test_param is not None:
            self.test_param_scaled = standardize(self.test_param, self.mean, self.var)
        if self.test_snaps is not None:
            self.test_snaps_scaled = min_max_scaler(self.test_snaps)
            self.test_coeff = np.matmul(self.basis_func_mtrx, self.test_snaps_scaled.T).T

    def setup(self) -> None:
        """Generates TensorDatasets for Training and Test.
        """        
        self.dataset_train = TensorDataset(torch.from_numpy(self.training_param_scaled.astype(np.float32)),
                                           torch.from_numpy(self.training_coeff.astype(np.float32)))
            
        if self.test_snaps is not None:
            self.dataset_test = TensorDataset(torch.from_numpy(self.test_param_scaled.astype(np.float32)),
                                              torch.from_numpy(self.test_coeff.astype(np.float32)))
            

    def train_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_test,
                          batch_size=len(self.dataset_test),  # All in one batch
                          **kwargs)