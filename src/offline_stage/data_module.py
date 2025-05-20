import numpy as np
from src.pod.normalizer import MeanNormalizer, MinMaxNormalizer, Standardizer
from typing import List
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch
from enum import Enum

class Normalizations(Enum):
    MinMax = "MinMax"
    Mean   = "Mean"

class NirbDataModule():
    def __init__(self,
                 basis_func_mtrx : np.ndarray,
                 training_snaps: np.ndarray,
                 training_param: np.ndarray,
                 test_snaps: np.ndarray = None,
                 test_param: np.ndarray = None,
                 val_snaps: np.ndarray = None,
                 val_param: np.ndarray = None,
                 batch_size: int = 20,
                 normalizer : Normalizations = Normalizations.MinMax):
        """Data Module for NIRB.

        Args:
            basis_func_mtrx (np.ndarray): Basis Functions (n_basis x n_points)
            training_snaps (np.ndarray): Training Snapshots (n_snaps x n_points)
            training_param (np.ndarray): Training Parameters (n_snaps x n_parameters)
            test_snaps (np.ndarray, optional): _description_. Defaults to None. (n_snaps x n_points)
            test_param (np.ndarray, optional): _description_. Defaults to None. (n_snaps x n_parameters)
            batch_size (int, optional): _description_. Defaults to 20.
            normalizer (Normalizations): Normalizer for Snapshots. Defaults to MinMaxNormalizer():
        """
        self.basis_func_mtrx = basis_func_mtrx 
        self.training_snaps = training_snaps 
        self.training_param = training_param 
        self.test_snaps = test_snaps 
        self.test_param = test_param 
        self.val_snaps = val_snaps
        self.val_param = val_param
        self.batch_size = batch_size 

        # Initialize the corresponding normalizer class based on the enum value
        if normalizer == Normalizations.MinMax:
            self.normalizer = MinMaxNormalizer()
        elif normalizer == Normalizations.Mean:
            self.normalizer = MeanNormalizer()
        else:
            raise ValueError(f"Unknown normalizer type: {normalizer}")

        
        self.standardizer = Standardizer() # for parameters
        self.compute_coefficients()
        self.setup()

    def compute_coefficients(self) -> None:
        """Calculcates the coefficients (output of NN) for training and test.
        """        
        self.training_param_scaled = self.standardizer.normalize(self.training_param, keep_scaling_params=True)
        self.training_snaps_scaled = self.normalizer.normalize(self.training_snaps, keep_scaling_params=True)
        self.training_coeff = np.matmul(self.basis_func_mtrx, self.training_snaps_scaled.T).T
        if self.test_param is not None:
            self.test_param_scaled = self.standardizer.normalize(self.test_param, keep_scaling_params=False)
        if self.val_param is not None:
            self.val_param_scaled = self.standardizer.normalize(self.val_param, keep_scaling_params=False)
        if self.test_snaps is not None:
            self.test_snaps_scaled = self.normalizer.normalize(self.test_snaps, keep_scaling_params=False)
            self.test_coeff = np.matmul(self.basis_func_mtrx, self.test_snaps_scaled.T).T
        if self.val_snaps is not None:
            self.val_snaps_scaled = self.normalizer.normalize(self.val_snaps, keep_scaling_params=False)
            self.val_coeff = np.matmul(self.basis_func_mtrx, self.val_snaps_scaled.T).T
    def setup(self) -> None:
        """Generates TensorDatasets for Training and Test.
        """        
        self.dataset_train = TensorDataset(torch.from_numpy(self.training_param_scaled.astype(np.float32)),
                                           torch.from_numpy(self.training_coeff.astype(np.float32)))
            
        if self.test_snaps is not None:
            self.dataset_test = TensorDataset(torch.from_numpy(self.test_param_scaled.astype(np.float32)),
                                              torch.from_numpy(self.test_coeff.astype(np.float32)))
            
        if self.val_snaps is not None:
            self.dataset_val = TensorDataset(torch.from_numpy(self.val_param_scaled.astype(np.float32)),
                                             torch.from_numpy(self.val_coeff.astype(np.float32)))
    

    def train_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self.dataset_test,
                          batch_size=len(self.dataset_test),  # All in one batch
                          **kwargs)
        
    def validation_dataloader(self, **kwargs) -> DataLoader:      
        return DataLoader(self.dataset_val,
                          batch_size=len(self.dataset_val), # All in one batch
                          **kwargs)
