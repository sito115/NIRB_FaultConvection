from dataclasses import dataclass
import numpy as np
from typing import Tuple
import logging

@dataclass
class POD:
    """This class contains all algorithms for the proper orthogonal
       decomposition (POD)
    
    Args:
        POD_snapshots (n_snaps x n_times x n_points)

    Returns:
        _type_: _description_
    """
       
    POD_snapshots : np.ndarray 
    is_time_dependent : bool = True
    
    
    @staticmethod
    def select_basis_functions(matrix : np.ndarray, accuracy: float) -> Tuple[np.ndarray]:
        svd = np.linalg.svd(matrix,  full_matrices=False)
        explained_variance  = (svd[1] ** 2) / np.sum(svd[1]**2)
        cumulative_variance = np.cumsum(explained_variance)
        num_components      = np.searchsorted(cumulative_variance, 1 - accuracy) + 1 # +1 to be greater than threshold/accuracy
        return svd[2][:num_components, :], explained_variance[:num_components]
    
    
    def perform_POD(self, accuracy: float = 1e-3) -> Tuple[np.ndarray]:
        """

        Args:
        accuracy = vector of dimension of the number of variables that
                            defines the desired accuracy

        Returns: basis_fts_matrix, information_content

        """
        
        if self.is_time_dependent:
            assert len(self.POD_snapshots.shape) == 3, "POD_snapshots must be in (n_snaps, n_time, n_points) format"
            basis_fts_matrix_time = []
            logging.info('Performing the singular value decomposition for the time-trajectory for every parameter')
            for snapshot in self.POD_snapshots:
                basis_fts_matrix, _ = self.select_basis_functions(snapshot, accuracy)
                basis_fts_matrix_time.extend(np.copy(basis_fts_matrix))
            logging.info('Performing the singular value decomposition for the parameters')
            basis_fts_matrix, information_content = self.select_basis_functions(np.asarray(basis_fts_matrix_time), accuracy)
            logging.info(f'Selected {len(information_content)} basis function.')
            return basis_fts_matrix, information_content
        
        else: # stationary - time-independent
            assert len(self.POD_snapshots.shape) == 2, "POD_snapshots must be in (n_snaps, n_points) format"
            logging.info('Selecting the basis functions for the reduced basis...')
            basis_fts_matrix, information_content = self.select_basis_functions(self.POD_snapshots, accuracy)
            logging.info(f'Selected {len(information_content)} basis function.')
            return basis_fts_matrix, information_content

        