"""
This script performs Proper Orthogonal Decomposition (POD) on the files exported by "C_process_map_export.py".
A key parameter is the ACCURACY variable, which sets the energy threshold, determining the number of basis functions.
The script exports the basis functions, the energy of each basis function, and the minimum and maximum values of the dataset (for min-max scaling).
"""
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import logging
import sys
from typing import Tuple
sys.path.append(str(Path(__file__).parents[1]))
from scr.utils import min_max_scaler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        Returns:

        """
        
        if self.is_time_dependent:
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
            logging.info('Selecting the basis functions for the reduced basis...')
            basis_fts_matrix, information_content = self.select_basis_functions(self.POD_snapshots, accuracy)
            logging.info(f'Selected {len(information_content)} basis function.')
            return basis_fts_matrix, information_content

        


if __name__ == "__main__":
    PARAMETER_SPACE = "01"
    ROOT = Path(__file__).parents[1]
    DATA_TYPE = "Training"
    ACCURACY = 1e-5
    
    # import_path = ROOT / "data" / PARAMETER_SPACE / "TrainingMapped" / "s100_100_100_b0_4000_0_5000_-4000_-0" / "Exports" / f"{DATA_TYPE}_temperatures.npy"
    import_path = ROOT / "data" / PARAMETER_SPACE / "TrainingMapped" /  f"{DATA_TYPE}_temperatures.npy"
    export_folder = import_path.parent.parent.joinpath("BasisFunctions")
    export_folder.mkdir(exist_ok=True)
    assert import_path.exists()
    assert export_folder.exists()
    # temperatures = np.load(ROOT / "Snapshots" / PARAMETER_SPACE / "Exports" / f"{DATA_TYPE}_temperatures.npy")
    temperatures = np.load(import_path)
    
    data_set = temperatures[:, -1:, :] # last time step
    data_set_scaled = min_max_scaler(data_set)
    
    pod = POD(POD_snapshots=data_set_scaled, is_time_dependent=True)
    basis_fts_matrix, information_content = pod.perform_POD(accuracy=ACCURACY)
    print(information_content)
    print(np.cumsum(information_content))
    
    np.save(export_folder / f"information_content_{ACCURACY:.1e}.npy", information_content)
    np.save(export_folder / f"basis_fts_matrix_{ACCURACY:.1e}.npy", basis_fts_matrix)
    np.save(export_folder / "min_max.npy", np.array([np.min(data_set), np.max(data_set)]))
