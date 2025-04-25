import numpy as np
from dataclasses import dataclass
from pathlib import Path
import logging
from helpers import min_max_scaler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class POD:
    """This class contains all algorithms for the proper orthogonal
       decomposition (POD)."""
       
    POD_snapshots      : np.ndarray 
    
    def perform_POD(self, accuracy: float = 1e-3) -> None:
        """ Performs the POD for the parameters. This POD algorithm should be
            used for time-independent (steady-state) applications

        Args:
        n_variables = number of variables for which the POD needs to be performed
        threshold_percent = vector of dimension of the number of variables that
                            defines the desired accuracy

        Returns:

        """
        # Perform the Singular Value Decomposition
        logging.info('Selecting the basis functions for the reduced basis...')
        svd = np.linalg.svd(self.POD_snapshots,  full_matrices=False)
        explained_variance  = (svd[1] ** 2) / np.sum(svd[1]**2)
        cumulative_variance = np.cumsum(explained_variance)
        num_components      = np.searchsorted(cumulative_variance, 1 - accuracy) + 1 # +1 to be greater than threshold
        information_content = explained_variance[:num_components]
        
        self.information_content = np.copy(information_content)
        self.basis_fts_matrix    = np.copy(svd[2][:num_components, :])
        logging.info(f'Selected {len(self.information_content)} basis function.')

if __name__ == "__main__":
    VERSION = "02"
    ROOT = Path().cwd()
    DATA_TYPE = "Training" # "Test"
    ACCURACY = 1e-3
    temperatures = np.load(ROOT / "Snapshots" / VERSION / "Exports" / f"{DATA_TYPE}_temperatures.npy")
    
    if VERSION == "02":
        for idx in [41, 62, 87]:
            temperatures[idx, -1, :] = temperatures[idx, 10, :]
    data_set = temperatures[:, -1, :] # last time step
    
    
    data_set_scaled = min_max_scaler(data_set)
    
    pod = POD(POD_snapshots=data_set_scaled)
    pod.perform_POD(accuracy=ACCURACY)
    print(pod.basis_fts_matrix)
    print(pod.information_content)
    print(np.cumsum(pod.information_content))
    print(len(pod.information_content))
    
    np.save(ROOT / "Snapshots" / VERSION / "BasisFunctions" / "information_content.npy", pod.information_content)
    np.save(ROOT / "Snapshots" / VERSION / "BasisFunctions" / "basis_fts_matrix.npy", pod.basis_fts_matrix)
    np.save(ROOT / "Snapshots" / VERSION / "BasisFunctions" / "min_max.npy", np.array([np.min(data_set), np.max(data_set)]))
