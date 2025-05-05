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
sys.path.append(str(Path(__file__).parents[1]))
from scr.utils import min_max_scaler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class POD:
    """This class contains all algorithms for the proper orthogonal
       decomposition (POD)."""
       
    POD_snapshots : np.ndarray 
    
    def perform_POD(self, accuracy: float = 1e-3) -> None:
        """

        Args:
        accuracy = vector of dimension of the number of variables that
                            defines the desired accuracy

        Returns:

        """
        # Perform the Singular Value Decomposition
        logging.info('Selecting the basis functions for the reduced basis...')
        svd = np.linalg.svd(self.POD_snapshots,  full_matrices=False)
        explained_variance  = (svd[1] ** 2) / np.sum(svd[1]**2)
        cumulative_variance = np.cumsum(explained_variance)
        num_components      = np.searchsorted(cumulative_variance, 1 - accuracy) + 1 # +1 to be greater than threshold/accuracy
        information_content = explained_variance[:num_components]
        
        self.information_content = np.copy(information_content)
        self.basis_fts_matrix    = np.copy(svd[2][:num_components, :])
        logging.info(f'Selected {len(self.information_content)} basis function.')

if __name__ == "__main__":
    PARAMETER_SPACE = "03"
    ROOT = Path(__file__).parents[1]
    DATA_TYPE = "Training"
    ACCURACY = 1e-4
    
    import_path = ROOT / "data" / PARAMETER_SPACE / "TrainingMapped" / "s100_100_100_b100_3900_100_4900_-3900_0" / f"{DATA_TYPE}_temperatures.npy"
    export_folder = import_path.parent.joinpath("BasisFunctions")
    export_folder.mkdir(exist_ok=True)
    assert import_path.exists()
    assert export_folder.exists()
    # temperatures = np.load(ROOT / "Snapshots" / PARAMETER_SPACE / "Exports" / f"{DATA_TYPE}_temperatures.npy")
    temperatures = np.load(import_path)
    
    data_set = temperatures[:, -1, :] # last time step
    data_set_scaled = min_max_scaler(data_set)
    
    pod = POD(POD_snapshots=data_set_scaled)
    pod.perform_POD(accuracy=ACCURACY)
    print(pod.basis_fts_matrix)
    print(pod.information_content)
    print(np.cumsum(pod.information_content))
    print(len(pod.information_content))
    
    np.save(export_folder / f"information_content_{ACCURACY:.1e}.npy", pod.information_content)
    np.save(export_folder / f"basis_fts_matrix_{ACCURACY:.1e}.npy", pod.basis_fts_matrix)
    np.save(export_folder / "min_max.npy", np.array([np.min(data_set), np.max(data_set)]))
