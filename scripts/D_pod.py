"""
This script performs Proper Orthogonal Decomposition (POD) on the files exported by "C_process_map_export.py".
A key parameter is the ACCURACY variable, which sets the energy threshold, determining the number of basis functions.
The script exports the basis functions, the energy of each basis function, and the minimum and maximum values of the dataset (for min-max scaling).
"""
import numpy as np
from pathlib import Path
import logging
import sys
sys.path.append(str(Path(__file__).parents[1]))
from scr.utils import min_max_scaler
from scr.pod import POD
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
