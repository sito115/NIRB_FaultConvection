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
from src.utils import find_snapshot_path
from src.pod import POD, match_scaler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    PARAMETER_SPACE = "09"
    ROOT = Path(__file__).parents[1]
    DATA_TYPE = "Training"
    ACCURACY = 1e-5
    IS_EXPORT = True
    FIELD_NAME = "Entropy"
    SUFFIX = "standard" #"min_max"
    PROJECTION = "Mapped" #"Mapped"
    spacing = 50
    control_mesh_suffix = f"s{spacing}_{spacing}_{spacing}_b0_4000_0_5000_-4000_0"
    
    import_path = find_snapshot_path(PROJECTION, SUFFIX, FIELD_NAME, ROOT / "data" / PARAMETER_SPACE, control_mesh_suffix, "Training")

    export_folder = import_path.parent.parent.joinpath(f"BasisFunctions{FIELD_NAME}")
    export_folder.mkdir(exist_ok=True)
    assert export_folder.exists(), f"Export folder {export_folder} does not exist."
    logging.info(f"{export_folder=}")

    snapshots = np.load(import_path)
    
    if PARAMETER_SPACE == "05":
        snapshots = snapshots[:, np.newaxis, :]
            
    # mask = ~(temperatures == 0).all(axis=(1, 2)) # omit indices that are not computed yet
    # temperatures = temperatures[mask]

    if PARAMETER_SPACE == "02":
        for idx in [41, 62, 87]:
            snapshots[idx, -1, :] = snapshots[idx, 10, :]

    if PARAMETER_SPACE == "09":
        zero_crossings = np.load(ROOT / f"data/{PARAMETER_SPACE}/Exports/Training_zero_crossings.npy")
        mask = zero_crossings != 6
        snapshots = snapshots[mask, :, :]
        
    data_set = snapshots[:, -1, :]
    logging.info(f"Shape of data set is {data_set.shape}")


    normalizer = match_scaler(SUFFIX)
    logging.info(f"{normalizer=}")
    if normalizer is not None:
        data_set_scaled = normalizer.normalize(data_set)
    else:
        data_set_scaled = data_set
    
    pod = POD(POD_snapshots=data_set_scaled, is_time_dependent=False) # is_time_dependent=False bei letztem Zeitschritt 
    basis_fts_matrix, information_content = pod.perform_POD(accuracy=ACCURACY)
    print(np.cumsum(information_content))
    
    if IS_EXPORT:
        np.save(export_folder / (f"information_content_{ACCURACY:.1e}" + SUFFIX + ".npy"), information_content)
        np.save(export_folder / (f"basis_fts_matrix_{ACCURACY:.1e}" + SUFFIX + ".npy"), basis_fts_matrix)
        logging.info(f"Exported {len(basis_fts_matrix)} Basis Functions!")