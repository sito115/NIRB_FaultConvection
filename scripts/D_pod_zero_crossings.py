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
from src.pod import POD, match_scaler
from src.utils import find_snapshot_path
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    PARAMETER_SPACE = "10"
    ROOT = Path(__file__).parents[1]
    ACCURACY = 1e-6
    IS_EXPORT = True
    FIELD_NAME = "Temperature"
    SUFFIX = "min_max_init_grad" #"min_max"
    PROJECTION = "Mapped" #"Mapped"
    spacing = 100
    control_mesh_suffix = f"s{spacing}_{spacing}_{spacing}_b0_4000_0_5000_-4000_0"
    
    import_path = find_snapshot_path(PROJECTION, SUFFIX, FIELD_NAME, ROOT / "data" / PARAMETER_SPACE, control_mesh_suffix, "Training")

    export_folder = import_path.parent.parent.joinpath(f"BasisFunctions{FIELD_NAME}ZeroCrossings")
    export_folder.mkdir(exist_ok=True)
    assert export_folder.exists(), f"Export folder {export_folder} does not exist."
    logging.info(f"{export_folder=}")

    snapshots = np.load(import_path)
    data_set = snapshots[:, -1, :]
    logging.info(f"Shape of data set is {data_set.shape}")     
    
    zero_crossings = np.load(ROOT / "data" / PARAMETER_SPACE / "Exports" / "Training_zero_crossings.npy")
    assert len(zero_crossings) == len(data_set)
    unique_zc = np.unique(zero_crossings)
    grouped_zc = {int(val): np.where(zero_crossings == val)[0] for val in unique_zc}

    normalizer = match_scaler(SUFFIX)
    logging.info(f"{normalizer=}")

    
    no_convection_idx = grouped_zc.pop(0, [])
    for zc, indices in grouped_zc.items():
        logging.info(f"Processing zero crossing {zc} with {len(indices)} indices.")
        indices_with_0_zc = np.append(indices, no_convection_idx).astype(np.int8)  # Append indices of no convection to each zero crossing group

        sub_set = data_set[indices_with_0_zc]
        if normalizer is not None:
            data_set_scaled  = normalizer.normalize(sub_set)
        else:
            data_set_scaled = sub_set
    
        pod = POD(POD_snapshots=data_set_scaled, is_time_dependent=False) # is_time_dependent=False bei letztem Zeitschritt 
        basis_fts_matrix, information_content = pod.perform_POD(accuracy=ACCURACY)
        print(np.cumsum(information_content))
        
        if IS_EXPORT:
            np.save(export_folder / (f"information_content_{FIELD_NAME}_{ACCURACY:.1e}" + SUFFIX + f"_zc{zc:02d}.npy"), information_content)
            np.save(export_folder / (f"basis_fts_matrix_{FIELD_NAME}_{ACCURACY:.1e}" + SUFFIX + f"_zc{zc:02d}.npy"), basis_fts_matrix)
            logging.info(f"Exported {len(basis_fts_matrix)} Basis Functions!")