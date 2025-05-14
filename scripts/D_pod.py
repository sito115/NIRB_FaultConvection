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
from scr.pod import POD, MinMaxNormalizer, MeanNormalizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    PARAMETER_SPACE = "01"
    ROOT = Path(__file__).parents[1]
    DATA_TYPE = "Training"
    ACCURACY = 1e-5
    IS_EXPORT = True
    SUFFIX = "mean_init"
    
    # import_path = ROOT / "data" / PARAMETER_SPACE / "TrainingMapped" / "s100_100_100_b0_4000_0_5000_-4000_-0" / "Exports" / f"{DATA_TYPE}_temperatures.npy"
    import_path = ROOT / "data" / PARAMETER_SPACE / "TrainingMapped" /  f"{DATA_TYPE}_temperatures_minus_tgrad.npy"
    export_folder = import_path.parent.parent.joinpath("BasisFunctions")
    export_folder.mkdir(exist_ok=True)
    assert import_path.exists()
    assert export_folder.exists()
    # temperatures = np.load(ROOT / "Snapshots" / PARAMETER_SPACE / "Exports" / f"{DATA_TYPE}_temperatures.npy")
    temperatures = np.load(import_path)
    
    if PARAMETER_SPACE == "02":
        for idx in [41, 62, 87]:
            temperatures[idx, -1, :] = temperatures[idx, 10, :]

    data_set = temperatures[:, -1, :]
    # assert np.all(data_set > 1)
    # data_set = data_set - temperatures[:, 0, :]
    if "mean" in SUFFIX.lower():
        normalizer = MeanNormalizer()
    elif "min_max" in SUFFIX.lower():
        normalizer = MinMaxNormalizer()
    else:
        raise ValueError("Please check suffix")
    logging.info(f"{normalizer=}")
    data_set_scaled = normalizer.normalize(data_set, keep_scaling_params=True)
    
    pod = POD(POD_snapshots=data_set_scaled, is_time_dependent=False) # is_time_dependent=False bei letztem Zeitschritt 
    basis_fts_matrix, information_content = pod.perform_POD(accuracy=ACCURACY)
    print(np.cumsum(information_content))
    
    if IS_EXPORT:
        np.save(export_folder / (f"information_content_{ACCURACY:.1e}" + SUFFIX + ".npy"), information_content)
        np.save(export_folder / (f"basis_fts_matrix_{ACCURACY:.1e}" + SUFFIX + ".npy"), basis_fts_matrix)
        logging.info(f"Exported {len(basis_fts_matrix)} Basis Functions!")