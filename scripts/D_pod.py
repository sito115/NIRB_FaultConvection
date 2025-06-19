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
from src.pod import POD, MinMaxNormalizer, MeanNormalizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    PARAMETER_SPACE = "09"
    ROOT = Path(__file__).parents[1]
    DATA_TYPE = "Training"
    ACCURACY = 1e-5
    IS_EXPORT = True
    FIELD_NAME = "Temperature"
    SUFFIX = "min_max_init_grad"
    PROJECTION = "Mapped"
    spacing = 50
    control_mesh_suffix = f"s{spacing}_{spacing}_{spacing}_b0_4000_0_5000_-4000_0"
    
    if "init" in SUFFIX.lower():
        import_suffix = "_minus_tgrad"
    else:
        import_suffix = ""
    
    match FIELD_NAME:
        case "Temperature":
            if PROJECTION == "Original":
                import_path = ROOT / "data" / PARAMETER_SPACE / f"Training{PROJECTION}" / f"{DATA_TYPE}_{FIELD_NAME}{import_suffix}.npy" #
            else:
                import_path = ROOT / "data" / PARAMETER_SPACE / f"Training{PROJECTION}" / control_mesh_suffix/ "Exports" / f"{DATA_TYPE}_{FIELD_NAME}{import_suffix}.npy" #
        case "EntropyNum":
            if PROJECTION == "Mapped":
                import_path = ROOT / "data" / PARAMETER_SPACE / f"Training{PROJECTION}" / control_mesh_suffix / "Exports" / "entropy_gen_number.npy"
            else:
                import_path = ROOT / "data" / PARAMETER_SPACE / f"Training{PROJECTION}" / "Training_entropy_gen_per_vol_thermal.npy" #
        case _:
            raise NotImplementedError()
        
    assert import_path.exists(), f"Import path {import_path} does not exist."
    
    export_folder = import_path.parent.parent.joinpath(f"BasisFunctions{FIELD_NAME}")
    export_folder.mkdir(exist_ok=True)
    assert export_folder.exists(), f"Export folder {export_folder} does not exist."
    logging.info(f"{export_folder=}")

    temperatures = np.load(import_path)
    
    if PARAMETER_SPACE == "05":
        temperatures = temperatures[:, np.newaxis, :]
            
    # mask = ~(temperatures == 0).all(axis=(1, 2)) # omit indices that are not computed yet
    # temperatures = temperatures[mask]

    if PARAMETER_SPACE == "02":
        for idx in [41, 62, 87]:
            temperatures[idx, -1, :] = temperatures[idx, 10, :]
        
    data_set = temperatures[:, -1, :]
    logging.info(f"Shape of data set is {data_set.shape}")

    if FIELD_NAME == "Entropy":
        data_set = np.log10(data_set)
        logging.info("Performed Log10 Transformation")
        #TODO: Logarihtmic!

    if "mean" in SUFFIX.lower():
        normalizer = MeanNormalizer()
    elif "min_max" in SUFFIX.lower():
        normalizer = MinMaxNormalizer()
    else:
        raise ValueError("Please check suffix")
    logging.info(f"{normalizer=}")
    data_set_scaled = normalizer.normalize(data_set)
    
    pod = POD(POD_snapshots=data_set_scaled, is_time_dependent=False) # is_time_dependent=False bei letztem Zeitschritt 
    basis_fts_matrix, information_content = pod.perform_POD(accuracy=ACCURACY)
    print(np.cumsum(information_content))
    
    if IS_EXPORT:
        np.save(export_folder / (f"information_content_{ACCURACY:.1e}" + SUFFIX + ".npy"), information_content)
        np.save(export_folder / (f"basis_fts_matrix_{ACCURACY:.1e}" + SUFFIX + ".npy"), basis_fts_matrix)
        logging.info(f"Exported {len(basis_fts_matrix)} Basis Functions!")