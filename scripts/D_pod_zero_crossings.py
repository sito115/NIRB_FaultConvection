"""
This script performs Proper Orthogonal Decomposition (POD) on the files exported by "C_process_map_export.py".
A key parameter is the ACCURACY variable, which sets the energy threshold, determining the number of basis functions.
The script exports the basis functions, the energy of each basis function, and the minimum and maximum values of the dataset (for min-max scaling).
"""
from matplotlib.pylab import f
import numpy as np
from pathlib import Path
import logging
import sys
sys.path.append(str(Path(__file__).parents[1]))
from src.pod import POD, MinMaxNormalizer, MeanNormalizer, Standardizer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    PARAMETER_SPACE = "01"
    ROOT = Path(__file__).parents[1]
    FIELD_NAME = "EntropyNum"
    DATA_TYPE = "Training"
    ACCURACY = 1e-5
    IS_EXPORT = True
    SUFFIX = "mean"
    PROJECTION = "Mapped"
    control_mesh_suffix = "s100_100_100_b0_4000_0_5000_-4000_0"
    
    # import_path = ROOT / "data" / PARAMETER_SPACE / "TrainingMapped" / "s100_100_100_b0_4000_0_5000_-4000_0" / "Exports" / f"{DATA_TYPE}_temperatures_minus_tgrad.npy" #
    # import_path = ROOT / "data" / PARAMETER_SPACE / "TrainingOriginal" / f"{DATA_TYPE}_temperatures_minus_tgrad.npy" #
    if "init" in SUFFIX.lower():
        import_suffix = "_minus_tgrad"
    else:
        import_suffix = ""
        
    match FIELD_NAME:
        case "Temperature":
            import_path = ROOT / "data" / PARAMETER_SPACE / f"Training{PROJECTION}" / f"{DATA_TYPE}_{FIELD_NAME}{import_suffix}.npy" #
        case "EntropyNum":
            if PROJECTION == "Mapped":
                import_path = ROOT / "data" / PARAMETER_SPACE / f"Training{PROJECTION}" / control_mesh_suffix / "Exports" / "entropy_gen_number.npy"
            else:
                import_path = ROOT / "data" / PARAMETER_SPACE / f"Training{PROJECTION}" / "Training_entropy_gen_number_therm.npy" #
        case _:
            raise NotImplementedError()
            

    # import_path = ROOT / "data" / PARAMETER_SPACE / "TrainingMapped" /  f"{DATA_TYPE}_temperatures_minus_tgrad.npy"
    export_folder = import_path.parent.parent.joinpath(f"BasisFunctionsPerZeroCrossing{FIELD_NAME}")
    export_folder.mkdir(exist_ok=True)
    assert import_path.exists(), f"Import path {import_path} does not exist."
    assert export_folder.exists(), f"Export folder {export_folder} does not exist."
    logging.info(f"{export_folder=}")
    # temperatures = np.load(ROOT / "Snapshots" / PARAMETER_SPACE / "Exports" / f"{DATA_TYPE}_temperatures.npy")

    match FIELD_NAME:
        case "Temperature":
            temperatures = np.load(import_path)
            if "init" in SUFFIX.lower() and "grad" in SUFFIX.lower():
                data_set = temperatures[:, -1, :]
            else:
                data_set = temperatures[:, -1, :]
            logging.info(f"Shape of data set is {data_set.shape}")
        case "EntropyNum":
            data_set = np.load(import_path)
            data_set = data_set[:, -1:]
    # mask = ~(temperatures == 0).all(axis=(1, 2)) # omit indices that are not computed yet
    # temperatures = temperatures[mask]
        
    
    zero_crossings = np.load(ROOT / "data" / PARAMETER_SPACE / "Exports" / "Training_zero_crossings.npy")
    assert len(zero_crossings) == len(data_set)
    unique_zc = np.unique(zero_crossings)
    grouped_zc = {int(val): np.where(zero_crossings == val)[0] for val in unique_zc}

    if "mean" in SUFFIX.lower():
        normalizer = MeanNormalizer()
    elif "min_max" in SUFFIX.lower():
        normalizer = MinMaxNormalizer()
    else:
        raise ValueError("Please check suffix")
    logging.info(f"{normalizer=}")
    data_set_scaled = normalizer.normalize(data_set)
    
    no_convection_idx = grouped_zc.pop(0, [])
    for zc, indices in grouped_zc.items():
        logging.info(f"Processing zero crossing {zc} with {len(indices)} indices.")
        indices_with_0_zc = np.append(indices, no_convection_idx)  # Append indices of no convection to each zero crossing group
    
        pod = POD(POD_snapshots=data_set_scaled[indices_with_0_zc], is_time_dependent=False) # is_time_dependent=False bei letztem Zeitschritt 
        basis_fts_matrix, information_content = pod.perform_POD(accuracy=ACCURACY)
        print(np.cumsum(information_content))
        
        if IS_EXPORT:
            np.save(export_folder / (f"information_content_{FIELD_NAME}_{ACCURACY:.1e}" + SUFFIX + f"_zc{zc:d}.npy"), information_content)
            np.save(export_folder / (f"basis_fts_matrix_{FIELD_NAME}_{ACCURACY:.1e}" + SUFFIX + f"_zc{zc:d}.npy"), basis_fts_matrix)
            logging.info(f"Exported {len(basis_fts_matrix)} Basis Functions!")