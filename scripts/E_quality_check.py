from math import log
import numpy as np
import torch
from pathlib import Path
import sys
import pandas as pd
import pint
import sqlite3
import logging
from tqdm import tqdm
import re
from datetime import datetime
sys.path.append(str(Path(__file__).parents[1]))
from src.offline_stage import NirbModule, NirbDataModule, Normalizations
from comsol_module import COMSOL_VTU
from src.utils import (load_pint_data,
                       setup_logger,
                       safe_parse_quantity,
                       Q2_metric,
                       calculate_thermal_entropy_generation)


def get_n_outputs(trained_model) -> int: 
    last_linear = None
    for layer in trained_model.model.layers:
        if isinstance(layer, torch.nn.Linear):
            last_linear = layer

    if last_linear:
        return last_linear.out_features
    else:
        return -1



def main():

    PARAMETER_SPACE = "01"
    ROOT = Path(__file__).parents[1] / "data" / PARAMETER_SPACE
    assert ROOT.exists()
    ureg = pint.get_application_registry()
    cutoff_datetime = datetime(2025, 5, 30, 14, 15, 0).timestamp()
    PATTERN = r"(\d+\.\d+e[+-]?\d+)(.*)"
    
    chk_pt_paths = sorted([path for path in ROOT.rglob("*.ckpt") if path.stat().st_mtime > cutoff_datetime])


    control_mesh_suffix =  "s100_100_100_b0_4000_0_5000_-4000_0"
    df_basis_functions = pd.DataFrame([
        {'path': str(p), 'shape': m.shape, 'n_basis': m.shape[0], 'n_points': m.shape[1] if m.ndim > 1 else 1}
        for p, m in {
            path: np.load(path) 
            for path in (ROOT / "TrainingMapped" / control_mesh_suffix / "BasisFunctions").rglob('*.npy') if 'basis' in path.stem
        }.items()
    ])
    df_basis_functions['basis_functions'] = df_basis_functions['path'].apply(np.load)
    
    dup_mask = df_basis_functions['n_basis'].duplicated(keep=False)  # Marks all duplicates (not just the later ones)
    if dup_mask.any():
        dup_indices = df_basis_functions[dup_mask].index.tolist()
        logging.warning(f"Duplicates found in column 'n_basis' at indices: {dup_indices}")
        for idx in dup_indices[:-1]:
            logging.warning(f"Deleting {Path(df_basis_functions.loc[idx, 'path']).name}")
        df_basis_functions = df_basis_functions.drop(dup_indices[:-1])


    df_basis_functions['match'] = df_basis_functions['path'].apply(lambda x: re.search(PATTERN, Path(x).stem))
    df_basis_functions['accuracy'] = df_basis_functions['match'].apply(lambda x: float(x.group(1)) if x else np.nan)
    df_basis_functions['suffix'] = df_basis_functions['match'].apply(lambda x: x.group(2)  if x else '')
    logging.debug(f'Loaded {len(df_basis_functions)} different basis function')
    logging.debug(df_basis_functions)
    
    training_parameters         = load_pint_data(ROOT / "training_samples.csv", is_numpy = True)
    test_parameters             = load_pint_data(ROOT / "test_samples.csv", is_numpy = True)

    if PARAMETER_SPACE == "01":
        training_parameters[:, 0] = np.log10(training_parameters[:, 0])
        test_parameters[:, 0] = np.log10(test_parameters[:, 0])

    comsol_data = COMSOL_VTU(ROOT / "TrainingMapped" / control_mesh_suffix /f"Training_000_{control_mesh_suffix}.vtu")
    comsol_data.mesh.clear_data()


    for chk_pt_path in tqdm(chk_pt_paths, total=len(chk_pt_paths)):
        logging.info(chk_pt_path.relative_to(chk_pt_path.parents[2]))
        version = chk_pt_path.parent.parent.stem
        logging.debug(version)

        
        try:
            trained_model : NirbModule = NirbModule.load_from_checkpoint(chk_pt_path)
        except FileNotFoundError as e:
            logging.error(e)
            continue
        
        trained_model = trained_model.to('cpu')
        trained_model.eval()
        
        n_outputs = get_n_outputs(trained_model)
        filtered_basis_df = df_basis_functions.loc[df_basis_functions['n_basis'] == n_outputs]
        if len(filtered_basis_df) == 0:
            logging.error(f'Did not find basis function for {chk_pt_path.name}')
            continue
        assert len(filtered_basis_df) == 1
        
        try:
            ACCURACY = filtered_basis_df.accuracy[0]
            SUFFIX = filtered_basis_df.suffix[0]
            basis_functions = filtered_basis_df.basis_functions[0]
        except KeyError:
            logging.error("Skipped.")
            continue
        
        logging.info(f"Loaded {version}")
        logging.info(f'{ACCURACY=}, {SUFFIX=}')  
        
        conn = sqlite3.connect(ROOT / "results_all.db")
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nirb_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norm TEXT,
            Version TEXT,
            Accuracy REAL,
            Q2_scaled FLOAT,
            Q2_unscaled FLOAT,
            R2_scaled FLOAT,
            R2_unscaled FLOAT,
            Entropy_MSE_test FLOAT,
            Entropy_R2_test FLOAT,
            Entropy_MSE_train FLOAT,
            Entropy_R2_train FLOAT,
            Path TEXT,
            UNIQUE(norm, Version, Accuracy, Path)  -- Required for ON CONFLICT
        )
        ''')
        conn.commit()
        
        if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
            training_snapshots_npy      = np.load(ROOT / "TrainingMapped" / control_mesh_suffix / "Exports" / "Training_temperatures_minus_tgrad.npy")
            test_snapshots_npy          = np.load(ROOT / "TestMapped" / control_mesh_suffix /"Exports" / "Test_temperatures_minus_tgrad.npy")
            training_snapshots  = training_snapshots_npy[:, -1, :]
            test_snapshots      = test_snapshots_npy[:, -1, :]
        elif 'init' in SUFFIX.lower():
            training_snapshots_npy      = np.load(ROOT / "TrainingMapped" / control_mesh_suffix / "Exports" / "Training_temperatures.npy")
            test_snapshots_npy          = np.load(ROOT / "TestMapped" / control_mesh_suffix /"Exports" / "Test_temperatures.npy")
            training_snapshots  = training_snapshots_npy[:, -1, :]
            training_snapshots  = training_snapshots_npy[:, -1, :] -  training_snapshots_npy[:, 0, :] # last time step
            test_snapshots      = test_snapshots_npy[:, -1, :] - test_snapshots_npy[:, 0, :]
        else:
            training_snapshots_npy      = np.load(ROOT / "TrainingMapped" / control_mesh_suffix / "Exports" / "Training_temperatures.npy")
            test_snapshots_npy          = np.load(ROOT / "TestMapped" / control_mesh_suffix /"Exports" / "Test_temperatures.npy")
            training_snapshots  = training_snapshots_npy[:, -1, :]
            test_snapshots      = test_snapshots_npy[:, -1, :]

        if "mean" in SUFFIX.lower():
            scaling = Normalizations.Mean
        elif "min_max" in SUFFIX.lower():
            scaling = Normalizations.MinMax
        else:
            raise ValueError("Invalid suffix.")


        data_module = NirbDataModule(
            basis_func_mtrx=basis_functions,
            training_snaps=training_snapshots,
            test_snaps=test_snapshots,
            training_param=training_parameters,
            test_param=test_parameters,
            normalizer=scaling
        )
        
        param_folder = ROOT / "Exports"
        param_files_test = sorted([file for file in param_folder.rglob("*.csv") if "test" in file.stem.lower()])
        param_files_train = sorted([file for file in param_folder.rglob("*.csv") if "train" in file.stem.lower()])

        # %% Test Predictions
        N = len(test_snapshots)
        samples = np.arange(N)
        predictions_scaled = np.zeros((len(samples), len(comsol_data.mesh.points)))
        predictions_unscaled = np.zeros((len(samples), len(comsol_data.mesh.points)))
        test_solutions_unscaled = np.zeros((len(samples), len(comsol_data.mesh.points)))
        entrpy_nums_test = np.zeros((len(samples)))
        entrpy_nums_prediction = np.zeros((len(samples)))
        for idx, sample_idx in enumerate(samples):
            parameters_df_file = param_files_test[sample_idx]
            param_df = pd.read_csv(parameters_df_file, index_col = 0)
            param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
            lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                                param_df.loc['host_phi', "quantity_pint"] * (4.2 * ureg.watt / (ureg.meter * ureg.kelvin))
            t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
            delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
            param = data_module.test_param_scaled[sample_idx]
            param_t = torch.from_numpy(param.astype(np.float32)).view(1, -1) # shape [1, n_param]
            res = trained_model(param_t)
            res_np = res.detach().numpy()
            my_sol_scaled = np.matmul(res_np.flatten(), basis_functions)
            predictions_scaled[idx, :] = my_sol_scaled
            
            if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                z_points = comsol_data.mesh.points[:, -1] * ureg.meter
                tgrad = (param_df.loc["T_c", "quantity_pint"] - (delta_T / param_df.loc["H", "quantity_pint"] * z_points)).to('K').magnitude
                assert np.all(tgrad > 0)
                test_snap_unscaled = data_module.test_snaps[sample_idx] + tgrad
                prediction_unscaled = data_module.normalizer.inverse_normalize(my_sol_scaled) + tgrad
            elif 'init' in SUFFIX.lower():
                test_snap_unscaled = data_module.test_snaps[sample_idx] + test_snapshots_npy[sample_idx, 0, :]
                prediction_unscaled = data_module.normalizer.inverse_normalize(my_sol_scaled) +  test_snapshots_npy[sample_idx, 0, :]
            else:
                test_snap_unscaled = data_module.test_snaps[sample_idx]
                prediction_unscaled = data_module.normalizer.inverse_normalize(my_sol_scaled)
                
            predictions_unscaled[idx, :] = prediction_unscaled
            test_solutions_unscaled[idx, :] = test_snap_unscaled
            
            _ , entrpy_num_test = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                    test_snap_unscaled,
                                                                    lambda_therm, t0, delta_T)
            
            _, entrpy_num_prediction = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                            prediction_unscaled,
                                                                            lambda_therm, t0, delta_T)
            entrpy_nums_test[idx] = entrpy_num_test.magnitude
            entrpy_nums_prediction[idx] = entrpy_num_prediction.magnitude
            
        
        q2_scaled = Q2_metric(data_module.test_snaps_scaled[samples], predictions_scaled)
        q2_unscaled  =  Q2_metric(test_solutions_unscaled, predictions_unscaled)
        entropy_corr_coeff_test = np.corrcoef(entrpy_nums_prediction, entrpy_nums_test)[0, 1]
        entropy_mse_test = np.mean((entrpy_nums_prediction-entrpy_nums_test)**2)
        
        
        # %% Training Predictions
        predictions_scaled = np.zeros((len(data_module.training_snaps), len(comsol_data.mesh.points)))
        predictions_unscaled = np.zeros_like(predictions_scaled)
        train_solutions_unscaled = np.zeros_like(predictions_scaled)
        entrpy_nums_training = np.zeros((len(data_module.training_snaps)))
        entrpy_nums_prediction = np.zeros((len(data_module.training_snaps)))
        for sample_idx, _ in enumerate(data_module.training_snaps):
            parameters_df_file = param_files_train[sample_idx]
            param_df = pd.read_csv(parameters_df_file, index_col = 0)
            param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
            lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                                param_df.loc['host_phi', "quantity_pint"] * (4.2 * ureg.watt / (ureg.meter * ureg.kelvin))
            t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
            delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
            param = data_module.training_param_scaled[sample_idx]
            param_t = torch.from_numpy(param.astype(np.float32)).view(1, -1) # shape [1, n_param]
            res = trained_model(param_t)
            res_np = res.detach().numpy()
            my_sol_scaled = np.matmul(res_np.flatten(), basis_functions)
            predictions_scaled[idx, :] = my_sol_scaled
            
            if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                z_points = comsol_data.mesh.points[:, -1] * ureg.meter
                tgrad = (param_df.loc["T_c", "quantity_pint"] - (delta_T / param_df.loc["H", "quantity_pint"] * z_points)).to('K').magnitude
                assert np.all(tgrad > 0)
                train_snap_unscaled = data_module.training_snaps[sample_idx] + tgrad
                prediction_unscaled = data_module.normalizer.inverse_normalize(my_sol_scaled) + tgrad
            elif 'init' in SUFFIX.lower():
                train_snap_unscaled = data_module.training_snaps[sample_idx] + training_snapshots_npy[sample_idx, 0, :]
                prediction_unscaled = data_module.normalizer.inverse_normalize(my_sol_scaled) +  training_snapshots_npy[sample_idx, 0, :]
            else:
                train_snap_unscaled = data_module.training_snaps[sample_idx]
                prediction_unscaled = data_module.normalizer.inverse_normalize(my_sol_scaled)
                
            predictions_unscaled[idx, :] = prediction_unscaled
            train_solutions_unscaled[idx, :] = train_snap_unscaled
            
            _ , entrpy_num_test = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                    train_snap_unscaled,
                                                                    lambda_therm, t0, delta_T)
            
            _, entrpy_num_prediction = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                            prediction_unscaled,
                                                                            lambda_therm, t0, delta_T)
            entrpy_nums_training[idx] = entrpy_num_test.magnitude
            entrpy_nums_prediction[idx] = entrpy_num_prediction.magnitude
            
        
        r2_scaled = Q2_metric(data_module.training_snaps_scaled, predictions_scaled)
        r2_unscaled  =  Q2_metric(train_solutions_unscaled, predictions_unscaled)
        entropy_corr_coeff_train = np.corrcoef(entrpy_nums_prediction, entrpy_nums_training)[0, 1]
        entropy_mse_train = np.mean((entrpy_nums_prediction-entrpy_nums_training)**2)
        

        cursor.execute('''
            INSERT INTO nirb_results (
                norm, Q2_scaled, Q2_unscaled, R2_scaled, R2_unscaled,
                Version, Entropy_MSE_test, Entropy_R2_test,
                Entropy_MSE_train, Entropy_R2_train, Accuracy, Path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(norm, Version, Accuracy, Path) DO UPDATE SET
                Q2_scaled = excluded.Q2_scaled,
                Q2_unscaled = excluded.Q2_unscaled,
                R2_scaled = excluded.R2_scaled,
                R2_unscaled = excluded.R2_unscaled,
                Entropy_MSE_test = excluded.Entropy_MSE_test,
                Entropy_R2_test = excluded.Entropy_R2_test,
                Entropy_MSE_train = excluded.Entropy_MSE_train,
                Entropy_R2_train = excluded.Entropy_R2_train,
                Path = excluded.Path,
                Accuracy = excluded.Accuracy
        ''', (
            SUFFIX, q2_scaled, q2_unscaled, r2_scaled, r2_unscaled,
            version, entropy_mse_test, entropy_corr_coeff_test,
            entropy_mse_train, entropy_corr_coeff_train, ACCURACY, chk_pt_path.stem
        ))
        conn.commit()
        conn.close()


if __name__ == "__main__":
    setup_logger(is_console=True, log_file='E_quality.log')
    main()
