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
from src.offline_stage import NirbModule, NirbDataModule, get_n_outputs
from src.pod import MinMaxNormalizer, MeanNormalizer, Standardizer, Normalizer, match_scaler
from comsol_module import COMSOL_VTU
from src.utils import (load_pint_data,
                       setup_logger,
                       safe_parse_quantity,
                       Q2_metric,
                       calculate_thermal_entropy_generation,
                       find_snapshot_path,
                       read_config)


    
# Create dummy enum with string values for safe unpickling
torch.serialization.add_safe_globals([torch.nn.modules.activation.Tanh,
                                        torch.nn.modules.activation.Sigmoid,
                                        torch.nn.modules.activation.LeakyReLU,
                                        torch.nn.modules.activation.ReLU,
                                        MinMaxNormalizer,
                                        MeanNormalizer,
                                        Normalizer,
                                        np.dtypes.Float64DType,
                                        np.float64,
                                        np.int64,
                                        np._core.multiarray.scalar,
                                        np.dtype,
                                        Standardizer])

def main():

    PARAMETER_SPACE = "10"
    ROOT = Path(__file__).parents[1] / "data" / PARAMETER_SPACE
    assert ROOT.exists()
    ureg = pint.get_application_registry()
    cutoff_datetime = datetime(2025, 6, 5, 15, 0, 0).timestamp()
    PATTERN = r"(\d+\.\d+e[+-]?\d+)(.*)"
    IS_OVERWRITE = False
    PROJECTION = "Mapped"  # "Mapped" or "Original"
    FIELD_NAME = "Entropy" #"Entropy"
    is_clean_mesh = False
    spacing = 100
    control_mesh_suffix = f"s{spacing}_{spacing}_{spacing}_b0_4000_0_5000_-4000_0"
    config = read_config()

    
    chk_pt_paths = sorted([path for path in ROOT.rglob("*.ckpt") if path.stat().st_mtime > cutoff_datetime])
    chk_pt_paths = [s for s in chk_pt_paths if 'zc' not in str(s)]
    if FIELD_NAME == "Temperature":
        chk_pt_paths = [s for s in chk_pt_paths if 'entropy' not in str(s).lower()]
    elif FIELD_NAME == "Entropy":
        chk_pt_paths = [s for s in chk_pt_paths if FIELD_NAME.lower() in str(s).lower()]

    # chk_pt_paths= [Path("/Users/thomassimader/Documents/NIRB/data/09/optuna_logsTemperature/trial_0_20250619-120700/checkpoints/last.ckpt")]

    # assert len(chk_pt_paths) > 0
    
    
    basis_functions_folder = ROOT / f"Training{PROJECTION}" / control_mesh_suffix / f"BasisFunctions{FIELD_NAME}" if control_mesh_suffix else ROOT / f"BasisFunctions{FIELD_NAME}"
    basis_functions_folder.exists()
    
    df_basis_functions = pd.DataFrame([
        {'path': str(p), 'shape': m.shape, 'n_basis': m.shape[0], 'n_points': m.shape[1] if m.ndim > 1 else 1}
        for p, m in {
            path: np.load(path) 
            for path in basis_functions_folder.rglob('*.npy') if 'basis_fts' in path.stem
        }.items()
    ])
    df_basis_functions['basis_functions'] = df_basis_functions['path'].apply(np.load)
    
    dup_mask = df_basis_functions['n_basis'].duplicated(keep=False)  # Marks all duplicates (not just the later ones)
    if dup_mask.any():
        dup_indices = df_basis_functions[dup_mask].index.tolist()
        logging.warning(f"Duplicates found in column 'n_basis' at indices: {dup_indices}")
        for dub_idx in dup_indices[:-1]:
            logging.warning(f"Deleting {Path(df_basis_functions.loc[dub_idx, 'path']).name}")
        df_basis_functions = df_basis_functions.drop(dup_indices[:-1])


    df_basis_functions['match'] = df_basis_functions['path'].apply(lambda x: re.search(PATTERN, Path(x).stem))
    df_basis_functions['accuracy'] = df_basis_functions['match'].apply(lambda x: float(x.group(1)) if x else np.nan)
    df_basis_functions['suffix'] = df_basis_functions['match'].apply(lambda x: x.group(2)  if x else '')
    df_basis_functions.to_csv(ROOT / f"df_basis_functions_{FIELD_NAME}.csv")
    logging.debug(f'Loaded {len(df_basis_functions)} different basis function')
    logging.debug(df_basis_functions)
    
    training_parameters         = load_pint_data(ROOT / "training_samples.csv", is_numpy = True)
    test_parameters             = load_pint_data(ROOT / "test_samples.csv", is_numpy = True)

    if PARAMETER_SPACE == "01":
        training_parameters[:, 0] = np.log10(training_parameters[:, 0])
        test_parameters[:, 0] = np.log10(test_parameters[:, 0])

    if control_mesh_suffix is not None:
        comsol_data = COMSOL_VTU(ROOT / f"Training{PROJECTION}" / control_mesh_suffix /f"Training_000_{control_mesh_suffix}.vtk", is_clean_mesh=is_clean_mesh)
    else:
        comsol_data = COMSOL_VTU(ROOT / "TrainingOriginal" / "Training_000.vtu", is_clean_mesh=is_clean_mesh)
    comsol_data.mesh.clear_data()

    for chk_pt_path in tqdm(chk_pt_paths, total=len(chk_pt_paths)):
        logging.info(chk_pt_path.relative_to(chk_pt_path.parents[2]))
        version = chk_pt_path.parent.parent.stem
        logging.debug(f'Selected {version}')


        try:
            trained_model : NirbModule = NirbModule.load_from_checkpoint(chk_pt_path)
        except (FileNotFoundError, ValueError, AttributeError) as e:
            logging.error(e)
            continue
        
        trained_model = trained_model.to('cpu')
        trained_model.eval()
        checkpoint = torch.load(chk_pt_path, map_location='cpu')
        
        
        epoch = checkpoint.get('epoch', None)
        global_step = checkpoint.get('global_step', None)
        scaler_features = checkpoint['hyper_parameters'].get('scaler_features', 'Standardizer')
        logging.info(f"{scaler_features=}")
        
        n_outputs = get_n_outputs(trained_model)
        filtered_basis_df = df_basis_functions.loc[df_basis_functions['n_basis'] == n_outputs]
        if len(filtered_basis_df) == 0:
            logging.error(f'Did not find basis function for {chk_pt_path.name}')
            continue
        assert len(filtered_basis_df) == 1
        
        try:
            ACCURACY = filtered_basis_df.accuracy.values[0]
            SUFFIX = filtered_basis_df.suffix.values[0]
            basis_functions = filtered_basis_df.basis_functions.values[0]
        except KeyError:
            logging.error("Skipped.")
            continue
        
        logging.info(f"Loaded {version}")
        logging.info(f'{ACCURACY=}, {SUFFIX=}')  
        
        conn = sqlite3.connect(ROOT / f"results_all{FIELD_NAME}.db")
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nirb_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norm TEXT,
            Version TEXT,
            feature_scaling TEXT,
            Accuracy REAL,
            Q2_scaled FLOAT,
            Q2_unscaled FLOAT,
            R2_scaled FLOAT,
            R2_unscaled FLOAT,
            Epoch INTEGER,
            Global_step INTEGER,
            Entropy_MSE_test FLOAT,
            Entropy_R2_test FLOAT,
            Entropy_MSE_train FLOAT,
            Entropy_R2_train FLOAT,
            Path TEXT,
            Path_mtime TEXT,
            UNIQUE(norm, Version, Accuracy, Path)  -- Required for ON CONFLICT
        )
        ''')
        conn.commit()
        
        
        cursor.execute('''
        SELECT id FROM nirb_results
        WHERE norm = ? AND Version = ? AND Accuracy = ? AND Path = ?
        ''', (SUFFIX, version, ACCURACY, str(chk_pt_path)))

        result = cursor.fetchone()
        
        if result is not None and not IS_OVERWRITE:
            logging.info(f'Skipped {chk_pt_path.name}: Already in database')
            continue
    
        # last time step
        training_snapshots = np.load(find_snapshot_path(PROJECTION, SUFFIX, FIELD_NAME, ROOT, control_mesh_suffix, "Training"))[:, -1, :]
        test_snapshots     = np.load(find_snapshot_path(PROJECTION, SUFFIX, FIELD_NAME, ROOT, control_mesh_suffix, "Test"))[:, -1, :]

        scaling_output = match_scaler(SUFFIX)
        scaling_features = match_scaler(scaler_features)
        
        param_folder = ROOT / "Exports"
        param_files_test  = np.array(sorted([file for file in param_folder.rglob("*.csv") if "test" in file.stem.lower()]))
        param_files_train = np.array(sorted([file for file in param_folder.rglob("*.csv") if "train" in file.stem.lower()]))
        
        target_date = datetime(2025, 6, 25).timestamp()
        if PARAMETER_SPACE == "09" and FIELD_NAME == "Entropy" and chk_pt_path.stat().st_mtime > target_date:
            zero_crossings = np.load(ROOT / "Exports/Training_zero_crossings.npy")
            mask = zero_crossings != 6
        else:
            mask = np.ones(len(training_snapshots), dtype=bool)
        
        data_module = NirbDataModule(
            basis_func_mtrx=basis_functions,
            training_snaps=training_snapshots[mask, :],
            test_snaps=test_snapshots,
            training_param=training_parameters[mask, :],
            test_param=test_parameters,
            normalizer=scaling_output,
            standardizer_features=scaling_features,
            batch_size = -1,
        )
        
        param_files_train = param_files_train[mask]
        assert len(param_files_test) == len(data_module.test_snaps)
        assert len(param_files_train) == len(data_module.training_snaps)

        # %% Test Predictions
        N = len(test_snapshots)
        predictions_scaled = np.zeros((N, basis_functions.shape[1]))
        predictions_unscaled = np.zeros((N, basis_functions.shape[1]))
        test_solutions_unscaled = np.zeros((N,basis_functions.shape[1]))
        entrpy_nums_test = np.zeros((N,))
        entrpy_nums_prediction = np.zeros((N,))
        for test_idx in np.arange(N):
            parameters_df_file = param_files_test[test_idx]
            param_df = pd.read_csv(parameters_df_file, index_col = 0)
            param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
            lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                                param_df.loc['host_phi', "quantity_pint"] * (config['lambda_f']  * ureg.watt / (ureg.meter * ureg.kelvin))
            t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
            delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
            param = data_module.test_param_scaled[test_idx]
            param_t = torch.from_numpy(param.astype(np.float32)).view(1, -1) # shape [1, n_param]
            res_np = trained_model(param_t).detach().numpy()
            logging.debug(f"{res_np=}")
            my_sol_scaled = np.matmul(res_np.flatten(), basis_functions)
            predictions_scaled[test_idx, :] = my_sol_scaled
            
            if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                z_points = comsol_data.mesh.points[:, -1] * ureg.meter
                tgrad = (param_df.loc["T_c", "quantity_pint"] - (delta_T / param_df.loc["H", "quantity_pint"] * z_points)).to('K').magnitude
                assert np.all(tgrad > 0)
                test_snap_unscaled = data_module.test_snaps[test_idx] + tgrad
                prediction_unscaled = data_module.normalizer.inverse_normalize(my_sol_scaled) + tgrad
            else:
                test_snap_unscaled = data_module.test_snaps[test_idx]
                if data_module.normalizer is None:
                    prediction_unscaled = my_sol_scaled
                else:
                    prediction_unscaled = data_module.normalizer.inverse_normalize(my_sol_scaled)
                
                if FIELD_NAME == "Entropy":
                    prediction_unscaled = np.power(10, prediction_unscaled)
                    test_snap_unscaled = np.power(10, test_snap_unscaled)

            predictions_unscaled[test_idx, :] = prediction_unscaled
            test_solutions_unscaled[test_idx, :] = test_snap_unscaled
            
            match FIELD_NAME:
                case "Temperature":
                    _ , entrpy_num_test = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                            test_snap_unscaled,
                                                                            lambda_therm, t0, delta_T)
                    
                    _, entrpy_num_prediction = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                                    prediction_unscaled,
                                                                                    lambda_therm, t0, delta_T)
                    
                case "Entropy":
                    comsol_data.mesh.point_data['temp_pred'] = prediction_unscaled
                    comsol_data.mesh.point_data['temp_test'] = test_snap_unscaled
                    integrated = comsol_data.mesh.integrate_data()
                    s0_total_pred = integrated.point_data['temp_pred'][0] * ureg.watt / ureg.kelvin
                    s0_total_test = integrated.point_data['temp_test'][0] * ureg.watt / ureg.kelvin
                    
                    L = param_df.loc["H", "quantity_pint"]
                    
                    s0_characteristic = (lambda_therm * delta_T**2) / (L**2 * t0**2)
                    entrpy_num_test = s0_total_test / s0_characteristic / (comsol_data.mesh.volume * ureg.meter**3)  
                    entrpy_num_prediction = s0_total_pred / s0_characteristic / (comsol_data.mesh.volume * ureg.meter**3)  
                    assert entrpy_num_test.check(['dimensionless']) # check for correct unit
                    assert entrpy_num_prediction.check(['dimensionless']) # check for correct unit
                    
            entrpy_nums_test[test_idx] = entrpy_num_test.magnitude
            entrpy_nums_prediction[test_idx] = entrpy_num_prediction.magnitude
            
        
        assert np.all(predictions_scaled != 0), " test predictions_scaled contains zero(s)"
        assert np.all(predictions_unscaled != 0), "test predictions_unscaled contains zero(s)"
        assert np.all(test_solutions_unscaled != 0), "train_solutions_unscaled contains zero(s)"
        
        q2_scaled = Q2_metric(data_module.test_snaps_scaled, predictions_scaled)
        q2_unscaled  =  Q2_metric(test_solutions_unscaled, predictions_unscaled)
        entropy_corr_coeff_test = np.corrcoef(entrpy_nums_prediction, entrpy_nums_test)[0, 1]
        entropy_mse_test = np.mean((entrpy_nums_prediction-entrpy_nums_test)**2)
        
        
        # %% Training Predictions
        predictions_scaled = np.zeros((len(data_module.training_snaps), basis_functions.shape[1]))
        predictions_unscaled = np.zeros_like(predictions_scaled)
        train_solutions_unscaled = np.zeros_like(predictions_scaled)
        entrpy_nums_training = np.zeros((len(data_module.training_snaps)))
        entrpy_nums_prediction = np.zeros((len(data_module.training_snaps)))
        for train_idx, _ in enumerate(data_module.training_snaps):
            parameters_df_file = param_files_train[train_idx]
            param_df = pd.read_csv(parameters_df_file, index_col = 0)
            param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
            lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                                param_df.loc['host_phi', "quantity_pint"] * (config['lambda_f']  * ureg.watt / (ureg.meter * ureg.kelvin))
            t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
            delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
            param = data_module.training_param_scaled[train_idx]
            param_t = torch.from_numpy(param.astype(np.float32)).view(1, -1) # shape [1, n_param]
            res_np = trained_model(param_t).detach().numpy()
            my_sol_scaled = np.matmul(res_np.flatten(), basis_functions)
            predictions_scaled[train_idx, :] = my_sol_scaled
            
            if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                z_points = comsol_data.mesh.points[:, -1] * ureg.meter
                tgrad = (param_df.loc["T_c", "quantity_pint"] - (delta_T / param_df.loc["H", "quantity_pint"] * z_points)).to('K').magnitude
                assert np.all(tgrad > 0)
                train_snap_unscaled = data_module.training_snaps[train_idx] + tgrad
                prediction_unscaled = data_module.normalizer.inverse_normalize(my_sol_scaled) + tgrad
            else:
                train_snap_unscaled = data_module.training_snaps[train_idx]
                if data_module.normalizer is None:
                    prediction_unscaled = my_sol_scaled
                else:
                    prediction_unscaled = data_module.normalizer.inverse_normalize(my_sol_scaled)
                
                if FIELD_NAME == "Entropy":
                    prediction_unscaled = np.power(10, prediction_unscaled)
                    train_snap_unscaled = np.power(10, train_snap_unscaled)
                    
            predictions_unscaled[train_idx, :] = prediction_unscaled
            train_solutions_unscaled[train_idx, :] = train_snap_unscaled
            
            match FIELD_NAME:
                case "Temperature":
                    _ , entrpy_num_train = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                            train_snap_unscaled,
                                                                            lambda_therm, t0, delta_T)
                    
                    _, entrpy_num_prediction = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                                    prediction_unscaled,
                                                                                    lambda_therm, t0, delta_T)

                case "Entropy":
                    comsol_data.mesh.point_data['temp_pred'] = prediction_unscaled
                    comsol_data.mesh.point_data['temp_test'] = train_snap_unscaled
                    integrated = comsol_data.mesh.integrate_data()
                    s0_total_pred = integrated.point_data['temp_pred'][0] * ureg.watt / ureg.kelvin
                    s0_total_train = integrated.point_data['temp_test'][0] * ureg.watt / ureg.kelvin
                    
                    L = param_df.loc["H", "quantity_pint"]
                    
                    s0_characteristic = (lambda_therm * delta_T**2) / (L**2 * t0**2)
                    entrpy_num_train = s0_total_train / s0_characteristic / (comsol_data.mesh.volume * ureg.meter**3)  
                    entrpy_num_prediction = s0_total_pred / s0_characteristic / (comsol_data.mesh.volume * ureg.meter**3)  
                    assert entrpy_num_test.check(['dimensionless']) # check for correct unit
                    assert entrpy_num_prediction.check(['dimensionless']) # check for correct unit
                   
            entrpy_nums_training[train_idx] = entrpy_num_train.magnitude
            entrpy_nums_prediction[train_idx] = entrpy_num_prediction.magnitude
        
        assert np.all(predictions_scaled != 0), " train predictions_scaled contains zero(s)"
        assert np.all(predictions_unscaled != 0), "train predictions_unscaled contains zero(s)"
        assert np.all(train_solutions_unscaled != 0), "train_solutions_unscaled contains zero(s)"
        
        r2_scaled = Q2_metric(data_module.training_snaps_scaled, predictions_scaled)
        r2_unscaled  =  Q2_metric(train_solutions_unscaled, predictions_unscaled)
        entropy_corr_coeff_train = np.corrcoef(entrpy_nums_prediction, entrpy_nums_training)[0, 1]
        entropy_mse_train = np.mean((entrpy_nums_prediction-entrpy_nums_training)**2)
        

        cursor.execute('''
            INSERT INTO nirb_results (
                norm, Q2_scaled, Q2_unscaled, R2_scaled, R2_unscaled,
                Version, Entropy_MSE_test, Entropy_R2_test,
                Entropy_MSE_train, Entropy_R2_train, Accuracy, Path,
                Epoch, Global_step, feature_scaling, Path_mtime
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(norm, Version, Accuracy, Path) DO UPDATE SET
                Q2_scaled = excluded.Q2_scaled,
                Q2_unscaled = excluded.Q2_unscaled,
                R2_scaled = excluded.R2_scaled,
                R2_unscaled = excluded.R2_unscaled,
                Entropy_MSE_test = excluded.Entropy_MSE_test,
                Entropy_R2_test = excluded.Entropy_R2_test,
                Entropy_MSE_train = excluded.Entropy_MSE_train,
                Entropy_R2_train = excluded.Entropy_R2_train,
                Epoch = excluded.Epoch,
                Global_step = excluded.Global_step,
                feature_scaling = excluded.feature_scaling,
                Path_mtime = excluded.Path_mtime
        ''', (
            SUFFIX, q2_scaled, q2_unscaled, r2_scaled, r2_unscaled,
            version, entropy_mse_test, entropy_corr_coeff_test,
            entropy_mse_train, entropy_corr_coeff_train, ACCURACY,
            str(chk_pt_path), epoch, global_step, scaler_features,
            datetime.fromtimestamp(chk_pt_path.stat().st_mtime).strftime("%y-%m-%d-%h")
        ))

        conn.commit()
        conn.close()

        logging.info(f"{q2_scaled=:.3e}")
        logging.info(f"{r2_scaled=:.3e}")
        logging.info(f"{entropy_corr_coeff_test=:.1%}")
        logging.info(f"{entropy_corr_coeff_train=:.1%}")
        logging.info(f"{entropy_mse_test=:.1e}")

if __name__ == "__main__":
    setup_logger(is_console=True, log_file=Path(__file__).parent / 'E_quality.log', level = logging.INFO)
    main()


