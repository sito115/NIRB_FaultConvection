import numpy as np
import torch
from pathlib import Path
import sys
import pandas as pd
import pint
import sqlite3
import re
import logging
from tqdm import tqdm
from datetime import datetime
sys.path.append(str(Path(__file__).parents[1]))
from src.offline_stage import NirbModule,  get_n_outputs
from src.pod import match_scaler, MinMaxNormalizer, MeanNormalizer, Standardizer, Normalizer
from comsol_module import COMSOL_VTU
from src.utils import (load_pint_data,
                       setup_logger,
                       safe_parse_quantity,
                       find_snapshot_path,
                       Q2_metric,
                       calculate_thermal_entropy_generation,
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
    IS_OVERWRITE = True
    PROJECTION = "Mapped"  # "Mapped" or "Original"
    FIELD_NAME = "Temperature" #"Entropy"
    is_clean_mesh = False
    spacing = 100
    control_mesh_suffix = f"s{spacing}_{spacing}_{spacing}_b0_4000_0_5000_-4000_0"
    config = read_config()

    
    chk_pt_paths = sorted([path for path in ROOT.rglob("*.ckpt") if (path.stat().st_mtime > cutoff_datetime and "zc" in str(path))])
    if FIELD_NAME == "Temperature":
        [s for s in chk_pt_paths if 'entropy' not in str(s).lower()]
    else:
        chk_pt_paths = [s for s in chk_pt_paths if FIELD_NAME.lower() in str(s).lower()]
    
    basis_functions_folder = ROOT / f"Training{PROJECTION}" / control_mesh_suffix / f"BasisFunctions{FIELD_NAME}ZeroCrossings" if control_mesh_suffix else ROOT / f"BasisFunctions{FIELD_NAME}ZeroCrossings"
    assert basis_functions_folder.exists()   
        
    df_basis_functions = pd.DataFrame([
        {'path': str(p), 'shape': m.shape, 'n_basis': m.shape[0], 'n_points': m.shape[1] if m.ndim > 1 else 1}
        for p, m in {
            path: np.load(path) 
            for path in basis_functions_folder.rglob('*.npy') if 'basis' in path.stem
        }.items()
    ])
    df_basis_functions['basis_functions'] = df_basis_functions['path'].apply(np.load)
    df_basis_functions['zc'] = df_basis_functions['path'].apply(lambda x: int(Path(x).stem.split("_")[-1][2:]) if "zc" in Path(x).stem else -1)
    df_basis_functions['match'] = df_basis_functions['path'].apply(lambda x: re.search(PATTERN, Path(x).stem))
    df_basis_functions['accuracy'] = df_basis_functions['match'].apply(lambda x: float(x.group(1)) if x else np.nan)
    df_basis_functions['suffix'] = df_basis_functions['match'].apply(lambda x: x.group(2)  if x else '')
    df_basis_functions.to_csv(ROOT / f"df_basis_functions_{FIELD_NAME}_zc.csv")
    logging.debug(f'Loaded {len(df_basis_functions)} different basis function')
    logging.debug(df_basis_functions)

    training_parameters         = load_pint_data(ROOT / "training_samples.csv", is_numpy = True)
    test_parameters             = load_pint_data(ROOT / "test_samples.csv", is_numpy = True)
    zero_crossings_train = np.load(ROOT / "Exports" / "Training_zero_crossings.npy")
    zero_crossings_test = np.load(ROOT / "Exports" / "Test_zero_crossings.npy")

    assert len(zero_crossings_test) == len(test_parameters), f"Test zero crossings {len(zero_crossings_test)} do not match test parameters {len(test_parameters)}"
    assert len(zero_crossings_train) == len(training_parameters), f"Train zero crossings {len(zero_crossings_train)} do not match train parameters {len(training_parameters)}"

    unique_zc = np.unique(zero_crossings_train)
    grouped_zc_train = {int(val): np.where(zero_crossings_train == val)[0] for val in unique_zc }
    grouped_zc_test = {int(val): np.where(zero_crossings_test == val)[0] for val in unique_zc}

    if control_mesh_suffix is not None:
        comsol_data = COMSOL_VTU(ROOT / f"Training{PROJECTION}" / control_mesh_suffix /f"Training_000_{control_mesh_suffix}.vtk", is_clean_mesh=is_clean_mesh)
    else:
        comsol_data = COMSOL_VTU(ROOT / "TrainingOriginal" / "Training_000.vtu", is_clean_mesh=is_clean_mesh)
    comsol_data.mesh.clear_data()


    param_folder = ROOT / "Exports"
    param_files_test_all = np.array(sorted([file for file in param_folder.rglob("*.csv") if "test" in file.stem.lower()]))
    param_files_train_all = np.array(sorted([file for file in param_folder.rglob("*.csv") if "train" in file.stem.lower()]))

    assert len(param_files_test_all) == len(test_parameters), f"Test parameters {len(param_files_test_all)} do not match test parameters {len(test_parameters)}"
    assert len(param_files_train_all) == len(training_parameters), f"Train parameters {len(param_files_train_all)} do not match train parameters {len(training_parameters)}"

    # Start loopoing through checkpoint paths   
    for chk_pt_path in tqdm(chk_pt_paths, total=len(chk_pt_paths)):
        logging.info(chk_pt_path.relative_to(chk_pt_path.parents[2]))
        version = chk_pt_path.parent.parent.stem
        logging.debug(f'Loading {chk_pt_path=}')

        match = re.findall(r"zc(\d+)", str(chk_pt_path))
        if len(match) == 0:
            continue
        zc = int(match[0])
        
        assert zc in unique_zc, f"Zero crossing {zc} not found in unique zero crossings."
        indices_train = grouped_zc_train.get(zc, [])
        indices_with_0_zc_train = np.append(indices_train, grouped_zc_train.get(0, [])).astype(np.int16)
        indices_test = grouped_zc_test.get(zc, [])
        indices_with_0_zc_test = np.append(indices_test, grouped_zc_test.get(0, [])).astype(np.int16)
    
        try:
            trained_model : NirbModule = NirbModule.load_from_checkpoint(chk_pt_path)
        except (FileNotFoundError, ValueError) as e:
            logging.error(e)
            continue
        
        trained_model = trained_model.to('cpu')
        trained_model.eval()
        checkpoint = torch.load(chk_pt_path, map_location='cpu')

        scaler_features = checkpoint['hyper_parameters'].get('standardizer_features', 'MinMax')
        scaler_outputs = checkpoint['hyper_parameters'].get('normalizer', 'Standardizer')
        epoch = checkpoint.get('epoch', None)
        global_step = checkpoint.get('global_step', None)
        
        n_outputs = get_n_outputs(trained_model)
        filtered_basis_df = df_basis_functions.loc[df_basis_functions['zc'] == zc]
        filtered_basis_df = filtered_basis_df.loc[df_basis_functions['n_basis'] == n_outputs]
        if len(filtered_basis_df) == 0:
            logging.error(f'Did not find basis function for {chk_pt_path.name}')
            continue
        elif len(filtered_basis_df) > 1:
            add_info = chk_pt_path.parents[2].name
            if "init" in add_info.lower():
                filtered_basis_df = filtered_basis_df[filtered_basis_df["suffix"].str.contains("init", na=False)]
            else:
                filtered_basis_df = filtered_basis_df[~filtered_basis_df["suffix"].str.contains("init", na=False)]
            
            if "mean" in str(chk_pt_path).lower():
                filtered_basis_df = filtered_basis_df[filtered_basis_df["suffix"].str.contains("mean", na=False)]
            else:
                filtered_basis_df = filtered_basis_df[~filtered_basis_df["suffix"].str.contains("mean", na=False)]

        assert len(filtered_basis_df) == 1
        
        try:
            ACCURACY = filtered_basis_df.accuracy.values[0]
            SUFFIX = filtered_basis_df.suffix.values[0]
            basis_functions = filtered_basis_df.basis_functions.values[0]
        except KeyError:
            logging.error("Skipped.")
            continue
        
        logging.info(f"Loaded {version}")
        logging.info(f'{ACCURACY=}, {scaler_features=}, {SUFFIX=} , {scaler_outputs=}')  
        
        conn = sqlite3.connect(ROOT / f"results_all{FIELD_NAME}_zero_crossing.db")
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nirb_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norm TEXT,
            Version TEXT,
            zc INTEGER,
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
            UNIQUE(norm, Version, Accuracy, Path, zc)  -- Required for ON CONFLICT
        )
        ''')
        conn.commit()
        
        
        cursor.execute('''
            SELECT id FROM nirb_results
            WHERE norm = ? AND Version = ? AND Accuracy = ? AND Path = ? AND zc = ?
        ''', (
            SUFFIX, version, ACCURACY, str(chk_pt_path), zc
        ))

        result = cursor.fetchone()
        
        if result is not None and not IS_OVERWRITE:
            logging.warning(f'Skipped {chk_pt_path.name}: Already in database')
            continue
        
        # last time step
        training_snapshots = np.load(find_snapshot_path(PROJECTION, SUFFIX, FIELD_NAME, ROOT, control_mesh_suffix, "Training"))[:, -1, :]
        test_snapshots     = np.load(find_snapshot_path(PROJECTION, SUFFIX, FIELD_NAME, ROOT, control_mesh_suffix, "Test"))[:, -1, :]

        scaling_output = match_scaler(SUFFIX)
        scaling_features = match_scaler(scaler_features)
        
        training_parameters_scaled = scaling_features.normalize(training_parameters)
        test_parameters_scaled = scaling_features.normalize_reuse_param(test_parameters)
        training_parameters_scaled  = training_parameters_scaled[indices_with_0_zc_train]
        test_parameters_scaled =  test_parameters_scaled[indices_with_0_zc_test]
        
        training_snapshots_scaled = scaling_output.normalize(training_snapshots[indices_with_0_zc_train])
        test_snapshots_scaled = scaling_output.normalize_reuse_param(test_snapshots[indices_with_0_zc_test])
        
        param_files_train = param_files_train_all[indices_with_0_zc_train] 
        param_files_test = param_files_test_all[indices_with_0_zc_test]

        logging.info(f'Scaling output: {scaling_output}, Scaling features: {scaling_features}')
        logging.info(f"Number of training samples with zc={zc}: {len(indices_with_0_zc_train)}")
        logging.info(f"Number of test samples with zc={zc}: {len(indices_with_0_zc_test)}")
        logging.info(f"Number of training samples with zc=0: {len(grouped_zc_train.get(0, []))}")
        logging.info(f"Number of test samples with zc=0: {len(grouped_zc_test.get(0, []))}")

        # %% Test Predictions
        predictions_scaled = np.zeros((len(indices_with_0_zc_test), basis_functions.shape[1]))
        predictions_unscaled = np.zeros((len(indices_with_0_zc_test), basis_functions.shape[1]))
        test_solutions_unscaled = np.zeros((len(indices_with_0_zc_test), basis_functions.shape[1]))
        entrpy_nums_test = np.zeros((len(indices_with_0_zc_test)))
        entrpy_nums_prediction = np.zeros((len(indices_with_0_zc_test)))
        
        for idx, test_idx in enumerate(indices_with_0_zc_test):
            parameters_df_file = param_files_test[idx]
            param_df = pd.read_csv(parameters_df_file, index_col = 0)
            param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
            lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                                param_df.loc['host_phi', "quantity_pint"] * (config['lambda_f']  * ureg.watt / (ureg.meter * ureg.kelvin))
            t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
            delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
            param = test_parameters_scaled[idx]
            logging.debug(f"{param=}")
            param_t = torch.from_numpy(param.astype(np.float32)).view(1, -1) # shape [1, n_param]
            res_np = trained_model(param_t).detach().numpy()
            my_sol_scaled = np.matmul(res_np.flatten(), basis_functions)
            predictions_scaled[idx, :] = my_sol_scaled
            
            if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                z_points = comsol_data.mesh.points[:, -1] * ureg.meter
                tgrad = (param_df.loc["T_c", "quantity_pint"] - (delta_T / param_df.loc["H", "quantity_pint"] * z_points)).to('K').magnitude
                assert np.all(tgrad > 0)
                test_snap_unscaled = test_snapshots[test_idx] + tgrad
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled) + tgrad
            else:
                test_snap_unscaled =  test_snapshots[test_idx]
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled)
                if FIELD_NAME == "Entropy":
                    prediction_unscaled = np.power(10, prediction_unscaled)
                    test_snap_unscaled = np.power(10, test_snap_unscaled)

            predictions_unscaled[idx, :] = prediction_unscaled
            test_solutions_unscaled[idx, :] = test_snap_unscaled
            
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
                    
            entrpy_nums_test[idx] = entrpy_num_test.magnitude
            entrpy_nums_prediction[idx] = entrpy_num_prediction.magnitude
        
        
        q2_scaled = Q2_metric(test_snapshots_scaled, predictions_scaled)
        q2_unscaled  =  Q2_metric(test_solutions_unscaled, predictions_unscaled)
        entropy_corr_coeff_test = np.corrcoef(entrpy_nums_prediction, entrpy_nums_test)[0, 1]
        entropy_mse_test = np.mean((entrpy_nums_prediction-entrpy_nums_test)**2)
        
        
        # %% Training Predictions
        predictions_scaled = np.zeros((len(indices_with_0_zc_train), basis_functions.shape[1]))
        predictions_unscaled = np.zeros_like(predictions_scaled)
        train_solutions_unscaled = np.zeros_like(predictions_scaled)
        entrpy_nums_training = np.zeros((len(indices_with_0_zc_train)))
        entrpy_nums_prediction = np.zeros((len(indices_with_0_zc_train)))
        for idx, train_idx in enumerate(indices_with_0_zc_train):
            parameters_df_file = param_files_train[idx]
            param_df = pd.read_csv(parameters_df_file, index_col = 0)
            param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
            lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                                param_df.loc['host_phi', "quantity_pint"] * (config['lambda_f']  * ureg.watt / (ureg.meter * ureg.kelvin))
            t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
            delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
            param = training_parameters_scaled[idx]
            param_t = torch.from_numpy(param.astype(np.float32)).view(1, -1) # shape [1, n_param]
            res_np = trained_model(param_t).detach().numpy()
            my_sol_scaled = np.matmul(res_np.flatten(), basis_functions)
            predictions_scaled[idx, :] = my_sol_scaled
            
            if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                z_points = comsol_data.mesh.points[:, -1] * ureg.meter
                tgrad = (param_df.loc["T_c", "quantity_pint"] - (delta_T / param_df.loc["H", "quantity_pint"] * z_points)).to('K').magnitude
                assert np.all(tgrad > 0)
                train_snap_unscaled = training_snapshots[train_idx] + tgrad
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled) + tgrad
            else:
                train_snap_unscaled = training_snapshots[train_idx]
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled)
                
                if FIELD_NAME == "Entropy":
                    prediction_unscaled = np.power(10, prediction_unscaled)
                    train_snap_unscaled = np.power(10, train_snap_unscaled)
                
            predictions_unscaled[idx, :] = prediction_unscaled
            train_solutions_unscaled[idx, :] = train_snap_unscaled
            
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
            
            
            entrpy_nums_training[idx] = entrpy_num_train.magnitude
            entrpy_nums_prediction[idx] = entrpy_num_prediction.magnitude
        
        r2_scaled = Q2_metric(training_snapshots_scaled, predictions_scaled)
        r2_unscaled  =  Q2_metric(train_solutions_unscaled, predictions_unscaled)
        entropy_corr_coeff_train = np.corrcoef(entrpy_nums_prediction, entrpy_nums_training)[0, 1]
        entropy_mse_train = np.mean((entrpy_nums_prediction-entrpy_nums_training)**2)
        
        cursor.execute('''
        INSERT INTO nirb_results (
            norm, zc, feature_scaling,
            Q2_scaled, Q2_unscaled, R2_scaled, R2_unscaled,
            Version, Entropy_MSE_test, Entropy_R2_test,
            Entropy_MSE_train, Entropy_R2_train, Accuracy, Path,
            Epoch, Global_step, Path_mtime
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(norm, Version, Accuracy, Path, zc) DO UPDATE SET
            Q2_scaled = excluded.Q2_scaled,
            Q2_unscaled = excluded.Q2_unscaled,
            R2_scaled = excluded.R2_scaled,
            R2_unscaled = excluded.R2_unscaled,
            Global_step = excluded.Global_step,
            Epoch = excluded.Epoch,
            Entropy_MSE_test = excluded.Entropy_MSE_test,
            Entropy_R2_test = excluded.Entropy_R2_test,
            Entropy_MSE_train = excluded.Entropy_MSE_train,
            Entropy_R2_train = excluded.Entropy_R2_train,
            Path = excluded.Path,
            Accuracy = excluded.Accuracy,
            feature_scaling = excluded.feature_scaling,
            zc = excluded.zc,
            Path_mtime = excluded.Path_mtime
        ''', (
            SUFFIX, zc, scaler_features,
            q2_scaled, q2_unscaled, r2_scaled, r2_unscaled,
            version, entropy_mse_test, entropy_corr_coeff_test,
            entropy_mse_train, entropy_corr_coeff_train, ACCURACY, str(chk_pt_path),
            epoch, global_step, datetime.fromtimestamp(chk_pt_path.stat().st_mtime).strftime("%y-%m-%d-%h")
            ))
    
        conn.commit()
        logging.info(f"Inserted/Updated results for {chk_pt_path.name} with zc={zc}, accuracy={ACCURACY}, suffix={SUFFIX}")
        conn.close()

        logging.info(f"{q2_scaled=:.3e}")
        logging.info(f"{r2_scaled=:.3e}")
        logging.info(f"{entropy_corr_coeff_test=:.1%}")
        logging.info(f"{entropy_corr_coeff_train=:.1%}")
        logging.info(f"{entropy_mse_test=:.1e}")

if __name__ == "__main__":
    setup_logger(is_console=True, level=logging.INFO)
    main()
