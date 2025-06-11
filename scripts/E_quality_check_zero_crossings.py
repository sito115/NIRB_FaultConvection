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
from src.offline_stage import NirbModule,  get_n_outputs, Normalizations
from src.pod import MinMaxNormalizer, MeanNormalizer, Standardizer, Normalizer
import src.offline_stage 
from comsol_module import COMSOL_VTU
from src.utils import (load_pint_data,
                       setup_logger,
                       safe_parse_quantity,
                       Q2_metric,
                       calculate_thermal_entropy_generation)

# Create dummy enum with string values for safe unpickling
torch.serialization.add_safe_globals([torch.nn.modules.activation.Tanh,
                                        torch.nn.modules.activation.Sigmoid,
                                        torch.nn.modules.activation.LeakyReLU,
                                        torch.nn.modules.activation.ReLU,
                                        Normalizations,
                                        MinMaxNormalizer,
                                        MeanNormalizer,
                                        Normalizer,
                                        src.offline_stage.data_module.Normalizations,
                                        np.dtypes.Float64DType,
                                        np.float64,
                                        np.int64,
                                        np._core.multiarray.scalar,
                                        np.dtype,
                                        Standardizer])

def main():
    PARAMETER_SPACE = "07"
    ROOT = Path(__file__).parents[1] / "data" / PARAMETER_SPACE
    assert ROOT.exists()
    ureg = pint.get_application_registry()
    cutoff_datetime = datetime(2025, 6, 5, 15, 0, 0).timestamp()
    PATTERN = r"(\d+\.\d+e[+-]?\d+)(.*)"
    FIELD_NAME = "Temperature"
    IS_OVERWRITE = True
    PROJECTION = "Original"  # "Mapped" or "Original"
    BASIS_FTS_FOLDER_NAME = f"BasisFunctionsPerZeroCrossing{FIELD_NAME}"  # "BasisFunctions" or "BasisFunctionsPerZeroCrossing"
    control_mesh_suffix = None # "s100_100_100_b0_4000_0_5000_-4000_0"
    
    chk_pt_paths = sorted([path for path in ROOT.rglob("*.ckpt") if (path.stat().st_mtime > cutoff_datetime and "zc" in str(path))])
    if FIELD_NAME == "Temperature":
        [s for s in chk_pt_paths if 'entropy' not in str(s).lower()]
    else:
        chk_pt_paths = [s for s in chk_pt_paths if FIELD_NAME.lower() in str(s).lower()]
    
    basis_functions_folder = ROOT / f"Training{PROJECTION}" / control_mesh_suffix / BASIS_FTS_FOLDER_NAME if control_mesh_suffix else ROOT / BASIS_FTS_FOLDER_NAME
        
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
        comsol_data = COMSOL_VTU(ROOT / f"Training{PROJECTION}" / control_mesh_suffix /f"Training_000_{control_mesh_suffix}.vtu")
    else:
        comsol_data = COMSOL_VTU(ROOT / "TrainingOriginal" / "Training_000.vtu")
    comsol_data.mesh.clear_data()


    param_folder = ROOT / "Exports"
    param_files_test = sorted([file for file in param_folder.rglob("*.csv") if "test" in file.stem.lower()])
    param_files_train = sorted([file for file in param_folder.rglob("*.csv") if "train" in file.stem.lower()])

    assert len(param_files_test) == len(test_parameters), f"Test parameters {len(param_files_test)} do not match test parameters {len(test_parameters)}"
    assert len(param_files_train) == len(training_parameters), f"Train parameters {len(param_files_train)} do not match train parameters {len(training_parameters)}"

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
        indices_with_0_zc_train = np.append(indices_train, grouped_zc_train.get(0, []))
        indices_test = grouped_zc_test.get(zc, [])
        indices_with_0_zc_test = np.append(indices_test, grouped_zc_test.get(0, []))
    
        try:
            trained_model : NirbModule = NirbModule.load_from_checkpoint(chk_pt_path)
        except (FileNotFoundError, ValueError) as e:
            logging.error(e)
            continue
        
        trained_model = trained_model.to('cpu')
        trained_model.eval()
        checkpoint = torch.load(chk_pt_path, map_location='cpu')
        epoch = checkpoint.get('epoch', None)
        global_step = checkpoint.get('global_step', None)
        
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
        
        conn = sqlite3.connect(ROOT / f"results_all_zero_crossing{FIELD_NAME}.db")
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nirb_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norm TEXT,
            zc INTEGER,
            scaler_features TEXT,
            scaler_output TEXT,
            Version TEXT,
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
        
        if PROJECTION == "Mapped":
            export_root_train = ROOT / f"Training{PROJECTION}" / control_mesh_suffix / "Exports"
            export_root_test = ROOT / f"Test{PROJECTION}" / control_mesh_suffix / "Exports"
        else:
            export_root_train = ROOT / f"Training{PROJECTION}" 
            export_root_test = ROOT / f"Test{PROJECTION}"
            
        assert export_root_train.exists(), f"Export root train {export_root_train} does not exist."
        assert export_root_test.exists(), f"Export root test {export_root_test} does not exist."
        
        match FIELD_NAME:
            case "Temperature":
                if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                    training_snapshots_npy      = np.load(export_root_train / "Training_Temperature_minus_tgrad.npy")
                    test_snapshots_npy          = np.load(export_root_test / "Test_Temperature_minus_tgrad.npy")
                    training_snapshots  = training_snapshots_npy[:, -1, :]
                    test_snapshots      = test_snapshots_npy[:, -1, :]
                    logging.info(f"Using snapshots with tgrad for {SUFFIX=}")

                else:
                    training_snapshots_npy      = np.load(export_root_train / "Training_Temperature.npy")
                    test_snapshots_npy          = np.load(export_root_test / "Test_Temperature.npy")
                    training_snapshots  = training_snapshots_npy[:, -1, :]
                    test_snapshots      = test_snapshots_npy[:, -1, :]
                    logging.info(f"Using snapshots without tgrad for {SUFFIX=}")
            case "EntropyNum":
                    training_snapshots_npy =  np.load(ROOT / "TrainingOriginal" / "Training_entropy_gen_number_therm.npy")
                    test_snapshots_npy =  np.load(ROOT / "TestOriginal" / "Test_entropy_gen_number_therm.npy")
                    training_snapshots = training_snapshots_npy[:, -1:]
                    test_snapshots  = test_snapshots_npy[:, -1:]

        if "mean" in SUFFIX.lower():
            scaling_output = MeanNormalizer()
        elif "min_max" in SUFFIX.lower():
            scaling_output = MinMaxNormalizer()
        else:
            raise ValueError("Invalid suffix.")
        training_snapshots_scaled = scaling_output.normalize(training_snapshots)
        test_snapshots_scaled = scaling_output.normalize_reuse_param(test_snapshots)

        if  "standard" in scaler_features.lower():
            scaling_features = Standardizer()
        elif "min_max" in scaler_features.lower() or "minmax" in scaler_features.lower():
            scaling_features = MinMaxNormalizer()
        elif "mean" in scaler_features.lower():
            scaling_features = MeanNormalizer()
        else:
            raise ValueError(f"Unknown scaler_features: {scaler_features}")
    
        training_parameters_scaled = scaling_features.normalize(training_parameters)
        test_parameters_scaled = scaling_features.normalize_reuse_param(test_parameters)
        
        logging.info(f'Scaling output: {scaling_output}, Scaling features: {scaling_features}')

        indices_train = grouped_zc_train.get(zc, [])
        indices_with_0_zc_train = np.append(indices_train, grouped_zc_train.get(0, []))
        indices_test = grouped_zc_test.get(zc, [])
        indices_with_0_zc_test = np.append(indices_test, grouped_zc_test.get(0, []))
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
        for idx, sample_idx in enumerate(indices_with_0_zc_test):
            parameters_df_file = param_files_test[sample_idx]
            param_df = pd.read_csv(parameters_df_file, index_col = 0)
            param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
            lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                                param_df.loc['host_phi', "quantity_pint"] * (4.2 * ureg.watt / (ureg.meter * ureg.kelvin))
            t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
            delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
            param = test_parameters_scaled[sample_idx]
            param_t = torch.from_numpy(param.astype(np.float32)).view(1, -1) # shape [1, n_param]
            res = trained_model(param_t)
            res_np = res.detach().numpy()
            my_sol_scaled = np.matmul(res_np.flatten(), basis_functions)
            predictions_scaled[idx, :] = my_sol_scaled
            
            if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                z_points = comsol_data.mesh.points[:, -1] * ureg.meter
                tgrad = (param_df.loc["T_c", "quantity_pint"] - (delta_T / param_df.loc["H", "quantity_pint"] * z_points)).to('K').magnitude
                assert np.all(tgrad > 0)
                test_snap_unscaled = test_snapshots[sample_idx] + tgrad
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled) + tgrad
            elif 'init' in SUFFIX.lower():
                test_snap_unscaled = test_snapshots[sample_idx] + test_snapshots_npy[sample_idx, 0, :]
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled) +  test_snapshots_npy[sample_idx, 0, :]
            else:
                test_snap_unscaled =  test_snapshots[sample_idx]
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled)
                
            predictions_unscaled[idx, :] = prediction_unscaled
            test_solutions_unscaled[idx, :] = test_snap_unscaled
            
            if "temperature" in FIELD_NAME.lower():
                _ , entrpy_num_test = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                        test_snap_unscaled,
                                                                        lambda_therm, t0, delta_T)
                
                _, entrpy_num_prediction = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                                prediction_unscaled,
                                                                                lambda_therm, t0, delta_T)
                entrpy_nums_test[idx] = entrpy_num_test.magnitude
                entrpy_nums_prediction[idx] = entrpy_num_prediction.magnitude
            
        
        q2_scaled = Q2_metric(test_snapshots_scaled[indices_with_0_zc_test], predictions_scaled)
        q2_unscaled  =  Q2_metric(test_solutions_unscaled, predictions_unscaled)
        entropy_corr_coeff_test = np.corrcoef(entrpy_nums_prediction, entrpy_nums_test)[0, 1]
        entropy_mse_test = np.mean((entrpy_nums_prediction-entrpy_nums_test)**2)
        
        
        # %% Training Predictions
        predictions_scaled = np.zeros((len(indices_with_0_zc_train), basis_functions.shape[1]))
        predictions_unscaled = np.zeros_like(predictions_scaled)
        train_solutions_unscaled = np.zeros_like(predictions_scaled)
        entrpy_nums_training = np.zeros((len(indices_with_0_zc_train)))
        entrpy_nums_prediction = np.zeros((len(indices_with_0_zc_train)))
        for idx, sample_idx in enumerate(indices_with_0_zc_train):
            parameters_df_file = param_files_train[sample_idx]
            param_df = pd.read_csv(parameters_df_file, index_col = 0)
            param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
            lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                                param_df.loc['host_phi', "quantity_pint"] * (4.2 * ureg.watt / (ureg.meter * ureg.kelvin))
            t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
            delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
            param = training_parameters_scaled[sample_idx]
            param_t = torch.from_numpy(param.astype(np.float32)).view(1, -1) # shape [1, n_param]
            res = trained_model(param_t)
            res_np = res.detach().numpy()
            my_sol_scaled = np.matmul(res_np.flatten(), basis_functions)
            predictions_scaled[idx, :] = my_sol_scaled
            
            if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                z_points = comsol_data.mesh.points[:, -1] * ureg.meter
                tgrad = (param_df.loc["T_c", "quantity_pint"] - (delta_T / param_df.loc["H", "quantity_pint"] * z_points)).to('K').magnitude
                assert np.all(tgrad > 0)
                train_snap_unscaled = training_snapshots[sample_idx] + tgrad
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled) + tgrad
            elif 'init' in SUFFIX.lower():
                train_snap_unscaled = training_snapshots[sample_idx] + training_snapshots_npy[sample_idx, 0, :]
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled) +  training_snapshots_npy[sample_idx, 0, :]
            else:
                train_snap_unscaled = training_snapshots[sample_idx]
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled)
                
            predictions_unscaled[idx, :] = prediction_unscaled
            train_solutions_unscaled[idx, :] = train_snap_unscaled
            
            if "temperature" in FIELD_NAME.lower():
                _ , entrpy_num_test = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                        train_snap_unscaled,
                                                                        lambda_therm, t0, delta_T)
                
                _, entrpy_num_prediction = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                                prediction_unscaled,
                                                                                lambda_therm, t0, delta_T)
                entrpy_nums_training[idx] = entrpy_num_test.magnitude
                entrpy_nums_prediction[idx] = entrpy_num_prediction.magnitude
            
        
        r2_scaled = Q2_metric(training_snapshots_scaled[indices_with_0_zc_train], predictions_scaled)
        r2_unscaled  =  Q2_metric(train_solutions_unscaled, predictions_unscaled)
        entropy_corr_coeff_train = np.corrcoef(entrpy_nums_prediction, entrpy_nums_training)[0, 1]
        entropy_mse_train = np.mean((entrpy_nums_prediction-entrpy_nums_training)**2)
        
        cursor.execute('''
        INSERT INTO nirb_results (
            norm, zc, scaler_features, scaler_output,
            Q2_scaled, Q2_unscaled, R2_scaled, R2_unscaled,
            Version, Entropy_MSE_test, Entropy_R2_test,
            Entropy_MSE_train, Entropy_R2_train, Accuracy, Path,
            Epoch, Global_step
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(norm, Version, Accuracy, Path, zc) DO UPDATE SET
            Q2_scaled = excluded.Q2_scaled,
            Q2_unscaled = excluded.Q2_unscaled,
            R2_scaled = excluded.R2_scaled,
            R2_unscaled = excluded.R2_unscaled,
            Global_step = excluded.Global_step,
            Epoch  = excluded.Epoch,
            Entropy_MSE_test = excluded.Entropy_MSE_test,
            Entropy_R2_test = excluded.Entropy_R2_test,
            Entropy_MSE_train = excluded.Entropy_MSE_train,
            Entropy_R2_train = excluded.Entropy_R2_train,
            Path = excluded.Path,
            Accuracy = excluded.Accuracy,
            scaler_features = excluded.scaler_features,
            scaler_output = excluded.scaler_output,
            zc = excluded.zc
        ''', (
            SUFFIX, zc, scaler_features, scaler_outputs,
            q2_scaled, q2_unscaled, r2_scaled, r2_unscaled,
            version, entropy_mse_test, entropy_corr_coeff_test,
            entropy_mse_train, entropy_corr_coeff_train, ACCURACY, str(chk_pt_path),
            epoch, global_step
        ))
        conn.commit()
        logging.info(f"Inserted/Updated results for {chk_pt_path.name} with zc={zc}, accuracy={ACCURACY}, suffix={SUFFIX}")
        conn.close()


if __name__ == "__main__":
    setup_logger(is_console=True, log_file='E_quality.log', level=logging.INFO)
    main()
