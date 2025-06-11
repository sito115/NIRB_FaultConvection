import numpy as np
import torch
from pathlib import Path
import sys
import pandas as pd
import pint
import sqlite3
import logging
from tqdm import tqdm
import plotly.graph_objects as go
sys.path.append(str(Path(__file__).parents[1]))
from src.offline_stage import NirbModule,  get_n_outputs, Normalizations
from src.pod import MinMaxNormalizer, MeanNormalizer, Standardizer, Normalizer
from comsol_module import COMSOL_VTU
from src.utils import (load_pint_data,
                       setup_logger,
                       safe_parse_quantity,
                       Q2_metric,
                       calculate_thermal_entropy_generation)



def main():
    PARAMETER_SPACE = "07"
    ROOT = Path(__file__).parents[1] / "data" / PARAMETER_SPACE
    assert ROOT.exists()
    ureg = pint.get_application_registry()
    PATTERN = r"(\d+\.\d+e[+-]?\d+)(.*)"
    FIELD_NAME = "Temperature"
    IS_OVERWRITE = True
    PROJECTION = "Original"  # "Mapped" or "Original"
    control_mesh_suffix = None # "s100_100_100_b0_4000_0_5000_-4000_0"

    df_basis_functions = pd.read_csv(ROOT / f"df_basis_functions_{FIELD_NAME}_zc.csv")
    conn = sqlite3.connect(ROOT / f"results_all_zero_crossing{FIELD_NAME}.db") 
    df_quality_checks = pd.read_sql_query("SELECT * FROM nirb_results", conn)
    conn.close()
    df_quality_checks['mean_metric'] = df_quality_checks[['Q2_unscaled', 'R2_unscaled']].mean(axis=1)
    df_quality_checks_grouped = df_quality_checks.groupby(by = "zc")

    if control_mesh_suffix is not None:
        comsol_data = COMSOL_VTU(ROOT / f"Training{PROJECTION}" / control_mesh_suffix /f"Training_000_{control_mesh_suffix}.vtu")
    else:
        comsol_data = COMSOL_VTU(ROOT / "TrainingOriginal" / "Training_000.vtu")
    comsol_data.mesh.clear_data()

    fig = go.Figure()
    vmin = df_quality_checks['zc'].values.min()
    vmax = df_quality_checks['zc'].values.max()
    cmap = 'viridis' #'portland'

    for group_name, group_df in tqdm(df_quality_checks_grouped, total = len(df_quality_checks_grouped)):
        zc = int(group_name)
        group_df_sorted = group_df.sort_values(by = 'mean_metric')
        row = group_df_sorted.iloc[0]
        if "temperature" in FIELD_NAME.lower():
            assert row['Entropy_R2_test'] > 0
            assert row['Entropy_R2_train'] > 0
        for col in group_df_sorted.columns:
            logging.info(f'\t {col} - {row[col]}')
        chk_pt_path = group_df_sorted.iloc[0]['Path']
        scaler_features = group_df_sorted.iloc[0]['scaler_features']
        scaler_output   = group_df_sorted.iloc[0]['scaler_output']
        suffix          = group_df_sorted.iloc[0]['norm']
        accuracy        = group_df_sorted.iloc[0]['Accuracy']

        filtered_df_basis_functions = df_basis_functions[(df_basis_functions['zc'] == zc) & 
                                                        (df_basis_functions['suffix'] == suffix) &
                                                        (df_basis_functions['accuracy'] == accuracy)]
        assert len(filtered_df_basis_functions) == 1
        bsf_fts_path = filtered_df_basis_functions.path.values
        basis_functions = np.load(bsf_fts_path[0])
        trained_model  : NirbModule = NirbModule.load_from_checkpoint(chk_pt_path)
        trained_model = trained_model.to('cpu')
        trained_model.eval()    
    
        training_parameters         = load_pint_data(ROOT / "training_samples.csv", is_numpy = True)
        test_parameters             = load_pint_data(ROOT / "test_samples.csv", is_numpy = True)
        training_parameters_pint = load_pint_data(ROOT / "training_samples.csv")
        test_parameters_pint = load_pint_data(ROOT / "test_samples.csv")
        
        zero_crossings_train = np.load(ROOT / "Exports" / "Training_zero_crossings.npy")
        zero_crossings_test = np.load(ROOT / "Exports" / "Test_zero_crossings.npy")
        assert len(zero_crossings_test) == len(test_parameters), f"Test zero crossings {len(zero_crossings_test)} do not match test parameters {len(test_parameters)}"
        assert len(zero_crossings_train) == len(training_parameters), f"Train zero crossings {len(zero_crossings_train)} do not match train parameters {len(training_parameters)}"
        param_folder = ROOT / "Exports"
        param_files_test = sorted([file for file in param_folder.rglob("*.csv") if "test" in file.stem.lower()])
        param_files_train = sorted([file for file in param_folder.rglob("*.csv") if "train" in file.stem.lower()])
        assert len(param_files_test) == len(test_parameters), f"Test parameters {len(param_files_test)} do not match test parameters {len(test_parameters)}"
        assert len(param_files_train) == len(training_parameters), f"Train parameters {len(param_files_train)} do not match train parameters {len(training_parameters)}"

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
                if 'init' in suffix.lower() and 'grad' in suffix.lower():
                    training_snapshots_npy      = np.load(export_root_train / "Training_Temperature_minus_tgrad.npy")
                    test_snapshots_npy          = np.load(export_root_test / "Test_Temperature_minus_tgrad.npy")
                    training_snapshots  = training_snapshots_npy[:, -1, :]
                    test_snapshots      = test_snapshots_npy[:, -1, :]
                    logging.info(f"Using snapshots with tgrad for {suffix=}")

                else:
                    training_snapshots_npy      = np.load(export_root_train / "Training_Temperature.npy")
                    test_snapshots_npy          = np.load(export_root_test / "Test_Temperature.npy")
                    training_snapshots  = training_snapshots_npy[:, -1, :]
                    test_snapshots      = test_snapshots_npy[:, -1, :]
                    logging.info(f"Using snapshots without tgrad for {suffix=}")
            case "EntropyNum":
                training_snapshots_npy =  np.load(ROOT / "TrainingOriginal" / "Training_entropy_gen_number_therm.npy")
                test_snapshots_npy =  np.load(ROOT / "TestOriginal" / "Test_entropy_gen_number_therm.npy")
                training_snapshots = training_snapshots_npy[:, -1:]
                test_snapshots  = test_snapshots_npy[:, -1:]

        if "mean" in suffix.lower():
            scaling_output = MeanNormalizer()
        elif "min_max" in suffix.lower():
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

        entrpy_nums_prediction = np.zeros((len(test_parameters_scaled)))
        for idx, test_param_scaled in enumerate(test_parameters_scaled):
            parameters_df_file = param_files_test[idx]
            param_df = pd.read_csv(parameters_df_file, index_col = 0)
            param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
            lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                                param_df.loc['host_phi', "quantity_pint"] * (4.2 * ureg.watt / (ureg.meter * ureg.kelvin))
            t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
            delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
            param = test_param_scaled
            param_t = torch.from_numpy(param.astype(np.float32)).view(1, -1) # shape [1, n_param]
            res = trained_model(param_t)
            res_np = res.detach().numpy()
            my_sol_scaled = np.matmul(res_np.flatten(), basis_functions)

            if 'init' in suffix.lower() and 'grad' in suffix.lower():
                z_points = comsol_data.mesh.points[:, -1] * ureg.meter
                tgrad = (param_df.loc["T_c", "quantity_pint"] - (delta_T / param_df.loc["H", "quantity_pint"] * z_points)).to('K').magnitude
                assert np.all(tgrad > 0)
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled) + tgrad
            elif 'init' in suffix.lower():
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled) +  test_snapshots_npy[idx, 0, :]
            else:
                prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled)
           
            if "temperature" in FIELD_NAME.lower():     
                _, entrpy_num_prediction = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                                prediction_unscaled,
                                                                                lambda_therm, t0, delta_T)
                entrpy_nums_prediction[idx] = entrpy_num_prediction.magnitude
            else:
                entrpy_nums_prediction[idx] = prediction_unscaled
        
        custom_data_test = np.column_stack([
        np.arange(len(test_parameters)),
        np.array([zc] * len(test_parameters))
    ])
        
        fig.add_trace(go.Scatter(
            x=test_parameters.flatten(),
            y=entrpy_nums_prediction.flatten(),
            mode='markers',
            name = f'Test Samples zc {zc}',
            visible="legendonly",
            marker=dict(
                size=8,
                color=np.ones((len(test_parameters), )) * zc,  # face color = scalar value
                colorscale=cmap,
                cmin=vmin,
                cmax=vmax,
                line=dict(
                    color='black',  # solid edge
                    width=1
                ),
                opacity=0.9,
                showscale=False
            ),
            customdata=custom_data_test,
            hovertemplate=(
                'Test Idx: %{customdata[0]:03d}<br>' +
                f'{training_parameters_pint.columns[0]}: %{{x:.2f}}<br>' +
                'Entropy Number: %{y:.2f}<br>' +
                'zero_crossings: %{customdata[1]:d}<extra></extra>'
            )
        ))

        entrpy_nums_prediction = np.zeros((len(training_parameters)))
        for idx, train_param_scaled in enumerate(training_parameters_scaled):
                    parameters_df_file = param_files_train[idx]
                    param_df = pd.read_csv(parameters_df_file, index_col = 0)
                    param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
                    lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                                        param_df.loc['host_phi', "quantity_pint"] * (4.2 * ureg.watt / (ureg.meter * ureg.kelvin))
                    t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
                    delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
                    param = train_param_scaled
                    param_t = torch.from_numpy(param.astype(np.float32)).view(1, -1) # shape [1, n_param]
                    res = trained_model(param_t)
                    res_np = res.detach().numpy()
                    my_sol_scaled = np.matmul(res_np.flatten(), basis_functions)

                    if 'init' in suffix.lower() and 'grad' in suffix.lower():
                        z_points = comsol_data.mesh.points[:, -1] * ureg.meter
                        tgrad = (param_df.loc["T_c", "quantity_pint"] - (delta_T / param_df.loc["H", "quantity_pint"] * z_points)).to('K').magnitude
                        assert np.all(tgrad > 0)
                        prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled) + tgrad
                    elif 'init' in suffix.lower():
                        prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled) +  test_snapshots_npy[idx, 0, :]
                    else:
                        prediction_unscaled = scaling_output.inverse_normalize(my_sol_scaled)
                
                    if "temperature" in FIELD_NAME.lower():                 
                        _, entrpy_num_prediction = calculate_thermal_entropy_generation(comsol_data.mesh,
                                                                                        prediction_unscaled,
                                                                                        lambda_therm, t0, delta_T)
                        entrpy_nums_prediction[idx] = entrpy_num_prediction.magnitude
                    else:
                        entrpy_nums_prediction[idx] = prediction_unscaled

        custom_data_test = np.column_stack([
        np.arange(len(training_parameters)),
        np.array([zc] * len(training_parameters))
        ])

        fig.add_trace(go.Scatter(
            x=training_parameters.flatten(),
            y=entrpy_nums_prediction.flatten(),
            mode='markers',
            name = f'Train Samples zc {zc}',
            marker=dict(
                size=8,
                color=np.ones((len(training_parameters), )) * zc,  # face color = scalar value
                colorscale=cmap,
                cmin=vmin,
                cmax=vmax,
                line=dict(
                    color='black',  # solid edge
                    width=1
                ),
                opacity=0.9,
                showscale=False
            ),
            customdata=custom_data_test,
            showlegend=True,
            hovertemplate=(
                'Train Idx: %{customdata[0]:03d}<br>' +
                f'{training_parameters_pint.columns[0]}: %{{x:.2f}}<br>' +
                'Entropy Number: %{y:.2f}<br>' +
                'zero_crossings: %{customdata[1]:d}<extra></extra>'
            )
        ))



    fig.update_layout(
        title=f'Parameter Space {PARAMETER_SPACE} - Number of Convection Cells',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="left",
            x=0,
            traceorder="normal",
            itemwidth=70  # control spacing
        )
    )

    fig.show()
    fig.write_html(ROOT / "Exports" / f"Zero_Crossing_{PARAMETER_SPACE}_{FIELD_NAME}_Test_QC.html")



if __name__ == "__main__":
    setup_logger(is_console=True, level=logging.INFO)
    main()