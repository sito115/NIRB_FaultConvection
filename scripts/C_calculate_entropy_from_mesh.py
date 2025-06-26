from pathlib import Path
import logging
from tqdm import tqdm   
import numpy as np
import pandas as pd
import pint # noqa: F401
import sys
from plotly import graph_objects as go
from plotly import express as px
sys.path.append(str(Path(__file__).parents[1]))
from src.utils import safe_parse_quantity, setup_logger, read_config
from comsol_module.comsol_classes import COMSOL_VTU
from comsol_module.entropy import caluclate_entropy_gen_number_isotherm


def main():

    ROOT = Path(__file__).parents[1] 
    PARAMETER_SPACE = "10"
    DATA_TYPE = "Test"
    IS_EXPORT_ARRAY = False
    IS_EXPORT_ENTROPY_NUMBERS = True
    IS_EXPORT_HTML = True
    PROJECTION = "Original"
    spacing = 50
    control_mesh_suffix = f"s{spacing}_{spacing}_{spacing}_b0_4000_0_5000_-4000_0"
    is_clean = False
    config = read_config()

    import_folder = ROOT / "data" / PARAMETER_SPACE / f"{DATA_TYPE}{PROJECTION}"
    export_folder = import_folder
    if PROJECTION == "Mapped":
        import_folder = import_folder / control_mesh_suffix
        export_folder = import_folder / "Exports"
    assert import_folder.exists(), f"Import folder {import_folder} does not exist."
    comsol_vtu_files = sorted([path for path in import_folder.rglob("*.vt*")])
    N_SNAPS = len(comsol_vtu_files)

    param_folder = ROOT / "data" / PARAMETER_SPACE / "Exports"
    param_files = sorted([path for path in param_folder.rglob(f"{DATA_TYPE}*.csv")])
    assert len(param_files) == N_SNAPS
    ureg = pint.get_application_registry()


   
    N_TIME_STEPS = len( COMSOL_VTU(comsol_vtu_files[0], is_clean_mesh=is_clean).times.keys())
    TIME_STEPS = np.arange(N_TIME_STEPS) 
    N_CELLS = COMSOL_VTU(comsol_vtu_files[0],  is_clean_mesh=is_clean).mesh.n_points

    entropy_gen_number_therm = np.zeros((N_SNAPS, N_TIME_STEPS))
    entropy_gen_number_visc = np.zeros((N_SNAPS, N_TIME_STEPS))
    entropy_gen_per_vol_thermal = np.zeros((N_SNAPS, N_TIME_STEPS, N_CELLS))
    entropy_gen_per_vol_visc = np.zeros_like(entropy_gen_per_vol_thermal)

    for idx_snap in tqdm(range(N_SNAPS), total = N_SNAPS):
        comsol_data = COMSOL_VTU(comsol_vtu_files[idx_snap],  is_clean_mesh=is_clean)
        param_df = pd.read_csv(param_files[idx_snap], index_col = 0)
        param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
        lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                            param_df.loc['host_phi', "quantity_pint"] * (config['lambda_f']  * ureg.watt / (ureg.meter * ureg.kelvin))
        t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
        delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
        
        model_dict = {
            'lambda_m' : lambda_therm.magnitude,
            'T0' : t0.magnitude,
            'mu0': 1e-3, # comsol_data.get_array('Dynamic_viscosity', is_cell_data=False)[-1, :],
            'k_m': 1.5e-13,
            'delta_T' : delta_T.magnitude,
            'H' : comsol_data.mesh.bounds.z_max - comsol_data.mesh.bounds.z_min,
        }
        
        if IS_EXPORT_ARRAY:
            entropy_per_vol = comsol_data.calculate_total_entropy_per_vol(model_data=model_dict,
                                                                            time_steps=TIME_STEPS,
                                                                            is_return_as_integration=False)
            
            entropy_gen_per_vol_thermal[idx_snap, :, :] = entropy_per_vol[:, : , 0]  # thermal entropy generation per volume
            entropy_gen_per_vol_visc[idx_snap, :, :] = entropy_per_vol[:, : , 1]
    
    
        
        entropy_integrated= comsol_data.calculate_total_entropy_per_vol(model_data=model_dict,
                                                    time_steps=TIME_STEPS,
                                                    is_return_as_integration=True,)
        logging.info(f"{np.max(entropy_integrated)=:.3f}")

        
        entropy_gen_number_therm[idx_snap, :] = caluclate_entropy_gen_number_isotherm(s_total=entropy_integrated[:, 0],
                                                            L = model_dict["H"], 
                                                            lambda_m=model_dict["lambda_m"],
                                                            T_0=model_dict["T0"],
                                                            delta_T=model_dict["delta_T"],
                                                            V = comsol_data.mesh.volume,)
        logging.info(f"{np.max(entropy_gen_number_therm[idx_snap, :])=:.3f}")

        entropy_gen_number_visc[idx_snap, :] = caluclate_entropy_gen_number_isotherm(s_total=entropy_integrated[:, 1],
                                                            L = model_dict["H"], 
                                                            lambda_m=model_dict["lambda_m"],
                                                            T_0=model_dict["T0"],
                                                            delta_T=model_dict["delta_T"],
                                                            V = comsol_data.mesh.volume,)

    if IS_EXPORT_ENTROPY_NUMBERS:
        np.save(export_folder /f"{DATA_TYPE}_entropy_gen_number_therm.npy", entropy_gen_number_therm)
        np.save(export_folder /f"{DATA_TYPE}_entropy_gen_number_visc.npy", entropy_gen_number_visc)
    if IS_EXPORT_ARRAY:
        np.save(export_folder /f"{DATA_TYPE}_entropy_gen_per_vol_thermal.npy", entropy_gen_per_vol_thermal)
        np.save(export_folder /f"{DATA_TYPE}_entropy_gen_per_vol_visc.npy", entropy_gen_per_vol_thermal)

    fig = go.Figure()
    colors = px.colors.sample_colorscale("jet", [n/(N_SNAPS) for n in range(N_SNAPS)])
    for idx in range(N_SNAPS):
        fig.add_trace(go.Scatter(x=TIME_STEPS,
                                y=entropy_gen_number_therm[idx, :],
                                line=dict(color=colors[idx]),
                                marker_symbol='circle',
                                mode='lines+markers',
                                name=f'Thermal Entropy Gen Snap {idx}',
                                showlegend=False))
        fig.add_trace(go.Scatter(x=TIME_STEPS,
                                y=entropy_gen_number_visc[idx, :],
                                mode='lines+markers',
                                line=dict(color=colors[idx], dash ='dash'),
                                marker_symbol='square',
                                name=f'Viscous Entropy Gen Snap {idx}',
                                showlegend=False))
        
    if IS_EXPORT_HTML:
        fig.write_html(export_folder / f"{DATA_TYPE}_entropy_numbers.html" )
    fig.show()

if __name__ == "__main__":
    setup_logger(is_console=True, level=logging.INFO)
    main()