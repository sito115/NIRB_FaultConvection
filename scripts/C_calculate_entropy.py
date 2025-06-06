from pathlib import Path
import logging
from tqdm import tqdm   
import numpy as np
import pandas as pd
import pint # noqa: F401
import sys
from plotly import graph_objects as go
sys.path.append(str(Path(__file__).parents[1]))
from src.utils import safe_parse_quantity, setup_logger
from comsol_module.comsol_classes import COMSOL_VTU
from comsol_module.entropy import caluclate_entropy_gen_number_isotherm


def main():

    ROOT = Path(__file__).parents[1] 
    PARAMETER_SPACE = "07"
    DATA_TYPE = "Training"

    import_folder = ROOT / "data" / PARAMETER_SPACE / f"{DATA_TYPE}Original"
    assert import_folder.exists(), f"Import folder {import_folder} does not exist."
    comsol_vtu_files = sorted([path for path in import_folder.rglob("*.vtu")])
    N_SNAPS = len(comsol_vtu_files)

    param_folder = ROOT / "data" / PARAMETER_SPACE / "Exports"
    param_files = sorted([path for path in param_folder.rglob(f"{DATA_TYPE}*.csv")])
    assert len(param_files) == N_SNAPS
    ureg = pint.get_application_registry()


   
    N_TIME_STEPS = len( COMSOL_VTU(comsol_vtu_files[0]).times.keys())
    TIME_STEPS = np.arange(N_TIME_STEPS) 
    N_CELLS = COMSOL_VTU(comsol_vtu_files[0]).mesh.n_cells
    entropy_gen_number_therm = np.zeros((N_SNAPS, N_TIME_STEPS))
    entropy_gen_number_visc = np.zeros((N_SNAPS, N_TIME_STEPS))
    entropy_gen_per_vol_thermal = np.zeros((N_SNAPS, N_TIME_STEPS, N_CELLS))
    entropy_gen_per_vol_visc = np.zeros_like(entropy_gen_per_vol_thermal)

    for idx_snap in tqdm(range(N_SNAPS), total = N_SNAPS):
        comsol_data = COMSOL_VTU(comsol_vtu_files[idx_snap])
        param_df = pd.read_csv(param_files[idx_snap], index_col = 0)
        param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
        lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                            param_df.loc['host_phi', "quantity_pint"] * (4.2 * ureg.watt / (ureg.meter * ureg.kelvin))
        t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
        delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
        
        model_dict = {
            'lambda_m' : lambda_therm.magnitude,
            'T0' : t0.magnitude,
            'mu0': comsol_data.get_array('Dynamic_viscosity', is_cell_data=True)[-1, :],
            'k_m': 1.5e-13,
            'delta_T' : delta_T.magnitude,
            'H' : comsol_data.mesh.bounds.z_max - comsol_data.mesh.bounds.z_min,
        }
        
        entropy_per_vol = comsol_data.calculate_total_entropy_per_vol(model_data=model_dict,
                                                    time_steps=TIME_STEPS,
                                                    is_return_as_integration=False,)
        
        entropy_gen_per_vol_thermal[idx_snap, :, :] = entropy_per_vol[:, : , 0]  # thermal entropy generation per volume
        entropy_gen_per_vol_visc[idx_snap, :, :] = entropy_per_vol[:, : , 1]
        
        entropy_integrated= comsol_data.calculate_total_entropy_per_vol(model_data=model_dict,
                                                    time_steps=TIME_STEPS,
                                                    is_return_as_integration=True,)
        
        entropy_gen_number_therm[idx_snap, :] = caluclate_entropy_gen_number_isotherm(s_total=entropy_integrated[:, 0],
                                                            L = model_dict["H"], 
                                                            lambda_m=model_dict["lambda_m"],
                                                            T_0=model_dict["T0"],
                                                            delta_T=model_dict["delta_T"],
                                                            V = comsol_data.mesh.volume,)


        entropy_gen_number_visc[idx_snap, :] = caluclate_entropy_gen_number_isotherm(s_total=entropy_integrated[:, 1],
                                                            L = model_dict["H"], 
                                                            lambda_m=model_dict["lambda_m"],
                                                            T_0=model_dict["T0"],
                                                            delta_T=model_dict["delta_T"],
                                                            V = comsol_data.mesh.volume,)


    np.save(ROOT / "data" / PARAMETER_SPACE / f"{DATA_TYPE}Original" /f"{DATA_TYPE}_entropy_gen_per_vol_thermal.npy", entropy_gen_per_vol_thermal)
    np.save(ROOT / "data" / PARAMETER_SPACE / f"{DATA_TYPE}Original" /f"{DATA_TYPE}_entropy_gen_per_vol_visc.npy", entropy_gen_per_vol_thermal)
    np.save(ROOT / "data" / PARAMETER_SPACE / f"{DATA_TYPE}Original" /f"{DATA_TYPE}_entropy_gen_number_therm.npy", entropy_gen_number_therm)
    np.save(ROOT / "data" / PARAMETER_SPACE / f"{DATA_TYPE}Original" /f"{DATA_TYPE}_entropy_gen_number_visc.npy", entropy_gen_number_visc)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(N_SNAPS), y=entropy_gen_number_therm[:, 0], mode='lines+markers', name='Thermal Entropy Generation Number'))
    fig.add_trace(go.Scatter(x=np.arange(N_SNAPS), y=entropy_gen_number_visc[:, 0], mode='lines+markers', name='Viscous Entropy Generation Number'))
    fig.show()

if __name__ == "__main__":
    setup_logger(is_console=True, level=logging.INFO)
    main()