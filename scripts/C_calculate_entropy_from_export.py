from pathlib import Path
import logging
from tqdm import tqdm   
import numpy as np
import pandas as pd
import pint # noqa: F401
import sys
from plotly import graph_objects as go
from plotly import express as px
import pyvista as pv
sys.path.append(str(Path(__file__).parents[1]))
from src.utils import safe_parse_quantity, setup_logger, read_config
from comsol_module.entropy import caluclate_entropy_gen_number_isotherm, calculate_S_therm

def main():

    ROOT = Path(__file__).parents[1] 
    PARAMETER_SPACE = "09"
    DATA_TYPE = "Training"
    IS_EXPORT_ARRAY = True
    IS_EXPORT_ENTROPY_NUMBERS = True
    IS_EXPORT_HTML = True
    spacing = 50
    config = read_config()
    control_mesh_suffix = f"s{spacing}_{spacing}_{spacing}_b0_4000_0_5000_-4000_0"

    import_folder = ROOT / "data" / PARAMETER_SPACE / f"{DATA_TYPE}Mapped" / f"{control_mesh_suffix}"
        
    assert import_folder.exists(), f"Import folder {import_folder} does not exist."
    temperatures_file = import_folder / "Exports" / f"{DATA_TYPE}_Temperature.npy"
    assert temperatures_file.exists()
    temperatures = np.load(temperatures_file)
    N_SNAPS = temperatures.shape[0]
    N_TIME_STEPS = temperatures.shape[1]
    N_POINTS = temperatures.shape[2]
    TIME_STEPS = np.arange(N_TIME_STEPS) 

    param_folder = ROOT / "data" / PARAMETER_SPACE / "Exports"
    param_files = sorted([path for path in param_folder.rglob(f"{DATA_TYPE}*.csv")])
    assert len(param_files) == N_SNAPS
    ureg = pint.get_application_registry()

    ref_comsol_data_file = [path for path in import_folder.rglob('*.vt*')][0]
    print(f"{ref_comsol_data_file}")
    ref_mesh = pv.read(ref_comsol_data_file)
    ref_mesh.clear_data()
    assert ref_mesh.n_points == N_POINTS

    if IS_EXPORT_ENTROPY_NUMBERS:
        entropy_gen_number_therm = np.zeros((N_SNAPS, N_TIME_STEPS))
    if IS_EXPORT_ARRAY:
        entropy_gen_per_vol_thermal = np.zeros((N_SNAPS, N_TIME_STEPS, N_POINTS))

    for idx_snap in tqdm(range(N_SNAPS), total = N_SNAPS):
        param_df = pd.read_csv(param_files[idx_snap], index_col = 0)
        param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
        lambda_therm = (1 - param_df.loc['host_phi', "quantity_pint"]) * param_df.loc['host_lambda', "quantity_pint"] + \
                            param_df.loc['host_phi', "quantity_pint"] * (config['lambda_f']  * ureg.watt / (ureg.meter * ureg.kelvin))
        t0      = 0.5 * (param_df.loc["T_h", "quantity_pint"] + param_df.loc["T_c", "quantity_pint"])
        delta_T = (param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"])
        H =  (ref_mesh.bounds.z_max - ref_mesh.bounds.z_min) * ureg.meter

        for idx_time, temperature in enumerate(temperatures[idx_snap]):

            ref_mesh.point_data["temp_field"] = temperature
            derivative  = ref_mesh.compute_derivative("temp_field", preference = "point")
            temp_gradient = derivative.point_data['gradient'] * ureg.kelvin / ureg.meter
            entropy_per_vol = calculate_S_therm(lambda_m=lambda_therm,
                                                T_0=t0,
                                                temp_gradient=temp_gradient)
            

            
            
            if IS_EXPORT_ARRAY:
                entropy_gen_per_vol_thermal[idx_snap, idx_time, :] = entropy_per_vol.magnitude  # thermal entropy generation per volume

            
            if IS_EXPORT_ENTROPY_NUMBERS:
                ref_mesh.point_data["temp_field"] = entropy_per_vol
                integrated = ref_mesh.integrate_data()
                s0_total = integrated.point_data['temp_field'][0] * ureg.watt / ureg.kelvin
                logging.debug(f"{np.max(s0_total)=:.3f}")
                entropy_number = caluclate_entropy_gen_number_isotherm(s_total=s0_total,
                                                                        L=H,
                                                                        lambda_m=lambda_therm,
                                                                        T_0=t0,
                                                                        V=ref_mesh.volume * ureg.meter**3,
                                                                        delta_T=delta_T)
                assert entropy_number.check(['dimensionless']) 
                entropy_gen_number_therm[idx_snap, idx_time] = entropy_number.magnitude
                logging.debug(f"{entropy_number=}")


    if IS_EXPORT_ENTROPY_NUMBERS:
        np.save(import_folder /  "Exports" / f"{DATA_TYPE}_entropy_gen_number_therm.npy", entropy_gen_number_therm)
    if IS_EXPORT_ARRAY:
        np.save(import_folder / "Exports" /f"{DATA_TYPE}_entropy_gen_per_vol_thermal.npy", entropy_gen_per_vol_thermal)

    if IS_EXPORT_ENTROPY_NUMBERS:
        fig = go.Figure()
        colors = px.colors.sample_colorscale("jet", [n/(N_SNAPS) for n in range(N_SNAPS)])
        for idx in range(N_SNAPS):
            fig.add_trace(go.Scatter(x=TIME_STEPS,
                                    y=entropy_gen_number_therm[idx, :],
                                    line=dict(color=colors[idx]),
                                    marker_symbol='circle',
                                    mode='lines+markers',
                                    name=f'{idx:03d}',
                                    showlegend=True))

        if IS_EXPORT_HTML:
            fig.write_html(import_folder / "Exports" / f"{DATA_TYPE}_entropy_numbers.html" )
        fig.show()

if __name__ == "__main__":
    setup_logger(is_console=True, level=logging.INFO)
    main()