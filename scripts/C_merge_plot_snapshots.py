"""
This script extracts and combines individual snapshots (exported from COMSOL as vtu files in "B_compute_snapshots.py") into a single file (either in npy or joblib format).
Additionally, it allows mapping each snapshot onto a control mesh before export.
The script also provides the option to generate MP4 movies with a cut along the fault plane.
"""
from pathlib import Path
import pandas as pd
import ast
import numpy as np
import pint
from tqdm import tqdm
import sys
from joblib import dump 
sys.path.append(str(Path(__file__).parents[1]))
from comsol_module.comsol_classes import COMSOL_VTU
from comsol_module.helper import calculate_normal
from src.utils import (load_pint_data,
                       format_quantity,
                       convert_str_to_pint)


def main():
    """Extract "EXPORT_FIELD" from  multiple vtu-Files and save it as npy-File.
    Optionally create mp4 movies of simulations. 
    """    
    IS_EXPORT_MP4 = True           # Export MP4 movies
    EXPORT_FIELD = "Temperature"   # Which field to save 
    IS_EXPORT_NPY = False           # export fields as npy, to use when n_points are the SAME for all vtu files
    IS_EXPORT_JOBLIB = False       # export fields as joblib, to use n_points are DIFFERENT for all vtu files 
    IS_EXPORT_DF = False           # export parameters in mesh.field_data as csv
    
    ROOT = Path(__file__).parents[1]
    PARAMETER_SPACE = "07"
    DATA_TYPE = "Test"
    # data_folder = Path(ROOT / "data" / PARAMETER_SPACE /  "TestMapped" / "s100_100_100_b0_4000_0_5000_-4000_-0") # data_type) #"Truncated") # data_type)
    # data_folder = ROOT / "data" / PARAMETER_SPACE /  f"{DATA_TYPE}Mapped" / "s100_100_100_b0_4000_0_5000_-4000_0"
    data_folder = ROOT / "data" / PARAMETER_SPACE /  f"{DATA_TYPE}Original"
    
    assert DATA_TYPE.lower() in str(data_folder).lower()
    assert data_folder.exists(), f"Data folder {data_folder} does not exist."
    # export_folder = data_folder.joinpath("Exports") # ROOT / "data" / PARAMETER_SPACE / "TrainingMapped" 
    # export_folder = data_folder / "Exports"
    # export_folder.mkdir(exist_ok=True)
    export_folder = Path().cwd() / f"data/{PARAMETER_SPACE}/Exports"
    assert export_folder.exists(), f"Export folder {export_folder} does not exist."
    vtu_files = sorted([path for path in data_folder.iterdir() if (path.suffix in [".vtu", ".vti"] and DATA_TYPE.lower() in path.stem.lower())])
    assert len(vtu_files) > 0
    N_SNAPS = len(vtu_files)
    # N_SNAPS = 300
    
    parameter_file = ROOT / "data" / PARAMETER_SPACE / f"{DATA_TYPE.lower()}_samples.csv"
    assert parameter_file.exists()
    pint_parameters_df = load_pint_data(parameter_file)
    assert len(pint_parameters_df) == N_SNAPS 
    free_parameter_names = pint_parameters_df.columns
    
    # extract time steps and points from first simulation
    sim_times = np.zeros((N_SNAPS, ))
    N_POINTS          = COMSOL_VTU(vtu_files[0]).mesh.points.shape[0]
    N_TIME_STEPS      = len(COMSOL_VTU(vtu_files[0]).times)
        
    if IS_EXPORT_NPY:
        temperatures      = np.zeros((N_SNAPS, N_TIME_STEPS, N_POINTS))
        temperatures_diff = np.zeros_like(temperatures)
    if IS_EXPORT_JOBLIB:
        temperatures = [np.array([]) for _ in range(N_SNAPS)]
        temperatures_diff =  [np.array([]) for _ in range(N_SNAPS)]
    
    for _ , vtu_file in tqdm(enumerate(vtu_files), total=len(vtu_files), desc="Reading COMSOL files"):
        idx = int(vtu_file.stem.split("_")[1])
        comsol_data = COMSOL_VTU(vtu_file)
        sim_time = comsol_data.mesh.field_data['SimTime'][0]
        sim_times[idx] = sim_time
        parameters = comsol_data.mesh.field_data['Parameters']
        parameters = ast.literal_eval(parameters[0])
        parameters_pint : dict[pint.Quantity] = {}
        for key, value in parameters.items():
            parameters_pint[key] = convert_str_to_pint(value)
        comsol_data.parameters = parameters_pint
        t_c = parameters_pint['T_c'].to('K').magnitude
        t_h = parameters_pint['T_h'].to('K').magnitude
        t_grad = (t_h - t_c) / parameters_pint['H'].to('m').magnitude
        
        ### Check that Parameters in field data are equal to parameters stored in data frame
        for free_parameter in free_parameter_names:
            field_value = parameters_pint[free_parameter]
            df_value = pint_parameters_df.loc[idx, free_parameter]
            assert field_value.units == df_value.units, f"Unit mismatch: {field_value.units} vs {df_value.units}"    
            assert np.isclose( field_value.magnitude, df_value.magnitude, rtol=1e-10)
            print(f"Parameter {free_parameter} is equal in field data and unit: {field_value} vs {df_value}")
            
        if IS_EXPORT_MP4:
            dip = parameters_pint['dip'].to('degree').magnitude
            strike = parameters_pint['strike'].to('degree').magnitude
            param_string = "\n".join([
                            f"{col} = {format_quantity(para, number_format='.4g')}"
                            for col, para in pint_parameters_df.loc[idx].items()
                            ])
            normal = calculate_normal(dip, strike)
            kwargs={'normal' : -np.array(normal),
                'origin' : comsol_data.mesh.center,
                'movie_field' : EXPORT_FIELD + "-T0",
                'is_diff' : True,
                'is_ind_cmap':True,
                'param_string' : param_string,
                'plot_last_frame' : True,
                'title_string' : f"{DATA_TYPE} {idx:03d} - ",
                't_grad': {'t0': t_c,
                          't_grad': t_grad}}
            comsol_data.export_mp4_movie(field='Temperature',
                                        mp4_file=export_folder / f"{comsol_data.vtu_path.stem}_{EXPORT_FIELD}_diff_{kwargs['is_diff']:d}.mp4",
                                        **kwargs)
            
        if IS_EXPORT_DF:
            df = pd.DataFrame().from_dict(parameters_pint,
                                        orient='index')
            df.columns = ['quantity'] 
                                        # columns=['Parameter', "Value"])
            df.index = df.index.astype(str)
            df.sort_index(key=lambda x : x.str.lower()).to_csv(export_folder / f"{comsol_data.vtu_path.stem}_parameters.csv")
            
        if IS_EXPORT_NPY or IS_EXPORT_JOBLIB: # min max temperatures differences to initial state (pure conduction)
            temp_array = comsol_data.get_array('Temperature')
            temp_diff = temp_array - (t_c - t_grad * comsol_data.mesh.points[:,-1])
            
            if IS_EXPORT_NPY:
                time_len = temp_array.shape[0]
                temperatures[idx, :time_len, :] = temp_array
                temperatures_diff[idx, :time_len, :] = temp_diff
            
            if IS_EXPORT_JOBLIB:
                temp_diff = temp_array - (t_c - t_grad * comsol_data.mesh.points[:,-1])
                temperatures[idx] = temp_array
                temperatures_diff[idx] = temp_diff

    if IS_EXPORT_NPY:
        np.save(export_folder / f"{DATA_TYPE}_temperatures.npy", temperatures)
        np.save(export_folder / f"{DATA_TYPE}_temperatures_minus_tgrad.npy", temperatures_diff )
        total_size = sum(file.stat().st_size for file in export_folder.iterdir() if file.suffix == ".npy")  / (1024 * 1024)
        print(f"Total size of all mapped .vti files: {total_size} MB")
        
    if IS_EXPORT_JOBLIB:
        dump(temperatures,export_folder / f"{DATA_TYPE}_temperatures.joblib")
        dump(temperatures_diff, export_folder / f"{DATA_TYPE}_temperatures_minus_tgrad.joblib")
        print("Joblib export successfull")
        total_size = sum(file.stat().st_size for file in export_folder.iterdir() if file.suffix == ".joblib")  / (1024 * 1024)
        print(f"Total size of all mapped .vti files: {total_size} MB")
        
    np.save(export_folder / f"{DATA_TYPE}_sim_times.npy", np.array(sim_times))


if __name__ == "__main__":
    main()