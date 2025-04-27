from pathlib import Path
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ComsolClasses.comsol_classes import COMSOL_VTU
from ComsolClasses.helper import calculate_normal
from pint import UnitRegistry, Quantity
import ast
import numpy as np
from tqdm import tqdm


def convert_str_to_pint(value) -> Quantity:
    """ Converts Comsol parameters to pint.Quantities.

    Args:
        value (_type_): Comsol parameter value (format is "Value[Unit]")

    Returns:
        _type_: _description_
    """    
    ureg = UnitRegistry()
    try:
        if "[" in value:
            splitted_value = value.split("[") 
            numeric_value = float(splitted_value[0])
            unit = splitted_value[1].split("]")[0]
            return ureg.Quantity(numeric_value, unit).to_base_units()
        else:
            return float(value) * ureg("dimensionless")
    except ValueError:
        return value  # Return the original value if conversion fails


def main():
    """Extract "EXPORT_FIELD" from  multiple vtu-Files and save it as npy-File.
    Optionally create mp4 movies of simulations. 
    """    
    IS_EXPORT_MP4 = False
    EXPORT_FIELD = "Temperature"
    IS_EXPORT_MINMAX_TEMP = True
    IS_EXPORT_NPY = True
    IS_EXPORT_DF = False
    
    ROOT = Path(__file__).parent.parent
    VERSION = "02"
    data_type = "Test"
    data_folder = Path(ROOT / "Snapshots" / VERSION / data_type)
    
    assert data_folder.exists(), f"Data folder {data_folder} does not exist."
    export_folder =  data_folder.parent / "Exports"
    assert export_folder.exists(), f"Export folder {export_folder} does not exist."
  
    vtu_files = sorted([path for path in data_folder.iterdir() if path.suffix == ".vtu"])
    
    # extract time steps and points from first simulation
    N_TIME_STEPS      = len(COMSOL_VTU(vtu_files[0]).times)
    N_POINTS          = COMSOL_VTU(vtu_files[0]).mesh.points.shape[0]
    temperatures      = np.zeros((len(vtu_files), N_TIME_STEPS, N_POINTS))
    temperatures_diff = np.zeros_like(temperatures)
    sim_times = np.zeros((len(vtu_files), ))
    
    for idx, vtu_file in tqdm(enumerate(vtu_files), total=len(vtu_files), desc="Reading COMSOL files"):
        comsol_data = COMSOL_VTU(vtu_file)
        sim_time = comsol_data.mesh.field_data['SimTime'][0]
        sim_times[idx] = sim_time
        parameters = comsol_data.mesh.field_data['Parameters']
        parameters = ast.literal_eval(parameters[0])
        parameters_pint : dict[Quantity] = {}
        for key, value in parameters.items():
            parameters_pint[key] = convert_str_to_pint(value)
        comsol_data.parameters = parameters_pint
        
        dip = parameters_pint['dip'].to('degree').magnitude
        strike = parameters_pint['strike'].to('degree').magnitude
        t_c = parameters_pint['T_c'].to('K').magnitude
        t_h = parameters_pint['T_h'].to('K').magnitude
        t_grad = (t_h - t_c) / parameters_pint['H'].to('m').magnitude

        if IS_EXPORT_MP4:
            normal = calculate_normal(dip, strike)
            kwargs={'normal' : -np.array(normal),
                'origin' : comsol_data.mesh.center,
                'movie_field' : EXPORT_FIELD + "-T0",
                'is_diff' : False,
                'is_ind_cmap':True,
                't_grad': {'t0': t_c,
                          't_grad': t_grad}}
            comsol_data.export_mp4_movie(field='Temperature',
                                        mp4_file=export_folder / f"{comsol_data.vtu_path.stem}_{EXPORT_FIELD}_Tot.mp4",
                                        **kwargs)
            
            if IS_EXPORT_DF:
                df = pd.DataFrame().from_dict(parameters_pint,
                                            orient='index')
                df.columns = ['quantity'] 
                                            # columns=['Parameter', "Value"])
                df.index = df.index.astype(str)
                df.sort_index(key=lambda x : x.str.lower()).to_csv(export_folder / f"{comsol_data.vtu_path.stem}_parameters.csv")
            
        if IS_EXPORT_MINMAX_TEMP: # min max temperatures
            temp_array = comsol_data.get_array('Temperature')
            temp_diff = temp_array - (t_c - t_grad * comsol_data.mesh.points[:,-1])
            time_len = temp_array.shape[0]
            temperatures[idx, :time_len, :] = temp_array
            temperatures_diff[idx, :time_len, :] = temp_diff

    if IS_EXPORT_NPY:
        np.save(export_folder / f"{data_type}_sim_times.npy", np.array(sim_times))
        np.save(export_folder / f"{data_type}_temperatures.npy", temperatures      )
        np.save(export_folder / f"{data_type}_temperatures_diff.npy", temperatures_diff )


def reduce_to_binary():
    ROOT = Path(__file__).parent.parent
    VERSION = "01"
    data_type = "Test"
    data_folder = Path(ROOT / "Snapshots" / VERSION / data_type)
    
    assert data_folder.exists(), f"Data folder {data_folder} does not exist."
    export_folder =  data_folder.parent / "Truncated"
    assert export_folder.exists(), f"Export folder {export_folder} does not exist."
    vtu_files = sorted([path for path in data_folder.iterdir() if path.suffix == ".vtu"])
    for idx, vtu_file in tqdm(enumerate(vtu_files), total=len(vtu_files), desc="Reading COMSOL files"):
        temp_data = COMSOL_VTU(vtu_file)
        fields_2_delete = temp_data.exported_fields
        fields_2_delete.remove("Temperature")
        for field in fields_2_delete:
            temp_data.delete_field(field)
        temp_data.mesh.save(export_folder / vtu_file.name)
    



if __name__ == "__main__":
    reduce_to_binary()
    main()