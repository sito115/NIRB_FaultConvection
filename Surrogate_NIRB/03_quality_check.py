from pathlib import Path
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ComsolClasses.comsol_classes import COMSOL_VTU
from ComsolClasses.helper import calculate_normal
from pint import UnitRegistry, Quantity
import ast
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.express as px

def convert_str_to_pint(value):
    ureg = UnitRegistry()
    try:
        if "[" in value:
            splitted_value = value.split("[") 
            numeric_value = float(splitted_value[0])
            unit = splitted_value[1].split("]")[0]
            return ureg.Quantity(numeric_value, unit)
        else:
            return float(value) * ureg("dimensionless")
    except ValueError:
        return value  # Return the original value if conversion fails


def main():
    IS_EXPORT_MP4 = False
    EXPORT_FIELD = "Temperature"
    IS_EXPORT_MINMAX_TEMP = True
    
    data_folder = Path("Snapshots/01/Training")
    assert data_folder.exists(), f"Data folder {data_folder} does not exist."
    export_folder =  data_folder.parent / "Exports"
    assert export_folder.exists(), f"Data folder {export_folder} does not exist."
  
    vtu_files = sorted([path for path in data_folder.iterdir() if path.suffix == ".vtu"])
    
    fig = go.Figure()
    colors = px.colors.sample_colorscale("jet", [n/(len(vtu_files)) for n in range(len(vtu_files))])
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
        host_k = parameters_pint['host_k'].to('m^2').magnitude
        t_grad = (t_h - t_c) / parameters_pint['H'].to('m').magnitude

        if IS_EXPORT_MP4:
            normal = calculate_normal(dip, strike)
            kwargs={'normal' : -np.array(normal),
                'origin' : comsol_data.mesh.center,
                'movie_field' : EXPORT_FIELD + "-T0",
                'is_diff' : True,
                'is_ind_cmap':True,
                't_grad': {'t0': t_c,
                            't_grad': t_grad}}
            comsol_data.export_mp4_movie(field='Temperature',
                                        mp4_file=export_folder / f"{comsol_data.vtu_path.stem}_{EXPORT_FIELD}.mp4",
                                        **kwargs)
        
        if IS_EXPORT_MINMAX_TEMP:
            temp_array = comsol_data.get_array('Temperature')
            # customdata=np.stack(([t_h] * len(comsol_data.times), [host_k] * len(comsol_data.times)) , axis=-1),
            input_args = {'x' : list(comsol_data.times.values()),
                        #   'customdata' : customdata,
                        #   'hovertemplate' : "Time: %{x:.2e}<br>Temperature: %{y:.2f}<br>T_h: %{customdata[0]:.2f}<br><br>host_k: %{customdata[1]:.2e}<br>",
                          'legendgroup' : f"{comsol_data.vtu_path.stem}",
                          'line' : dict(color=colors[idx])}
            fig.add_trace(go.Scatter(y=np.max(temp_array, axis=1), **input_args))
            fig.add_trace(go.Scatter(y=np.min(temp_array, axis=1), **input_args))

        df = pd.DataFrame().from_dict(parameters_pint, orient='index')
        df.index = df.index.astype(str)
        df.sort_index(key=lambda x : x.str.lower()).to_csv(export_folder / f"{comsol_data.vtu_path.stem}_parameters.csv")
    
    fig.update_layout(
        title="Min/Max Temperature",
        xaxis_title="Time (s)",
            xaxis=dict(\
            tickformat = '.2e',
            showexponent = 'all',
            exponentformat = 'e'),
        yaxis_title="Temperature (K)",
        showlegend=False,
    )
    # fig = go.Figure(data=[go.Histogram(x=sim_times)])
    fig.show()
    
if __name__ == "__main__":
    main()