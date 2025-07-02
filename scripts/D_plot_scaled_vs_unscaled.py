from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
sys.path.append(str(Path(__file__).parents[1]))
from src.offline_stage import NirbDataModule
from src.utils import find_snapshot_path, plot_data
from src.pod import match_scaler


def plot_data_as_html(data: np.ndarray, title:str , path: Path):
    n_points = np.arange(data.shape[1])
    fig = go.Figure()
    for idx, array in enumerate(data):
        if not np.any(array < -9):
            continue
        # if not idx == 34:
        #     continue
        fig.add_trace(
            go.Scatter(
             x = n_points,
             y = array,
             name = f"{idx:03d}",   
             mode="lines",
             line_shape = "hv",
            #  hoverinfo='name' ,
            line=dict(width=1)      # thinner lines reduce rendering load
            )
        )
    fig.update_layout(
        title = title,
        xaxis_title='Point ID',
        yaxis_title='Values'   ,
        template='plotly_white', 
    )
    fig.write_html(path)
    # fig.write_image(path.with_suffix('.png'))
    # fig.show()


PARAMETER_SPACE = "09"
PROJECTION = "Mapped"
FIELD_NAME = "Entropy"
spacing = 50
control_mesh_suffix = f"s{spacing}_{spacing}_{spacing}_b0_4000_0_5000_-4000_0" #None
ROOT =  Path(__file__).parent.parent / "data" / PARAMETER_SPACE
assert ROOT.exists()

for suffix in tqdm(['none' ,'min_max', 'mean',  'standard']):
    normalizer = match_scaler(suffix)
    training_data = np.load(find_snapshot_path(PROJECTION, suffix, FIELD_NAME, ROOT, control_mesh_suffix, "Training"))[:, -1, :]
    test_data     = np.load(find_snapshot_path(PROJECTION, suffix, FIELD_NAME, ROOT, control_mesh_suffix, "Test"))[:, -1, :]
    
    if PARAMETER_SPACE == "09":
        zero_crossings = np.load(ROOT /"Exports/Training_zero_crossings.npy")
        mask = zero_crossings != 6
        training_data = training_data[mask, :]
    
    print(f"{normalizer=}")
    data_module = NirbDataModule(basis_func_mtrx=None,
                                training_param=None,
                                test_param=None,
                                standardizer_features=None,
                                training_snaps=training_data,
                                test_snaps=test_data,
                                auto_process=False,
                                normalizer=normalizer,
                                batch_size=-1
                                )
    
    # if suffix.lower() == "none":
    #     plot_data_as_html(data = data_module.training_snaps_scaled,
    #                     title = f"PS{PARAMETER_SPACE} - Training scaled {suffix}",
    #                     path = ROOT / "Exports" / f"Training_{FIELD_NAME}{suffix}.html")
        
    plot_data(data_module.training_snaps_scaled, title = f"PS{PARAMETER_SPACE} - Training scaled {suffix}",
              export_path = ROOT / "Exports" / f"Training_{FIELD_NAME}{suffix}.png")
    plot_data(data_module.test_snaps_scaled, title = f"PS{PARAMETER_SPACE} - Test scaled {suffix}",
              export_path = ROOT / "Exports" / f"Test_{FIELD_NAME}{suffix}.png")
    

    
    
    # plot_data(data_module.test_snaps, title = f"PS{PARAMETER_SPACE} - Test unscaled {suffix}",
    #           export_path = ROOT / "Exports" / f"Test_{suffix}_unscaled.png")
    # plot_data(data_module.training_snaps, title = f"PS{PARAMETER_SPACE} - Training unscaled {suffix}",
    #           export_path = ROOT / "Exports" / f"Training_{suffix}_unscaled.png")
