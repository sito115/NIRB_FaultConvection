from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
sys.path.append(str(Path(__file__).parents[1]))
from src.offline_stage import NirbDataModule
from src.utils import find_snapshot_path, plot_data
from src.pod import match_scaler


PARAMETER_SPACE = "09"
PROJECTION = "Mapped"
FIELD_NAME = "Entropy"
control_mesh_suffix = "s50_50_50_b0_4000_0_5000_-4000_0" #None
ROOT =  Path(__file__).parent.parent / "data" / PARAMETER_SPACE
assert ROOT.exists()

for suffix in tqdm(['min_max', 'mean', 'none', 'standard']):
    normalizer = match_scaler(suffix)
    training_data = np.load(find_snapshot_path(PROJECTION, suffix, FIELD_NAME, ROOT, control_mesh_suffix, "Training"))[:, -1, :]
    test_data     = np.load(find_snapshot_path(PROJECTION, suffix, FIELD_NAME, ROOT, control_mesh_suffix, "Training"))[:, -1, :]
    
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
    plot_data(data_module.training_snaps_scaled, title = f"PS{PARAMETER_SPACE} - Training scaled {suffix}",
              export_path = ROOT / "Exports" / f"Training_{FIELD_NAME}{suffix}.png")
    plot_data(data_module.test_snaps_scaled, title = f"PS{PARAMETER_SPACE} - Test scaled {suffix}",
              export_path = ROOT / "Exports" / f"Test_{FIELD_NAME}{suffix}.png")
    # plot_data(data_module.test_snaps, title = f"PS{PARAMETER_SPACE} - Test unscaled {suffix}",
    #           export_path = ROOT / "Exports" / f"Test_{suffix}_unscaled.png")
    # plot_data(data_module.training_snaps, title = f"PS{PARAMETER_SPACE} - Training unscaled {suffix}",
    #           export_path = ROOT / "Exports" / f"Training_{suffix}_unscaled.png")
