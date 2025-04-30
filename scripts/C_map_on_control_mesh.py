from ast import List
import sys
import numpy as np
import pyvista as pv
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from scr.utils import (create_control_mesh,
                       map_on_control_mesh,
                       delete_comsol_fields)
from scr.comsol_module.comsol_classes import COMSOL_VTU
from tqdm import tqdm





def main():   
    ROOT = Path(__file__).parents[1]
    PARAMETER_SPACE = "03"
    DATA_TYPE = "Training"
    FIELDS_TO_EXPORT : List[str] = ["Temperature"]

    data_folder = Path(ROOT / "data" / PARAMETER_SPACE /  "Training_Original") # data_type) #"Truncated") # data_type)    
    assert data_folder.exists(), f"Data folder {data_folder} does not exist."
    export_folder =  data_folder.parent / (DATA_TYPE + "Mapped")
    export_folder.mkdir(exist_ok=True)
    assert export_folder.exists(), f"Export folder {export_folder} does not exist."
    assert str(data_folder.absolute()) != str(export_folder.absolute()), "Import and Export from same folder not allowed"

    vtu_files = sorted([path for path in data_folder.iterdir() if path.suffix == ".vtu"])
    
    x_min, x_max, y_min, y_max, z_min, z_max = COMSOL_VTU(vtu_files[0]).mesh.bounds
    spacing = (50, 50, 50)
    
    # Reduce bounds of control mesh to be within source mesh to avoid interpolation errors.
    x_min = int(x_min) + spacing[0]
    x_max = int(x_max) - spacing[0]
    y_min = int(y_min) + spacing[1]
    y_max = int(y_max) - spacing[1]
    z_min = int(z_min) + spacing[2] # z_min is negativ
    # z_max = int(z_max) - spacing[2] # z_max is positive
    
    control_mesh = create_control_mesh((x_min, x_max,
                                        y_min, y_max,
                                        z_min, z_max),
                                       spacing)
    
    for vtu_path in tqdm(vtu_files, total=len(vtu_files), desc="Mapping files on control mesh"):
        comsol_data = COMSOL_VTU(vtu_path)
        idx = comsol_data.mesh.field_data["Idx"]
        # Delete fields of no interest to reduce file size after interpolation
        comsol_data = delete_comsol_fields(comsol_data, FIELDS_TO_EXPORT)
        for i in range(len(comsol_data.times) - 1): # delete every time step except last one (-1)
            comsol_data.mesh.point_data.remove(comsol_data.format_field("Temperature", i))
        mapped : pv.ImageData = map_on_control_mesh(comsol_data.mesh, control_mesh)
        assert np.min(mapped.point_data["vtkValidPointMask"]) > 0, f"Error in interpolation in file {vtu_path.name}"
        mapped.point_data.remove("vtkValidPointMask")
        spacing_str = '_'.join(f"{x:.0f}" for x in spacing)
        bounds_str = '_'.join(f"{x:.0f}" for x in (x_min, x_max, y_min, y_max, z_min, z_max))
        mapped.save(export_folder / f"{vtu_path.stem}_s{spacing_str}_b{bounds_str}.vti") # VTK ImageData (best for structured image data)

if __name__ == "__main__":
    main()