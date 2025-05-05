"""
Export last time step on control mesh.
"""
from typing import List
import logging
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
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')




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
    
    spacing = (50, 50, 50)
    
    original_comsol_bounds = COMSOL_VTU(vtu_files[0]).mesh.bounds
    x_min, x_max, y_min, y_max, z_min, z_max = original_comsol_bounds
    # Reduce bounds of control mesh to be within source mesh to avoid interpolation errors.
    # x_min = int(x_min) + spacing[0]
    # x_max = int(x_max) - spacing[0]
    # y_min = int(y_min) + spacing[1]
    # y_max = int(y_max) - spacing[1]
    # z_min = int(z_min) + spacing[2] # z_min is negativ
    # z_max = int(z_max) - spacing[2] # z_max is positive
    
    control_mesh = create_control_mesh((x_min, x_max,
                                        y_min, y_max,
                                        z_min, z_max),
                                        spacing)
    
    logging.debug(f"Old bounds of control mesh: {control_mesh.bounds}")
    bbox = pv.Box(np.trunc(original_comsol_bounds))
    control_mesh = control_mesh.clip_box(bbox, invert=False)
    logging.debug(f"New bounds of control mesh: {control_mesh.bounds}")
            
    spacing_str = '_'.join(f"{x:.0f}" for x in spacing)
    bounds_str = '_'.join(f"{x:.0f}" for x in (x_min, x_max, y_min, y_max, z_min, z_max))
    export_folder = export_folder / f"s{spacing_str}_b{bounds_str}"
    export_folder.mkdir(exist_ok=True)
    for vtu_path in tqdm(vtu_files, total=len(vtu_files), desc="Mapping last time step on control mesh"):
        logging.debug(f"Mapping {vtu_path.name}")
        comsol_data = COMSOL_VTU(vtu_path)
        # Delete fields of no interest to reduce file size after interpolation
        comsol_data = delete_comsol_fields(comsol_data, FIELDS_TO_EXPORT)
        for i in range(len(comsol_data.times) - 1): # delete every time step except last one (-1)
            comsol_data.mesh.point_data.remove(comsol_data.format_field("Temperature", i))
        mapped : pv.ImageData = map_on_control_mesh(comsol_data.mesh, control_mesh)
        field_name = comsol_data.mesh.point_data.keys()[-1]
        
        validity_array = mapped.point_data['vtkValidPointMask']  # Replace with your actual array name
        invalid_mask = validity_array == 0  # This assumes NaN marks invalid points
        n_invalid_points = np.sum(invalid_mask)
        if n_invalid_points > 0:
            logging.warning(f"\t Found {n_invalid_points} invalid points in {vtu_path.name}")
            for idx_inavalid, is_invalid in enumerate(invalid_mask):
                if not is_invalid:
                    continue
                invalid_point = mapped.points[idx_inavalid]
                if np.round(invalid_point[-1]) == np.round(comsol_data.mesh.bounds.z_min):
                    closest_temperature = np.max(comsol_data.mesh.point_data[field_name])
                else:
                    closest_point_id = comsol_data.mesh.find_closest_point(invalid_point, 3)
                    closest_temperature = np.mean(comsol_data.mesh.point_data[field_name][closest_point_id])
                mapped.point_data[field_name][idx_inavalid] = closest_temperature
        
        # assert np.min(mapped.point_data["vtkValidPointMask"]) > 0, f"Error in interpolation in file {vtu_path.name}"
        tolerance = 1
        assert np.round(np.min(mapped.point_data[field_name]), tolerance) >= np.round(np.min(comsol_data.mesh.point_data[field_name]), tolerance), f"Mapped minimum ({np.min(mapped.point_data[field_name]):.3e}) is smaller than original minimum ({np.min(comsol_data.mesh.point_data[field_name]):.2e})"
        assert np.round(np.max(mapped.point_data[field_name]), tolerance) <= np.round(np.max(comsol_data.mesh.point_data[field_name]), tolerance), f"Mapped maximum ({np.max(mapped.point_data[field_name]):.3e}) is bigger than original minimum  ({np.max(comsol_data.mesh.point_data[field_name]):.2e})"
        
        mapped.point_data.remove("vtkValidPointMask")
        for key, value in comsol_data.mesh.field_data.items(): # Transfer meta data (Parameters, Idx, SimTime)
            mapped.field_data[key] = value
        mapped.save(export_folder / f"{vtu_path.stem}_s{spacing_str}_b{bounds_str}.vtu") # VTK ImageData (best for structured image data)
        
    total_size = sum(file.stat().st_size for file in export_folder.iterdir() if file.suffix == ".vtu")  / (1024 * 1024)
    print(f"Total size of all mapped .vtu files: {total_size} MB")

if __name__ == "__main__":
    main()