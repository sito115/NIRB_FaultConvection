"""
Export last time step on control mesh.
"""
import logging
import sys
import numpy as np
import pyvista as pv
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from scr.utils import (create_control_mesh,
                       map_on_control_mesh,
                       delete_comsol_fields,
                       inverse_distance_weighting)
from scr.comsol_module.comsol_classes import COMSOL_VTU
from tqdm import tqdm
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


def handle_invalid_point_mask(target_point: np.ndarray,
                              source_mesh: pv.DataSet,
                              field_name: str) -> float:
    
    K_NEIGHBOURS = 3
    if np.isclose(np.round(target_point[-1]), np.round(source_mesh.bounds.z_min), atol=5):
        closest_temperature = np.max(source_mesh.point_data[field_name])
    else:
        closest_point_ids = source_mesh.find_closest_point(target_point, K_NEIGHBOURS)
        neighbour_points = source_mesh.points[closest_point_ids]
        neighbbour_values = source_mesh.point_data[field_name][closest_point_ids]
        closest_temperature = inverse_distance_weighting(target_point,
                                                         neighbour_points,
                                                         neighbbour_values)
    return closest_temperature


def main():   
    ROOT = Path(__file__).parents[1]
    PARAMETER_SPACE = "03"
    DATA_TYPE = "Training"
    FIELD_TO_EXPORT : str = "Temperature"
    SPACING = (50, 50, 50) # dx, dy, dz

    data_folder = Path(ROOT / "data" / PARAMETER_SPACE /  "Training_Original") # data_type) #"Truncated") # data_type)    
    assert data_folder.exists(), f"Data folder {data_folder} does not exist."
    export_folder =  data_folder.parent / (DATA_TYPE + "Mapped")
    export_folder.mkdir(exist_ok=True)
    assert export_folder.exists(), f"Export folder {export_folder} does not exist."
    assert str(data_folder.absolute()) != str(export_folder.absolute()), "Import and Export from same folder not allowed"

    # Find vtu files of original snapshots
    vtu_files = sorted([path for path in data_folder.iterdir() if path.suffix == ".vtu"])
    original_comsol_bounds = COMSOL_VTU(vtu_files[0]).mesh.bounds
    
    control_mesh = create_control_mesh(original_comsol_bounds,
                                       SPACING)
    
    # Clip bounds of control mesh to avoid interpolation errors
    logging.debug(f"Old bounds of control mesh: {control_mesh.bounds}")
    bbox = pv.Box(np.trunc(original_comsol_bounds)) # converts image data to unstructered grid
    control_mesh = control_mesh.clip_box(bbox, invert=False)
    logging.debug(f"New bounds of control mesh: {control_mesh.bounds}")
    
    # create export folder with spacing and bounds information
    spacing_str = '_'.join(f"{x:.0f}" for x in SPACING)
    bounds_str = '_'.join(f"{x:.0f}" for x in (control_mesh.bounds))
    export_folder = export_folder / f"s{spacing_str}_b{bounds_str}"
    export_folder.mkdir(exist_ok=True)
    
    
    for vtu_path in tqdm(vtu_files, total=len(vtu_files), desc="Mapping last time step on control mesh"):
        logging.debug(f"Mapping {vtu_path.name}")
        comsol_data = COMSOL_VTU(vtu_path)
        
        # Delete fields of no interest to reduce file size after interpolation
        comsol_data = delete_comsol_fields(comsol_data, [FIELD_TO_EXPORT])
        for i in range(len(comsol_data.times) - 1): # delete every time step except last one (-1)
            comsol_data.mesh.point_data.remove(comsol_data.format_field(FIELD_TO_EXPORT, i))
        mapped : pv.ImageData = map_on_control_mesh(comsol_data.mesh, control_mesh)
        field_name = comsol_data.mesh.point_data.keys()[-1]
        
        validity_array = mapped.point_data['vtkValidPointMask']  
        invalid_indices = np.where(validity_array == 0)[0]
        
        # If there are invalid points, loop over them 
        if invalid_indices.size > 0:
            logging.warning(f"\t Found {invalid_indices.size} invalid points in {vtu_path.name}")
            for idx in invalid_indices:
                invalid_point = mapped.points[idx]
                interpolated_value = handle_invalid_point_mask(invalid_point, comsol_data.mesh, field_name)
                mapped.point_data[field_name][idx] = interpolated_value

        tolerance = 1
        assert np.round(np.min(mapped.point_data[field_name]), tolerance) >= np.round(np.min(comsol_data.mesh.point_data[field_name]), tolerance), f"Mapped minimum ({np.min(mapped.point_data[field_name]):.3e}) is smaller than original minimum ({np.min(comsol_data.mesh.point_data[field_name]):.2e})"
        assert np.round(np.max(mapped.point_data[field_name]), tolerance) <= np.round(np.max(comsol_data.mesh.point_data[field_name]), tolerance), f"Mapped maximum ({np.max(mapped.point_data[field_name]):.3e}) is bigger than original minimum  ({np.max(comsol_data.mesh.point_data[field_name]):.2e})"
        assert np.sum(np.isnan(mapped.point_data[field_name])) == 0, "Found nan-values in mapped mesh"
        mapped.point_data.remove("vtkValidPointMask")
        
        # Transfer meta data (Parameters, Idx, SimTime)
        for key, value in comsol_data.mesh.field_data.items(): 
            mapped.field_data[key] = value
        mapped.save(export_folder / f"{vtu_path.stem}_s{spacing_str}_b{bounds_str}.vtu") # VTK ImageData (best for structured image data)
    
    
    total_size = sum(file.stat().st_size for file in export_folder.iterdir() if file.suffix == ".vtu")  / (1024 * 1024)
    print(f"Total size of all mapped .vtu files: {total_size} MB")

if __name__ == "__main__":
    main()