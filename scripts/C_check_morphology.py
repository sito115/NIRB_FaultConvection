import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import time
sys.path.append(str(Path(__file__).parents[1]))
from comsol_module.comsol_classes import COMSOL_VTU


ROOT = Path(__file__).parents[1]
PARAMETER_SPACE = "10"
FIELD_NAME = "Temperature"
DATA_TYPE = "Test"
IS_LOAD_NPY : bool = True
spacing = 50
control_mesh_suffix = f"s{spacing}_{spacing}_{spacing}_b0_4000_0_5000_-4000_0"

if control_mesh_suffix is None:
    import_folder = ROOT / "data" / PARAMETER_SPACE / (DATA_TYPE + "Original")
    assert import_folder.exists()
    comsol_data = COMSOL_VTU(import_folder / f"{DATA_TYPE}_000.vtu")
else:
    import_folder = ROOT / "data" / PARAMETER_SPACE / (DATA_TYPE + "Mapped") / control_mesh_suffix / "Exports"
    assert import_folder.exists(), f"{import_folder}"
    comsol_data = COMSOL_VTU(import_folder.parent / f"{DATA_TYPE}_000_{control_mesh_suffix}.vtk", is_clean_mesh=False)

export_folder = ROOT / "data" / PARAMETER_SPACE / "Exports" / "Zero-Crossings"
export_folder.mkdir(exist_ok=True)

temperatures_diff = np.load(import_folder  / f"{DATA_TYPE}_{FIELD_NAME}_minus_tgrad.npy" )
N_SNAPS = len(temperatures_diff)

comsol_data.mesh.clear_data()

bounds = comsol_data.mesh.bounds
start = [bounds[1] / 2, bounds[2], bounds[4] / 2]
end = [bounds[1]/ 2, bounds[3],  bounds[4] / 2]

plt.ion()
resolution = 100
fig, ax = plt.subplots()
line, = ax.plot(np.arange(resolution + 1), label="")  # comma is important to unpack the line object
legend = ax.legend()
fig.canvas.manager.set_window_title('Live Plot')  # Optional: name the window


# --- Plotting setup ---
plt.ion()
fig, ax = plt.subplots()
x_init = np.arange(resolution + 1)
line, = ax.plot(np.zeros_like(x_init), label="")
scatter = ax.scatter([], [], color='red', label='Zero crossings')
legend = ax.legend()
fig.canvas.manager.set_window_title('Live Plot')

zero_crossings_array = np.zeros((N_SNAPS, ))

for idx_snap, temperature in enumerate(temperatures_diff):
    # Update mesh point data
    comsol_data.mesh.point_data['temp'] = temperature[-1, :]

    # Sample along the line
    sampled = comsol_data.mesh.sample_over_line(start, end, resolution=resolution)

    # Extract x (line distance) and temperature
    temperature_diff_along_line = sampled.point_data['temp']
    zero_crossings = np.where(np.diff(np.signbit(temperature_diff_along_line)))[0]
    zero_crossings_array[idx_snap] = len(zero_crossings)
    
    # Update plot data
    line.set_ydata(temperature_diff_along_line)
    line.set_label(f'{DATA_TYPE} {idx_snap:03d}')

    # Update scatter
    scatter.set_offsets(np.c_[zero_crossings, temperature_diff_along_line[zero_crossings]])

    legend.remove()
    legend = ax.legend()

    ax.relim()
    ax.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.savefig(export_folder / f"Zero_Crossing_{DATA_TYPE}_{idx_snap:03d}.png")
    time.sleep(0.01)

np.save(ROOT / "data" / PARAMETER_SPACE / "Exports" / f"{DATA_TYPE}_zero_crossings.npy", zero_crossings_array)
# --- Final blocking show to keep window open after loop ---
plt.ioff()
plt.show()
    