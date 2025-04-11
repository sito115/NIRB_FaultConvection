import mph
from pathlib import Path
import logging
import time
import pyvista as pv
import numpy as np
import pandas as pd
import pint
import pint_pandas
from pint.delegates.formatter.plain import DefaultFormatter
from pint.delegates.formatter._format_helpers import formatter
from pint.delegates.formatter._compound_unit_helpers import prepare_compount_unit, localize_per

logging.basicConfig(
    filename= Path(__file__).parent / 'NIRB.log',    # Log file name
    level=logging.DEBUG,               # Set the lowest level of logging to capture
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the format of log messages
)


def format_comsol_unit(unit):
    return "[" + unit.replace("**", "^") + "]"
    


def training_loop():

    pint_pandas.PintType.ureg.formatter.default_format = "#D~"

    path = Path(r"nirb\NIRB-TH.mph")
    export_folder = Path(r"C:\Users\tsi-lokal\Documents\MPH-PY\MPh\nirb\Exports")
    assert export_folder.exists()
    client = mph.start(version='6.2')
    model = client.load(path)
    t_end = 4e13
    model.parameter('t_end', f'{t_end:.2e}[s]')
    # client.java.showProgress(str(Path(__file__).parent.joinpath('my_comsol.log').absolute()))

    time_steps = np.arange(0, t_end + 1, 1e12)
    study1 = (model / 'studies' / 'Study 1' / 'Time Dependent')
    study1.property('tlist', time_steps)

    training_param = pd.read_csv(r"nirb\test_samples.csv", header=[0, 1])
    training_param : pd.DataFrame = training_param.pint.quantify(level = -1)
    # units = [format_unit_simple(training_param[col].pint.units) for col in training_param.columns]
    for idx, row in training_param.iterrows():
        logging.info(f'Iteration: {idx=:03d}')
        export_file = export_folder.joinpath(f'Training_{idx:03d}.vtu')

        for col in training_param.columns:
            quantity = row[col]
            magnitude = quantity.magnitude
            unit = format_comsol_unit(str(quantity.units))
            formatted_quantity = f"{magnitude} {unit}" 
            logging.info(f"\t\t {col} : {formatted_quantity}")
            model.parameter(col, formatted_quantity)

        model.mesh()
        start_time = time.time()
        logging.info('\tSolving...')
        model.solve()
        end_time = time.time() - start_time
        model.export('Data 1', f'{export_file}')
        logging.info('\tExport sucessfull')
        mesh = pv.read(export_file)
        mesh.field_data['SimTime'] = end_time
        mesh.field_data['Parameters'] = model.parameters()
        mesh.save(export_file)
        del mesh
        logging.info('\tAdded Meta Data for Mesh')


if __name__ == "__main__":
    training_loop()