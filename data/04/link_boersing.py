import sys
from pathlib import Path
import logging
import numpy as np
sys.path.append(f'{Path.home().joinpath(r"Documents/COMSOL_MPH_API/")}')
from MPh import mph


def loop_2D_model(save_folder: Path, model: mph.Model):
    

    thermal_gradients = np.arange(0.02, 0.07, 0.005) # K/m

    for idx, t_grad in enumerate(np.sort(thermal_gradients)):
        logging.info(f'Entering sim for : {t_grad=}')


        model.parameter('T_grad', f'{t_grad:.3e}[K/m]')
        model.solve('Study 1')
        
        for export in model.exports():
            save_path = save_folder.joinpath(f'{export}_tgrad_{t_grad:.3f}.vtu')
            
            # if "data" in export.lower():
            #     save_path = save_path.with_suffix('.vtu')
            
            model.export(export, save_path)


def load_mph_model(model_path: Path, log_path : Path = None) -> mph.Model:
    logging.info(model_path)
    assert model_path.exists()
    
    client = mph.start()
    model = client.load(model_path)
    
    if log_path is None:
        log_path = Path(__file__).parent.joinpath(f'{model_path.stem}.log')
    logging.info(f'Log Path: {log_path}')
    client.java.showProgress(str(log_path))
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    
    model_path = Path("/Users/thomassimader/Documents/COMSOL_MPH_API/Börsing_2017/Model_Boersing_2D_DummyLayer.mph")
    save_folder = Path("/Users/thomassimader/Documents/COMSOL_MPH_API/Börsing_2017/Exports")
    
    assert save_folder.exists()

    model = load_mph_model(model_path)
    loop_2D_model(save_folder, model)