import configparser
from pathlib import Path
import matplotlib.pyplot as plt

CONFIG_PATH = Path(__file__).parents[1] / 'config.ini'


def read_config() -> dict:
    
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(CONFIG_PATH)

    # Access values from the configuration file
    log_level = config.get('General', 'log_level')
    cmap = config.get('Plotting', 'cmap')
    levels = config.getint('Plotting', 'levels')
    ps = config.get('General', 'parameter_space')

    if cmap not in plt.colormaps():
        raise ValueError(f"Colormap '{cmap}' is unknown.")
    my_cmap = plt.get_cmap(cmap, levels)
    data_folder = Path(__file__).parents[1] / "data" / ps
    if not data_folder.exists():
        data_folder = None
    

    # Return a dictionary with the retrieved values
    config_values = {
        'log_level': log_level,
        'data_folder': data_folder,
        'cmap': my_cmap,
    }

    return config_values