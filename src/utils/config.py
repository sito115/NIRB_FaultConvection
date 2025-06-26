import configparser
from pathlib import Path
import matplotlib.pyplot as plt

CONFIG_PATH = Path(__file__).parents[2] / 'config.ini'


def read_config() -> dict:
    
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(CONFIG_PATH)

    # Access values from the configuration file
    log_level = config.getint('General', 'log_level')
    config.remove_option('General', 'log_level')
    is_clean_mesh = config.getboolean('General', 'is_clean_mesh')
    config.remove_option('General', 'is_clean_mesh')

    cmap = config.get('Plotting', 'cmap')
    levels = config.getint('Plotting', 'levels')
    
    config_values = {key: value for key, value in config.items('General')}
    config_values['log_level'] = log_level
    config_values['is_clean_mesh'] = is_clean_mesh

    if cmap not in plt.colormaps():
        raise ValueError(f"Colormap '{cmap}' is unknown.")
    my_cmap = plt.get_cmap(cmap, levels)
    
    data_folder = Path(config_values['root']) / "data" / config_values['parameter_space']
    if not data_folder.exists():
        data_folder = None
    config_values['data_folder'] = data_folder
    config_values['cmap'] = my_cmap

    section_name = "FluidProperties"
    if config.has_section(section_name):
        for key in config[section_name].keys():
            config_values[key] = config.getfloat(section_name, key)
    
    
    return config_values