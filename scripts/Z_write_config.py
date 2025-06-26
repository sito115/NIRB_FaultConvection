import configparser
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from src.utils import CONFIG_PATH, read_config

def create_config():
    config = configparser.ConfigParser()

    spacing = 50
    # Add sections and key-value pairs
    config['General'] = {'log_level': logging.DEBUG,
                         'parameter_space': '09',
                         'field_name': 'Entropy',
                         'projection': 'Mapped',
                         'is_clean_mesh': False,
                         'root': Path(__file__).parents[1],
                         'control_mesh_suffix': f's{spacing}_{spacing}_{spacing}_b0_4000_0_5000_-4000_0'}
    config['Plotting'] = {'cmap': 'turbo',
                          'levels' : 15,
                          'label_font_size' : 10,
                          'title_font_size': 8} # 
    config['FluidProperties'] = {
        'lambda_f' : 0.65, # W/(K*m)
        'c_pf': 4200 # J/(kg*K)
    }

    # Write the configuration to a file
    with open(CONFIG_PATH, 'w') as configfile:
        config.write(configfile)
        print("Exported" + str(CONFIG_PATH))



if __name__ == '__main__':
    create_config()
    config_values = read_config()
    print(config_values)