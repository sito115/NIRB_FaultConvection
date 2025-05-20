import configparser
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from src.utils import CONFIG_PATH, read_config

def create_config():
    config = configparser.ConfigParser()

    # Add sections and key-value pairs
    config['General'] = {'log_level': logging.DEBUG,
                         'parameter_space': '01'}
    config['Plotting'] = {'cmap': 'turbo',
                          'levels' : 15,
                          'label_font_size' : 10,
                          'title_font_size': 8} # 

    # Write the configuration to a file
    with open(CONFIG_PATH, 'w') as configfile:
        config.write(configfile)
        print("Exported" + str(CONFIG_PATH))



if __name__ == '__main__':
    create_config()
    config_values = read_config()
    print(config_values)