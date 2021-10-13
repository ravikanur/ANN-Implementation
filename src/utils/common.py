import yaml


def read_config(config_path):
    with  open(config_path) as config_file:  # command to open the file
        content = yaml.safe_load(config_file)  # to load the data of config file

    return content
