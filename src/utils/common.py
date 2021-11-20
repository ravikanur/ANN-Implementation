import yaml
import matplotlib.pyplot as plt
import time

import os


def read_config(config_path):
    with open(config_path) as config_file:  # command to open the file
        content = yaml.safe_load(config_file)  # to load the data of config file

    return content


def save_plot(history_data, plot_dir_path, plot_name):
    unique_name = time.strftime(f"{plot_name}_%Y%m%d_%H%M%S.png")
    history_data.plot(figsize=(10, 10))
    plt.grid

    plot_path = os.path.join(plot_dir_path, unique_name)
    plt.savefig(plot_path)
