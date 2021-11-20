from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model
from utils.model import save_model
from utils.common import save_plot

import argparse
import pandas as pd

import os


def training(config_path):
    config = read_config(config_path)

    validation_size = config["params"]["validation_datasize"]

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_size)

    EPOCHS = config["params"]["epochs"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    LOSS_FUNCTION = config["params"]["loss_function"]
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_name = config["artifacts"]["model_name"]
    plots_dir = config["artifacts"]["plots_dir"]
    plot_name = config["artifacts"]["plot_name"]
    log_dir = config["logs"]["logs_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    plot_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plot_dir_path, exist_ok=True)

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS)

    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),epochs=EPOCHS, verbose=True)
    history_data = pd.DataFrame(history.history)

    save_model(model, model_name, model_dir_path)
    save_plot(history_data, plot_dir_path, plot_name)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml", help="mention the config file name")

    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
