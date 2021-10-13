from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model
from utils.model import save_model

import argparse

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

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS)

    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),epochs=EPOCHS, verbose=True)

    save_model(model, model_name, model_dir_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml", help="mention the config file name")

    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
