import tensorflow as tf
import os
import time


def create_model(Loss_function, Optimizer, Metrics):
    LAYERS = [
          tf.keras.layers.Flatten(input_shape = [28,28]),
          tf.keras.layers.Dense(300, activation="relu", name="HiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="HiddenLayer2"),
          tf.keras.layers.Dense(10, activation="softmax", name="OutputLayer")
    ]

    model_clf = tf.keras.models.Sequential(LAYERS)
     
    model_clf.summary()

    model_clf.compile(loss=Loss_function, optimizer=Optimizer, metrics=Metrics)

    return model_clf  # returns untrained model


def get_unique_name(filename):
    unique_name = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_name


def save_model(model, model_name, model_dir_path):
    unique_name = get_unique_name(model_name)
    path = os.path.join(model_dir_path, unique_name)
    model.save(path)






