import tensorflow as tf

from datetime import datetime
from models.src.SimplerModel import SimplerModel
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from settings import PATH_TO_OUTPUT
from utils.experiment_runner.experiment_runner import ExperimentRunner, Param, Runner, MetricRunner
from sklearn.model_selection import train_test_split
from utils.data.loader import load_architectural_styles_training_set
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATE_FORMAT = "%Y_%m_%d_%H_%M_%S"

from tensorflow.compat.v1 import ConfigProto, GPUOptions
from tensorflow.compat.v1 import InteractiveSession
import gc
import os

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from numba import cuda

optimizers_dict = {
    "SGD": tf.keras.optimizers.SGD,
    "Adam": tf.keras.optimizers.Adam,
    "RMSProp": tf.keras.optimizers.RMSprop,
}


def save_experiment_results(run_name, run_results):
    results = np.array([np.arange(run_results['params']['epochs']) + 1,
                        run_results['train_accuracies'],
                        run_results['test_accuracies'],
                        run_results['test_losses'],
                        run_results['train_losses']]).transpose()

    results_df = pd.DataFrame(data=results,
                              columns=['epochs', 'train_accuracy',
                                       'test_accuracy', 'test_loss',
                                       'train_loss'])

    results_df.to_csv(
        f"{PATH_TO_OUTPUT}/{run_name}/"
        f"lr_{run_results['params']['learning_rate']}"
        f"_wd_{run_results['params']['weight_decay_rate']}"
        f"_batch_size_{run_results['params']['batch_size']}.csv")


def normalize_img(img, labels):
    return tf.cast(img, dtype="float32") / 255.0, labels


def run_experiment(params):
    model_name = f"{params['model_name']}_{datetime.now().strftime(DATE_FORMAT)}"
    x_train = params['x_train']
    x_test = params['x_test']
    y_train = params['y_train']
    y_test = params['y_test']

    train_ds = params['train_ds']
    test_ds = params['test_ds']

    model = SimplerModel(hidden_dense_size=512, dropout=0.5, freezed_resnet_layers=4)
    results = model.fit(train_dataset=train_ds, test_dataset=test_ds,
                        optimizer=tfa.optimizers.AdamW(learning_rate=params["learning_rate"],
                                                       weight_decay=params["weight_decay_rate"]),
                        max_epochs=params["epochs"], early_stopping=-1, steps_per_epoch=400, run_name="run_4")

    train_losses, train_accuracies, test_losses, test_accuracies, fit_time, mean_epoch_time = results

    results = {
        "params": params,
        "y_train_pred": np.argmax(model.predict(x_train).squeeze(), axis=1),
        "y_test_pred": np.argmax(model.predict(x_test).squeeze(), axis=1),
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "fit_time": fit_time,
        "mean_epoch_time": mean_epoch_time,
    }

    return results


def run_first_experiment():
    train_data, train_labels = load_architectural_styles_training_set()

    x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                        train_labels,
                                                        test_size=0.1,
                                                        random_state=123)

    data_generator = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.5,
        height_shift_range=0.5,
        brightness_range=(0.7, 1.1),
        zoom_range=[0.8, 1.2],
    )

    data_generator.fit(x_train)
    data_iterator = data_generator.flow(x_train, y_train, batch_size=25)

    train_ds = tf.data.Dataset.from_generator(lambda: data_iterator,
                                              output_types=(
                                                  tf.float32, tf.int8))

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
        15)

    runner = ExperimentRunner(
        name="deep_learning_experiment",
        params=[
            Param(
                name="model_name",
                default_value="simple_model",
            ),
            Param(
                name="epochs",
                default_value=1,
            ),
            Param(
                name="train_ds",
                default_value=train_ds,
                ignore_in_results=True
            ),
            Param(
                name="test_ds",
                default_value=test_ds,
                ignore_in_results=True
            ),
            Param(
                name="x_train",
                default_value=x_train,
                ignore_in_results=True
            ),
            Param(
                name="x_test",
                default_value=x_test,
                ignore_in_results=True
            ),
            Param(
                name="y_test",
                default_value=y_test,
                ignore_in_results=True
            ),
            Param(
                name="y_train",
                default_value=y_train,
                ignore_in_results=True
            ),
            Param(
                name="learning_rate",
                default_value=0.01
            ),
            Param(
                name="batch_size",
                default_value=100
            ),
            Param(
                name="weight_decay_rate",
                default_value=0.0001
            )
        ],

        metrics=[
            MetricRunner(
                name="train_accuracy",
                run=(lambda results: accuracy_score(y_train,
                                                    results['y_train_pred']))
            ),
            MetricRunner(
                name="train_f1",
                run=(
                    lambda results: f1_score(y_train, results['y_train_pred'],
                                             average='macro'))
            ),
            MetricRunner(
                name="test_accuracy",
                run=(lambda results: accuracy_score(y_test,
                                                    results['y_test_pred']))
            ),
            MetricRunner(
                name="test_f1",
                run=(
                    lambda results: f1_score(y_test, results['y_test_pred'],
                                             average='macro'))
            ),
            MetricRunner(
                name="train_loss",
                run=(
                    lambda results: results["train_losses"][-1])
            ),
            MetricRunner(
                name="test_loss",
                run=(
                    lambda results: results["test_losses"][-1])
            ),
            MetricRunner(
                name="fit_time",
                run=(
                    lambda results: results["fit_time"])
            ),
            MetricRunner(
                name="mean_epoch_time",
                run=(
                    lambda results: results["mean_epoch_time"])
            )
        ],
        output_path="./output"
    )

    runner.run_experiment(Runner(
        name="simple_model_experiment_inc_res",
        run_function=run_experiment,
        experiment_params={
            "learning_rate": [0.0001, 0.00007, 0.00005, 0.00002],
            "weight_decay_rate": [0.00007, 0.00004, 0.00001],
        },
        default_params={
            "optimizer": "AdamW",
            "batch_size": 20,
            "epochs": 12,
        },
        results_callbacks=[save_experiment_results]
    ))

    runner.run()


def main():
    run_first_experiment()


if __name__ == '__main__':
    main()
