from models.src.FirstToughtModel import FirstThoughtModel
from models.src.SimplerModel import SimplerModel

from utils.data.loader import load_architectural_styles_training_set, \
    load_architectural_styles_test_set
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from settings import PATH_TO_SAVED_MODELS
from sklearn.metrics import accuracy_score

"""GPU ON TF 2.0 cuDNN initialization fix"""
from tensorflow.compat.v1 import ConfigProto, GPUOptions
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
"""END OF FIX"""


def normalize_img(img, labels):
    return tf.cast(img, dtype="float32") / 255.0, labels


def main():
    train_data, train_labels = load_architectural_styles_training_set()
    test_data = load_architectural_styles_test_set()

    x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                        train_labels,
                                                        test_size=0.1,
                                                        random_state=123)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train[:2, :, :, :], y_train[:2])).shuffle(10000).batch(1)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_train[:2, :, :, :], y_train[:2])).batch(
        50)

    model = SimplerModel()
    results = model.fit(train_dataset=train_ds, test_dataset=test_ds,
                        optimizer=tf.optimizers.Adam(), run_name=None,
                        max_epochs=1, early_stopping=-1)
    model.load_weights(f"{PATH_TO_SAVED_MODELS}/simpler_model/run_5/best.h5",
                       by_name=True)

    pred = np.argmax(model.predict(test_data), axis=1)
    #print(accuracy_score(y_test, pred))
    submission_df = pd.DataFrame(
        data=np.array([np.arange(pred.shape[0]), pred]).transpose(),
        columns=['Id', 'Category'])
    submission_df.to_csv("../../submissions/new.csv", index=False)


if __name__ == '__main__':
    main()
