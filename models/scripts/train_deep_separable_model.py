from models.src.DeepSeparableModel import DeepSeparableModel
from utils.data.loader import load_architectural_styles_training_set
from sklearn.model_selection import train_test_split
import tensorflow as tf
from settings import PATH_TO_SAVED_MODELS
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2

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

    x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                        train_labels,
                                                        test_size=0.2,
                                                        random_state=123)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(25)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
        25)

    model = DeepSeparableModel()
    results = model.fit(train_dataset=train_ds, test_dataset=test_ds,
                        optimizer=tf.optimizers.Adam(0.0001),
                        max_epochs=50, early_stopping=10, run_name="run_2")


if __name__ == '__main__':
    main()
