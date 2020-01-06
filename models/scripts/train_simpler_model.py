import sys

sys.path.append("../..")
from models.src.SimplerModel import SimplerModel
from utils.data.loader import load_architectural_styles_training_set
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from settings import PATH_TO_SAVED_MODELS
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""GPU ON TF 2.0 cuDNN initialization fix"""
from tensorflow.compat.v1 import ConfigProto, GPUOptions
from tensorflow.compat.v1 import InteractiveSession
from utils.image.visualisation import show_image

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
"""END OF FIX"""


def normalize_img(img, labels):
    return tf.cast(img, dtype="float32") / 255.0, labels


def main():
    train_data, train_labels = load_architectural_styles_training_set()
    train_labels = train_labels.astype("int8")

    x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                        train_labels,
                                                        test_size=0.1,
                                                        random_state=123)

    data_generator = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=(0.4, 1.5),
        zoom_range=[0.9, 1.1],
        fill_mode='reflect'
    )

    data_generator.fit(x_train)
    data_iterator = data_generator.flow(x_train, y_train, batch_size=25)

    train_ds = tf.data.Dataset.from_generator(lambda: data_iterator,
                                              output_types=(
                                                  tf.float32, tf.int8))

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
        15)

    model = SimplerModel(hidden_dense_size=512, dropout=0.5, freezed_resnet_layers=4)
    print("start fitting")
    results = model.fit(train_dataset=train_ds, test_dataset=test_ds,
                        optimizer=tfa.optimizers.AdamW(learning_rate=0.00002,
                                                       weight_decay=0.00001),
                        max_epochs=100, early_stopping=20, steps_per_epoch=200, run_name="run_9")


if __name__ == '__main__':
    main()
