from models.src.BlackAndWhiteModel import BlackAndWhiteModel
from utils.data.loader import load_architectural_styles_training_set
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils.image.visualisation import show_image
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


def preprocess_image(img):
    img = img.numpy()
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR)
    return grayImage


def generate_data(img, labels):
    gray_images = tf.convert_to_tensor(
        tf.py_function(preprocess_image, [img], tf.uint8))
    return img, gray_images, labels


def main():
    train_data, train_labels = load_architectural_styles_training_set()

    x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                        train_labels,
                                                        test_size=0.1,
                                                        random_state=123)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(300).map(generate_data).batch(20).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(
        generate_data).batch(
        20).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = BlackAndWhiteModel()
    results = model.fit(train_dataset=train_ds, test_dataset=test_ds,
                        optimizer=tf.optimizers.Adam(0.00015),
                        max_epochs=50, early_stopping=20, run_name="run_1")


if __name__ == '__main__':
    main()
