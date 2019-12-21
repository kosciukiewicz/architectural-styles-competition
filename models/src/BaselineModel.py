import tensorflow as tf
import tensorflow.keras.layers as layers
from models.src.BaseCompetitionModel import BaseCompetitionModel


class BaselineModel(BaseCompetitionModel):
    def __init__(self):
        """Inits the class."""
        super(BaselineModel, self).__init__()
        self.base_layers = [
            tf.keras.layers.Conv2D(32, (5, 5),
                                   activation='relu',
                                   input_shape=(224, 224, 3)),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Conv2D(32, (5, 5),
                          activation='relu',
                          input_shape=(None, None, 12)
                          ),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(384, activation=tf.nn.relu),
            layers.Dense(192, activation=tf.nn.relu),
            layers.Dense(14, activation=tf.nn.softmax)
        ]

    def call(self, inputs, training=False):
        print(inputs.shape)
        x = inputs
        for layer in self.base_layers:
            x = layer(x)
        return x
