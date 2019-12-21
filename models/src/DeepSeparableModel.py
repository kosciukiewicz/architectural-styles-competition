import tensorflow as tf
import tensorflow.keras.layers as layers
from models.src.BaseCompetitionModel import BaseCompetitionModel

INPUT_SIZE = (224, 224, 3)


class DeepSeparableModel(BaseCompetitionModel):
    def __init__(self):
        """Inits the class."""
        super(DeepSeparableModel, self).__init__()
        self.dense_layers = None
        self.conv_layers = None

        self._init_conv_module()
        self._init_dense_module()

    def get_model_name(self):
        return "deep_separable_model"

    @tf.function
    def call(self, inputs, training=False):
        conv_output = inputs

        for conv_layer in self.conv_layers:
            conv_output = conv_layer(conv_output)
            print(conv_output.shape)

        dense_input = conv_output

        for dense_layer in self.dense_layers:
            dense_input = dense_layer(dense_input)

        return dense_input

    def _init_conv_module(self):
        self.conv_layers = [
            layers.SeparableConv2D(4, (3, 3),
                                   kernel_initializer=tf.initializers.he_normal(),
                                   padding='valid', strides=(1, 1),
                                   depth_multiplier=1),
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Conv2D(24, (5, 5),
                          activation='relu',
                          ),
            layers.Conv2D(24, (5, 5),
                          activation='relu',
                          ),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Conv2D(32, (5, 5),
                          activation='relu',
                          ),
            layers.Conv2D(32, (5, 5),
                          activation='relu',
                          ),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Conv2D(64, (5, 5),
                          activation='relu',
                          ),
            layers.Conv2D(64, (5, 5),
                          activation='relu',
                          ),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
        ]

    def _init_dense_module(self):
        self.dense_layers = [
            layers.Dense(256, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
            layers.Dropout(0.5),
            layers.Dense(14, activation=tf.nn.softmax,
                         kernel_initializer="random_normal"
                         )
        ]
