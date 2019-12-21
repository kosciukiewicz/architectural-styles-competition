import tensorflow as tf
import tensorflow.keras.layers as layers
from models.src.BaseCompetitionModel import BaseCompetitionModel

INPUT_SIZE = (224, 224, 3)


class SimplerModel(BaseCompetitionModel):
    def __init__(self):
        """Inits the class."""
        super(SimplerModel, self).__init__()
        self.resnet_layers = None
        self.resnet_dense = None
        self.dense_layers = None
        self.conv_layers = None

        self._init_resnet_module()
        # self._init_conv_module()
        # self._init_dense_module()

    def _init_resnet_module(self):
        self.resnet_layers = tf.keras.applications.resnet_v2.ResNet50V2(
            weights='imagenet', include_top=False, pooling='avg',
            input_tensor=layers.Input(INPUT_SIZE))

        for layer in self.resnet_layers.layers[:4]:
            layer.trainable = False

        self.resnet_dense = [
            layers.Flatten(),
            layers.Dense(512, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
            layers.Dropout(0.5),
            layers.Dense(64, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
            layers.Dropout(0.5),
            layers.Dense(14, activation=tf.nn.softmax)
        ]

    def get_model_name(self):
        return "simpler_model"

    @tf.function
    def call(self, inputs, training=False):
        resnet_output = self.resnet_layers(inputs)

        for resnet_dense_layer in self.resnet_dense:
            resnet_output = resnet_dense_layer(resnet_output)
            print(resnet_output.shape)

        return resnet_output
        # conv_output = inputs

        # for conv_layer in self.conv_layers:
        #    conv_output = conv_layer(conv_output)
        #    print(conv_output.shape)

        # conv_output = tf.reshape(conv_output, [conv_output.shape[0],
        #                                       -1])  # flatten with remaining batch dim

        # dense_input = tf.concat([conv_output, resnet_output], axis=1)
        # print(dense_input.shape)
        # for dense_layer in self.dense_layers:
        #    dense_input = dense_layer(dense_input)

        # return dense_input

    def _init_conv_module(self):
        self.conv_layers = [
            layers.SeparableConv2D(12, (3, 3),
                                   kernel_initializer=tf.initializers.he_normal(),
                                   padding='valid', strides=(1, 1),
                                   depth_multiplier=2),
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Conv2D(32, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.Conv2D(32, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Conv2D(48, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.Conv2D(48, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Conv2D(64, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.Conv2D(64, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
        ]

    def _init_dense_module(self):
        self.dense_layers = [
            layers.Dense(128, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
            layers.Dropout(0.5),
            layers.Dense(512, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
            layers.Dropout(0.5),
            layers.Dense(14, activation=tf.nn.softmax,
                         kernel_initializer="random_normal"
                         )
        ]
