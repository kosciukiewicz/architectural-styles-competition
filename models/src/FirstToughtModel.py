import tensorflow as tf
import tensorflow.keras.layers as layers
from models.src.BaseCompetitionModel import BaseCompetitionModel

INPUT_SIZE = (224, 224, 3)


class FirstThoughtModel(BaseCompetitionModel):
    def __init__(self):
        """Inits the class."""
        super(FirstThoughtModel, self).__init__()
        self.VGG16_module = None
        self.InceptionResNetV2_module = None

        self.vgg_dense = None
        self.resnet_dense = None

        self._init_vgg16_module()
        self._init_resnet_module()
        self._init_dense_module()

    def _init_vgg16_module(self):
        self.VGG16_module = tf.keras.applications.vgg16.VGG16(
            weights='imagenet', include_top=False,
            input_tensor=layers.Input(INPUT_SIZE))

        self.vgg_dense = layers.Dense(300, activation=tf.nn.relu,
                                      kernel_initializer="glorot_normal")

        for layer in self.VGG16_module.layers:
            layer.trainable = False

    def _init_resnet_module(self):
        self.InceptionResNetV2_module = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            weights='imagenet', include_top=False,
            input_tensor=layers.Input(INPUT_SIZE))

        self.resnet_dense = layers.Dense(300, activation=tf.nn.relu,
                                         kernel_initializer="glorot_normal")

        for layer in self.InceptionResNetV2_module.layers:
            layer.trainable = False

    @tf.function
    def call(self, inputs, training=False):
        vgg_output = self.VGG16_module(inputs)
        vgg_output = tf.reshape(vgg_output, [vgg_output.shape[0],
                                             -1])  # flatten with remaining batch dim
        vgg_output = self.vgg_dense(vgg_output)

        resnet_output = self.InceptionResNetV2_module(inputs)
        resnet_output = tf.reshape(resnet_output, [resnet_output.shape[0],
                                                   -1])  # flatten with remaining batch dim
        resnet_output = self.resnet_dense(resnet_output)

        dense_input = tf.concat([vgg_output, resnet_output], axis=1)

        for dense_layer in self.dense_layers:
            dense_input = dense_layer(dense_input)

        return dense_input

    def _init_dense_module(self):
        self.dense_layers = [
            layers.Dense(256, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
            layers.Dropout(0.6),
            layers.Dense(64, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
            layers.Dropout(0.6),
            layers.Dense(14, activation=tf.nn.softmax,
                         kernel_initializer="random_normal"
                         )
        ]
