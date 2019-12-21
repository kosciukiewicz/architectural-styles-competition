import tensorflow as tf
import tensorflow.keras.layers as layers
from models.src.BaseCompetitionModel import BaseCompetitionModel
from time import time
from tqdm import tqdm
from utils.image.visualisation import show_image

INPUT_SIZE = (224, 224, 3)


class BlackAndWhiteModel(BaseCompetitionModel):
    def __init__(self):
        """Inits the class."""
        super(BlackAndWhiteModel, self).__init__()
        self.resnet_layers = None
        self.resnet_dense = None
        self.dense_layers = None
        self.white_conv_layers = None
        self.color_conv_layers = None

        self._init_resnet_module()
        self._init_white_conv_module()
        self._init_color_conv_module()
        self._init_dense_module()

    def _init_resnet_module(self):
        self.resnet_layers = tf.keras.applications.resnet_v2.ResNet50V2(
            weights='imagenet', include_top=False, pooling='avg',
            input_tensor=layers.Input(INPUT_SIZE))

        for layer in self.resnet_layers.layers:
            layer.trainable = False

        self.resnet_dense = [
            layers.Flatten(),
            layers.Dense(128, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
        ]

    def get_model_name(self):
        return "black_and_white"

    @tf.function
    def call(self, inputs, gray_images, training=False):
        resnet_output = self.resnet_layers(inputs)
        print("RESNET")
        for resnet_dense_layer in self.resnet_dense:
            resnet_output = resnet_dense_layer(resnet_output)
            print(resnet_output.shape)


        print("GRAY")

        conv_white_output = gray_images

        for conv_white_layer in self.white_conv_layers:
            conv_white_output = conv_white_layer(conv_white_output)
            print(conv_white_output.shape)

        conv_white_output = tf.reshape(conv_white_output, [conv_white_output.shape[0],
                                               -1])  # flatten with remaining batch dim

        print("COLOR")

        conv_color_output = inputs

        for conv_color_layer in self.color_conv_layers:
            conv_color_output = conv_color_layer(conv_color_output)
            print(conv_color_output.shape)

        conv_color_output = tf.reshape(conv_color_output, [conv_color_output.shape[0],
                                                           -1])  # flatten with remaining batch dim


        dense_input = tf.concat([conv_white_output, conv_color_output,  resnet_output], axis=1)
        for dense_layer in self.dense_layers:
            dense_input = dense_layer(dense_input)

        return dense_input

    @tf.function
    def test_step(self, loss_object, images, gray_images, labels):
        predictions = self.call(images, gray_images)
        t_loss = loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    @tf.function
    def train_step(self, optimizer, loss_object, images, gray_images, labels):
        with tf.GradientTape() as tape:
            predictions = self.call(images, gray_images)
            labels = tf.expand_dims(labels, 1)
            l2_loss = tf.add_n(
                [tf.nn.l2_loss(v) for v in self.trainable_variables])
            loss = loss_object(labels, predictions) + 0.015 * l2_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def _init_color_conv_module(self):
        self.color_conv_layers = [
            layers.SeparableConv2D(4, (3, 3),
                                   kernel_initializer=tf.initializers.he_normal(),
                                   padding='valid', strides=(1, 1),
                                   depth_multiplier=2),
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Conv2D(12, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.Conv2D(12, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Conv2D(24, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.Conv2D(24, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
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
            layers.Flatten(),
            layers.Dense(128, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
        ]

    def _init_white_conv_module(self):
        self.white_conv_layers = [
            layers.Conv2D(12, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.Conv2D(12, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.MaxPool2D((3, 3),
                             strides=(2, 2)),
            layers.Conv2D(24, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
            layers.Conv2D(24, (5, 5),
                          activation='relu',
                          kernel_initializer=tf.initializers.he_normal(),
                          ),
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
            layers.Dense(64, activation=tf.nn.relu,
                         kernel_initializer="glorot_normal"
                         ),
            layers.Dropout(0.5),
            layers.Dense(14, activation=tf.nn.softmax,
                         kernel_initializer="random_normal"
                         )
        ]

    def fit(self, **kwargs):
        train_dataset = kwargs['train_dataset']
        test_datasest = kwargs['test_dataset']
        max_epochs = kwargs['max_epochs']
        optimizer = kwargs['optimizer']
        early_stopping = kwargs['early_stopping']
        run_name = kwargs['run_name']
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        test_losses = []
        train_losses = []
        test_accuracies = []
        train_accuracies = []
        epoch_times = []
        fit_time_start = time()
        best_val_accuracy = 0.0
        best_val_loss = 100
        learn = True
        epoch = 0

        while learn and epoch < max_epochs:
            epoch += 1
            epoch_time_start = time()

            data = [(images, gray_images, labels) for
                    images, gray_images, labels in
                    train_dataset]

            for i in tqdm(range(len(data))):
                images, gray_images, labels = data[i]
                images = tf.cast(images, dtype="float32") / 255.0
                gray_images = tf.cast(gray_images, dtype="float32") / 255.0
                self.train_step(optimizer, loss_object, images, gray_images,
                                labels)

            for images, gray_images, labels in test_datasest:
                images = tf.cast(images, dtype="float32") / 255.0
                gray_images = tf.cast(gray_images, dtype="float32") / 255.0
                self.test_step(loss_object, images, gray_images, labels)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'

            epoch_time = time() - epoch_time_start

            epoch_times.append(epoch_time)
            train_losses.append(self.train_loss.result().numpy())
            train_accuracies.append(self.train_accuracy.result().numpy() * 100)
            test_losses.append(self.test_loss.result().numpy())
            test_accuracies.append(self.test_accuracy.result().numpy() * 100)

            print(template.format(epoch,
                                  train_losses[-1],
                                  train_accuracies[-1],
                                  test_losses[-1],
                                  test_accuracies[-1]))
            # Reset the metrics for the next epoch
            val_accuracy = self.test_accuracy.result().numpy()
            val_loss = self.test_loss.result().numpy()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_best_weights(run_name)
                early_stopping = kwargs['early_stopping']
            else:
                if early_stopping == 0:
                    learn = False
                    print("Eearly stopped")
                else:
                    early_stopping -= 1

        self.save_last_weights(run_name)
        fit_time = time() - fit_time_start
        mean_epoch_time = sum(epoch_times) / len(epoch_times)
        return train_losses, train_accuracies, test_losses, test_accuracies, fit_time, mean_epoch_time
        pass
