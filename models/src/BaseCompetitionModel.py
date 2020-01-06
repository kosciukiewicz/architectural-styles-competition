import tensorflow as tf
import abc
import numpy as np
from time import time
from tqdm import tqdm
from settings import PATH_TO_SAVED_MODELS
import os


class BaseCompetitionModel(tf.keras.Model):
    def __init__(self):
        """Inits the class."""
        super(BaseCompetitionModel, self).__init__()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

    @abc.abstractmethod
    def call(self, inputs, training=False):
        """Makes forward pass of the network."""
        pass

    @abc.abstractmethod
    def get_model_name(self):
        """Makes forward pass of the network."""
        pass

    @tf.function
    def test_step(self, loss_object, images, labels):
        predictions = self.call(images)
        t_loss = loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    @tf.function
    def train_step(self, optimizer, loss_object, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.call(images)
            labels = tf.expand_dims(labels, 1)
            # l2_loss = tf.add_n(
            #     [tf.nn.l2_loss(v) for v in self.trainable_variables])
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def fit(self, **kwargs):
        train_dataset = kwargs['train_dataset']
        test_datasest = kwargs['test_dataset']
        max_epochs = kwargs['max_epochs']
        optimizer = kwargs['optimizer']
        early_stopping = kwargs['early_stopping']
        run_name = kwargs['run_name']
        steps_per_epoch = kwargs.get("steps_per_epoch", None)

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

            if steps_per_epoch is None:
                data = [(images, labels) for images, labels in
                        train_dataset]
                for i in tqdm(range(len(data))):
                    images, labels = data[i]
                    images = tf.cast(images, dtype="float32") / 255.0
                    self.train_step(optimizer, loss_object, images, labels)
            else:
                it = iter(train_dataset)
                for _ in tqdm(range(steps_per_epoch)):
                    images, labels = next(it)
                    images = tf.cast(images, dtype="float32") / 255.0
                    self.train_step(optimizer, loss_object, images, labels)

            for images, labels in test_datasest:
                images = tf.cast(images, dtype="float32") / 255.0
                self.test_step(loss_object, images, labels)

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

    def crate_directory_if_not_exist(self, run_name):
        if not os.path.exists(
                f"{PATH_TO_SAVED_MODELS}/{self.get_model_name()}/{run_name}"):
            os.makedirs(
                f"{PATH_TO_SAVED_MODELS}/{self.get_model_name()}/{run_name}")

    def save_best_weights(self, run_name):
        if run_name:
            self.crate_directory_if_not_exist(run_name)
            self.save_weights(
                f"{PATH_TO_SAVED_MODELS}/{self.get_model_name()}/{run_name}/best.h5")

    def save_last_weights(self, run_name):
        if run_name:
            self.crate_directory_if_not_exist(run_name)
            self.save_weights(
                f"{PATH_TO_SAVED_MODELS}/{self.get_model_name()}/{run_name}/last.h5")

    def predict(self, inputs):
        results = []

        for i in tqdm(self.batch(range(0, inputs.shape[0]), 10)):
            images = inputs[i, :, :, :] / 255.0
            pred = self.call(images).numpy()
            results.extend(pred)

        return np.array(results)

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
