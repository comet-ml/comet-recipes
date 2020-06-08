import comet_ml
{%- if cookiecutter.confusion_matrix == "Yes" %}
from comet_ml import ConfusionMatrix
{%- endif %}

import logging

{%- if cookiecutter.framework == 'keras' %}
import keras
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
{%- endif %}

{%- if cookiecutter.embedding == "Yes" %}
import keract
import numpy as np
{%- endif %}


def get_comet_experiment():
    {%- if cookiecutter.online_or_offline == "Online" %}
    return comet_ml.Experiment(project_name="{{ cookiecutter.project_slug }}")
    {%- elif cookiecutter.online_or_offline == "Offline" %}
    return comet_ml.OfflineExperiment(
        offline_directory=".", project_name="{{ cookiecutter.project_slug }}"
    )
    {%- endif %}


{%- if cookiecutter.framework == 'keras' %}


def build_model_graph(experiment):
    model = Sequential()
    model.add(
        Dense(
            experiment.get_parameter("first_layer_units"),
            activation="sigmoid",
            input_shape=(784,),
        )
    )
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop(),
        metrics=["accuracy"],
    )

    return model


def train(experiment, model, x_train, y_train, x_test, y_test):
    {%- if cookiecutter.confusion_matrix == "Yes" %}
    class ConfusionMatrixCallback(Callback):
        def __init__(self, experiment, inputs, targets):
            self.experiment = experiment
            self.inputs = inputs
            self.targets = targets
            # We make one ConfusionMatrix so to share images:
            self.confusion_matrix = ConfusionMatrix(
                index_to_example_function=self.index_to_example)

        def on_epoch_begin(self, epoch, logs={}):
            # Get Confusion Matrix before training:
            if epoch == 0:
                self.on_epoch_end(-1)

        def on_epoch_end(self, epoch, logs={}):
            predicted = self.model.predict(self.inputs)
            # First we compute the matrix:
            self.confusion_matrix.compute_matrix(self.targets, predicted)
            # Then log it from the ConfusionMatrix object:
            self.experiment.log_confusion_matrix(
                matrix=self.confusion_matrix,
                title="Confusion Matrix, Epoch #%d" % (epoch + 1),
                file_name="confusion-matrix-%03d.json" % (epoch + 1),
            )

        def index_to_example(self, index):
            image_array = self.inputs[index]
            image_name = "confusion-matrix-%05d.png" % index
            result = self.experiment.log_image(
                image_array,
                name=image_name,
                image_shape=(28, 28, 1))
            # Return sample name and assetId
            return {"sample": image_name, "assetId": result["imageId"]}

    {%- endif %}

    {%- if cookiecutter.histogram == "Yes" %}
    class HistogramCallback(Callback):
        def __init__(self, experiment):
            self.experiment = experiment

        def on_epoch_begin(self, epoch, logs={}):
            # Get Histogram before training:
            if epoch == 0:
                self.on_epoch_end(-1)

        def on_epoch_end(self, epoch, logs={}):
            for layer in range(len(self.model.layers)):
                weights_biases = self.model.layers[layer].get_weights()
                self.experiment.log_histogram_3d(
                    weights_biases,
                    name="histogram-layer-%03d.json" % layer,
                    step=(epoch + 1),
                )

    {%- endif %}
    {%- if cookiecutter.embedding == "Yes" %}
    class EmbeddingCallback(Callback):
        def __init__(self, experiment, inputs, targets):
            self.experiment = experiment
            self.inputs = inputs
            self.targets = targets
            self.labels = targets.argmax(axis=1)
            self.image_size = (28, 28)
            image, self.sprite_url = self.experiment.create_embedding_image(
                image_data=self.inputs,
                # we round the pixels to 0 and 1, and multiple by 2
                # to keep the non-zero colors dark (if they were 1, they
                # would get distributed between 0 and 255):
                image_preprocess_function=lambda matrix: np.round(matrix,0) * 2,
                # Set the transparent color:
                image_transparent_color=(0, 0, 0),
                image_size=self.image_size,
                # Fill in the transparent color with a background color:
                image_background_color_function=self.label_to_color,
            )

        def label_to_color(self, index):
            label = self.labels[index]
            if label == 0:
                return (255, 0, 0)
            elif label == 1:
                return (0, 255, 0)
            elif label == 2:
                return (0, 0, 255)
            elif label == 3:
                return (255, 255, 0)
            elif label == 4:
                return (0, 255, 255)
            elif label == 5:
                return (128, 128, 0)
            elif label == 6:
                return (0, 128, 128)
            elif label == 7:
                return (128, 0, 128)
            elif label == 8:
                return (255, 0, 255)
            elif label == 9:
                return (255, 255, 255)

        def on_epoch_begin(self, epoch, logs={}):
            # Get Embedding before training:
            if epoch == 0:
                self.on_epoch_end(-1)

        def on_epoch_end(self, epoch, logs={}):
            # The vectors are big, only log them every 10 epochs:
            if (epoch + 1) % 10 != 0:
                return
            # Assuming one input bank:
            input_tensor = self.inputs
            activations = keract.get_activations(self.model, input_tensor)
            keys = list(activations.keys())
            # Get the activations before output layer:
            layer_name = keys[-2]
            # Group the embeddings by layer_name:
            self.experiment.log_embedding(
                activations[layer_name],
                self.labels,
                image_data=self.sprite_url,
                image_size=self.image_size,
                title="%s-%s" % (layer_name, epoch + 1),
                group=layer_name)

    {%- endif %}

    callbacks = [
        {%- if cookiecutter.predictor == "Yes" %}
        experiment.get_predictor_callback("keras"),
        {%- endif %}
        {%- if cookiecutter.confusion_matrix == "Yes" %}
        ConfusionMatrixCallback(experiment, x_test, y_test),
        {%- endif %}
        {%- if cookiecutter.histogram == "Yes" %}
        HistogramCallback(experiment),
        {%- endif %}
        {%- if cookiecutter.embedding == "Yes" %}
        EmbeddingCallback(experiment, x_test, y_test),
        {%- endif %}
    ]

    model.fit(
        x_train,
        y_train,
        batch_size=experiment.get_parameter("batch_size"),
        epochs=experiment.get_parameter("epochs"),
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )

    {%- if cookiecutter.embedding == "Yes" %}
    experiment.log_embedding_groups()
    {%- endif %}
    experiment.send_notification("Training done", "finished", {"Data": "100"})


def evaluate(experiment, model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    logging.info("Score %s", score)


def get_dataset():
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


{%- endif %}


def main():
    x_train, y_train, x_test, y_test = get_dataset()

    {%- if cookiecutter.optimizer == 'No' %}
    {%- if cookiecutter.framework == 'keras' %}
    experiment = get_comet_experiment()

    experiment.log_parameter("epochs", 10)
    experiment.log_parameter("batch_size", 120)
    experiment.log_parameter("first_layer_units", 128)
    {%- endif %}

    model = build_model_graph(experiment)

    train(experiment, model, x_train, y_train, x_test, y_test)

    evaluate(experiment, model, x_test, y_test)
    {%- endif %}

    {%- if cookiecutter.optimizer == "Yes" %}
    config = {
        "algorithm": "bayes",
        "name": "Optimize MNIST Network",
        "spec": {"maxCombo": 10, "objective": "minimize", "metric": "loss"},
        "parameters": {
            "first_layer_units": {
                "type": "integer",
                "mu": 500,
                "sigma": 50,
                "scalingType": "normal",
            },
            "epochs": {"type": "integer", "min": 1, "max": 100},
            "batch_size": {"type": "discrete", "values": [64, 128, 256]},
        },
        "trials": 1,
        {%- if cookiecutter.predictor == "Yes" %}
        "predictor": {
            "start": 1,
            "patience": 2,
            "threshold": 0.01,
            "loss_name": "val_loss",
        },
        {%- endif %}
    }

    {%- if cookiecutter.online_or_offline == "Online" %}
    experiment_class = comet_ml.Experiment
    {%- elif cookiecutter.online_or_offline == "Offline" %}
    experiment_class = comet_ml.OfflineExperiment
    {%- endif %}

    opt = comet_ml.Optimizer(config, experiment_class)

    experiment_kwargs = {
        "project_name": "{{ cookiecutter.project_slug }}",
        {%- if cookiecutter.online_or_offline == "Offline" %}
        "offline_dir": ".",
        {%- endif %}
    }

    for experiment in opt.get_experiments(**experiment_kwargs):
        model = build_model_graph(experiment)

        train(experiment, model, x_train, y_train, x_test, y_test)

        evaluate(experiment, model, x_test, y_test)
    {%- endif %}


if __name__ == "__main__":
    main()