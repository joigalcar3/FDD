import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from IMU_processing.FFT.ImuModelLSTM import ImuModelLstmFunctional, BatchLogging
from IMU_processing.FFT.StftGenerator import StftGenerator
from IMU_processing.FFT.helper_func import convert_to_dataset

propeller_dictionary = {0: "front left", 1: "front right", 2: "back right", 3: "back left"}


def label2failure(labels, loss=None, acc=None):
    """
    Convert a tensor of labels or a label to the meaning of the class
    :param labels: int or list of labels
    :param loss: loss for the present flight
    :param acc: accuracy for the present flight
    :return:
    """
    if type(labels) != int:
        failure_mode = tf.reduce_max(labels).numpy()
    else:
        failure_mode = labels

    if failure_mode == 0:
        flight_title = "No failure"
    else:
        propeller = (failure_mode - 1) // 4
        propeller_name = propeller_dictionary[propeller]
        failure_magnitude = ((failure_mode - 1) % 4 * 0.2 + 0.2) * 100
        flight_title = f"{np.round(failure_magnitude, 0)}% damage in {propeller_name} propeller"
        if loss is not None:
            flight_title += f"\n Loss: {np.round(loss, 4)}. Acc:{np.round(acc, 3)}."
    return flight_title


def plot_predictions(model, generator_class, BATCH_SIZE, generator_input, dataset="train", flights_per_plot=6):
    """
    Tool for progressively plotting the labels and predictions from random flights from the provided dataset
    :param model: the ML model used
    :param generator_class: generator from which the data is retrieved
    :param BATCH_SIZE: the size of the batch
    :param generator_input: arguments for the function used to convert a generator to dataset
    :param dataset: dataset used for the generation of the CM, either train or val
    :param flights_per_plot: the number of flights shown in a single figure
    :return:
    """
    _, _, _, [train_gen, val_gen] = convert_to_dataset(generator_class, BATCH_SIZE, False, **generator_input)
    stop = False
    if dataset == "train":
        generator = iter(train_gen())
    else:
        generator = iter(val_gen())
    figure_number = 1
    rows = int(np.ceil(flights_per_plot/3))
    counter = 0
    ax = None
    while not stop:
        # Create the plots
        if counter % flights_per_plot == 0:
            fig, ax = plt.subplots(rows, 3, num=figure_number)
            figure_number += 1
            ax = ax.flatten()
            fig.set_size_inches(19.24, 10.55)
            plt.pause(0.01)
        ax_local = ax[counter % flights_per_plot]

        # Retrieve the data and provide correct shape
        data, labels = next(generator)
        if len(data.shape) != 3:
            data = tf.expand_dims(data, axis=0)
            labels = tf.expand_dims(labels, axis=0)

        # Re-evaluate the model
        raw_prediction = model.predict(data)
        prediction = tf.squeeze(tf.argmax(raw_prediction, axis=-1))
        metrics_out = model.evaluate(data, labels, verbose=2)
        loss, acc = metrics_out[0], metrics_out[1]
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
        action = input(f"Loss: {loss}")

        # Obtain failure type
        flight_title = label2failure(labels, loss, acc)

        # Plot the results
        ax_local.plot(prediction)
        ax_local.plot(tf.squeeze(labels), linestyle="--")
        ax_local.set_xlabel("STFT timestep")
        ax_local.set_ylabel("Prediction/labels")
        ax_local.set_ylim([0, 17])
        ax_local.title.set_text(flight_title)
        ax_local.grid(True)
        fig.tight_layout()
        plt.pause(0.3)

        if action:
            stop = True
        counter += 1


def create_confusion_matrix(model, generator_class, BATCH_SIZE, generator_input, dataset="train", ask_rate_it=1,
                            ignore_healthy=False):
    """
    Obtain (progressively) the confusion matrix (CM) from dataset batches
    :param model: the ML model used
    :param generator_class: generator from which the data is retrieved
    :param BATCH_SIZE: the size of the batch
    :param generator_input: arguments for the function used to convert a generator to dataset
    :param dataset: dataset used for the generation of the CM, either train or val
    :param ask_rate_it: number of iterations that have to pass before the user is asked to have the CM plotted
    :param ignore_healthy: whether the healthy label is ignored in the generation of the CM
    :return:
    """
    train_ds, val_ds, _, _ = convert_to_dataset(generator_class, BATCH_SIZE, False, **generator_input)
    if dataset == "train":
        generator = iter(train_ds)
    else:
        generator = iter(val_ds)
    stop = False
    labels_lst = []
    predictions_lst = []
    counter = 1
    while not stop:
        # Retrieve the data and provide correct shape
        data, labels = next(generator)
        if len(data.shape) != 3:
            data = tf.expand_dims(data, axis=0)
            labels = tf.expand_dims(labels, axis=0)

        # Re-evaluate the model
        raw_prediction = model.predict(data)
        prediction = tf.squeeze(tf.argmax(raw_prediction, axis=-1))

        # Retrieve confusion matrix inputs
        labels = list(np.array(tf.reshape(labels, [-1])))
        predictions = list(np.array(tf.reshape(prediction, [-1])))
        labels_lst.extend(labels)
        predictions_lst.extend(predictions)

        # Retrieve action from user
        if counter % ask_rate_it == 0:
            decision = input("Plot confusion matrix [y/n]? Press 's' for stop!")
            if decision == "y":
                # Create confusion matrix
                display_labels = list(range(17))
                C = confusion_matrix(labels_lst, predictions_lst, labels=display_labels)
                class_acc = C.diagonal()/C.sum(axis=1)
                for i in range(len(class_acc)):
                    flight_title = label2failure(i)
                    print(f"Class {i} ({flight_title}): {np.round(class_acc[i]*100, 2)}%")

                plt.close("all")
                plt.figure(1)
                plt.bar(display_labels, class_acc)
                plt.xlabel("Classes")
                plt.ylabel("Accuracy [0,1]")
                plt.grid(True)

                if ignore_healthy:
                    print(f"True label = 0. Predictions: {C[0, :]}.")
                    print(f"True label = {C[:, 0]}. Predictions: 0.")
                    C = C[1:, 1:]
                    display_labels = list(range(1, 17))
                disp = ConfusionMatrixDisplay(C, display_labels=display_labels)
                disp.plot()
                plt.pause(1)
            elif decision == "s":
                stop = True
        counter += 1


if __name__ == "__main__":
    #%% Input data
    base_folder = "D:\\AirSim_project_512_288"
    flight_number = 43
    sampling_frequency = 10
    start_time = 1.0
    desired_timesteps = 55
    switch_failure_modes = True
    switch_flatten = True
    BATCH_SIZE = 10
    generator_input = {"data_base_folder": base_folder, "flight_data_number": flight_number,
                       "STFT_frequency": sampling_frequency, "recording_start_time": start_time,
                       "n_time_steps_des": desired_timesteps, "switch_failure_modes": switch_failure_modes,
                       "switch_flatten": switch_flatten, "shuffle_flights": False}

    #%% Input model
    l = 3                                         # Detection: 2
    f = 30                                        # Detection: 10
    n_classes = 17                                # Detection: 2
    checkpoint_name = "batched_multiclass_sm"     # Detection: saved_model
    epochs = 50                                   # Detection: 10   Classification: 30
    patience = 10                                 # Detection: 2   Classification: 3

    #%% Define model
    data_sample_shape = StftGenerator(generator_type="train", **generator_input).example[0].shape
    model = ImuModelLstmFunctional(l, f, n_classes).model(data_sample_shape)
    model = BatchLogging(model, None, None)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    #%% Loads the weights
    checkpoint_path = f"checkpoints/{checkpoint_name}"
    model.load_weights(checkpoint_path)

    # create_confusion_matrix(model, StftGenerator, BATCH_SIZE, generator_input, ask_rate_it=1, ignore_healthy=True)
    plot_predictions(model, StftGenerator, BATCH_SIZE, generator_input)


