#!/usr/bin/env python3
"""
Functional code to generate the plots for the results of the FDD framework
"""

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from IMU_processing.FFT.helper_func import convert_to_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from IMU_processing.FFT.FlightDataGenerator import FlightDataGenerator

mpl.use('TkAgg')

propeller_dictionary = {0: "front left", 1: "front right", 2: "back right", 3: "back left"}
propeller_names = {0: "FL", 1: "FR", 2: "BR", 3: "BL"}


def label2failure(labels, loss=None, acc=None):
    """
    Convert a tensor of labels or a label to the meaning of the class
    :param labels: int or list of labels
    :param loss: loss for the present flight
    :param acc: accuracy for the present flight
    :return: label of the flight, which includes the quantity of damage, the name of the damaged propeller and, if the
    loss and accuracy are provided by the user, they are included in the flight label
    """
    if type(labels) != int:
        failure_mode = tf.reduce_max(labels).numpy()
    else:
        failure_mode = labels

    if failure_mode == 0:
        flight_title = "No failure"
    else:
        propeller = (failure_mode - 1) // 4  # compute the number of the propeller
        propeller_name = propeller_dictionary[propeller]  # obtain the name of the propeller
        failure_magnitude = ((failure_mode - 1) % 4 * 0.2 + 0.2) * 100  # computes the magnitude of the failure
        flight_title = f"{np.round(failure_magnitude, 0)}% damage in {propeller_name} propeller"  # title of the failure
        if loss is not None:
            flight_title += f"\n Loss: {np.round(loss, 4)}. Acc:{np.round(acc, 3)}."  # add the accuracy and loss info
    return flight_title


def plot_predictions(model, generator_class, BATCH_SIZE, generator_input, dataset="train", flights_per_plot=6):
    """
    Tool for progressively plotting the labels and predictions from random flights from the provided dataset
    :param model: the ML model used
    :param generator_class: generator from which the data is retrieved
    :param BATCH_SIZE: the size of the batch
    :param generator_input: arguments for the function used to convert a generator to dataset
    :param dataset: dataset used for the generation of the Confusion Matrix, either the train or val dataset
    :param flights_per_plot: the number of flights shown in a single figure
    :return: None
    """
    # Obtain the train and validation generators
    _, _, _, _, [train_gen, val_gen] = convert_to_dataset(generator_class, BATCH_SIZE, False, **generator_input)
    stop = False
    if dataset == "train":
        generator = iter(train_gen())
    else:
        generator = iter(val_gen())

    # Create a plot with 3 columns
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
                            ignore_healthy=False, switch_single_frame=False):
    """
    Obtain (progressively) the confusion matrix (CM) from dataset batches
    :param model: the ML model used
    :param generator_class: generator from which the data is retrieved
    :param BATCH_SIZE: the size of the batch
    :param generator_input: arguments for the function used to convert a generator to dataset
    :param dataset: dataset used for the generation of the Confusion Matrix, either train or val
    :param ask_rate_it: number of iterations that have to pass before the user is asked to have the CM plotted
    :param ignore_healthy: whether the healthy label is ignored in the generation of the CM. This is done to emphasize
    the results associated with the failure diagnosis and not the failure detection
    :param switch_single_frame: whether a single frame is evaluated at a time
    :return: None
    """
    # Matplotlib settings
    fontsize = 19
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['grid.alpha'] = 0.5
    mpl.use('TkAgg')
    font = {'size': fontsize,
            'family': "Arial"}
    mpl.rc('font', **font)

    # Obtain all the datasets (train, val and test) and filter for the user desired one
    train_ds, val_ds, test_ds, _, _ = convert_to_dataset(generator_class, BATCH_SIZE, False, **generator_input)
    if dataset == "train":
        generator = iter(train_ds)
    elif dataset == "val":
        generator = iter(val_ds)
    elif dataset == "test":
        generator = iter(test_ds)
    else:
        raise ValueError("The provided dataset type is not recognised.")

    # Create confusion matrix x and y labels
    CM_labels = ["H"]  # stands for Healthy
    for i in range(16):
        name = f"{propeller_names[i//4]}{int(100*(i%4 * 0.2 + 0.2))}"  # name is a combination of failure prop and mag
        CM_labels.append(name)

    # Plot the Confusion Matrix (CM)
    stop = False
    labels_lst = []
    predictions_lst = []
    counter = 1
    while not stop:
        # Retrieve the data and provide correct shape
        data, labels = next(generator)  # provides the data of a complete flight
        if len(data.shape) != 3:
            data = tf.expand_dims(data, axis=0)
            labels = tf.expand_dims(labels, axis=0)

        # Re-evaluate the model
        if switch_single_frame:
            data = data[0]
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

                # Compute per class accuracy
                class_acc = C.diagonal()/C.sum(axis=1)
                for i in range(len(class_acc)):
                    flight_title = label2failure(i)
                    print(f"Class {i} ({flight_title}): {np.round(class_acc[i]*100, 2)}%")

                # General model accuracy
                general_accuracy = C.diagonal().sum()/C.sum()
                print(f"General accuracy: {np.round(general_accuracy * 100, 2)}%")

                # Compute detection accuracy
                detection_accuracy = 1-(C[0, 1:].sum() + C[1:, 0].sum())/C.sum()
                print(f"Detection accuracy: {np.round(detection_accuracy*100, 2)}%")

                # Compute classification accuracy
                classification_accuracy = C[1:, 1:].diagonal().sum() / C[1:, :].sum()
                print(f"Classification accuracy: {np.round(classification_accuracy * 100, 2)}%")

                # Bar plot with the class accuracy
                plt.close("all")
                plt.figure(1)
                plt.bar(CM_labels, class_acc)
                plt.xlabel("Classes", fontsize=24)
                plt.ylabel("Accuracy", fontsize=24)
                plt.grid(True)

                # If the healthy class is ignored for the confusion matrix, still provide the healthy class results
                if ignore_healthy:
                    print(f"True label = 0. Predictions: {C[0, :]}.")
                    print(f"True label = {C[:, 0]}. Predictions: 0.")
                    C = C[1:, 1:]
                    display_labels = CM_labels[1:]

                # Plot confusion matrix
                disp = ConfusionMatrixDisplay(C, display_labels=display_labels)
                disp.plot()
                fig = disp.figure_
                fig.subplots_adjust(left=0, top=0.99, right=0.99, bottom=0.11)
                ax = disp.ax_
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_xlabel('Predicted label', fontsize=24)
                ax.set_ylabel('True label', fontsize=24)
                plt.pause(1)
            elif decision == "s":
                stop = True
        counter += 1


def compute_maximum_dataset_sample_timesteps(figure_number, data_base_folder, flight_data_number, STFT_frequency,
                                             recording_start_time):
    """
    Computes the minimum number of time steps among all data samples
    :param figure_number: number of the next figure
    :param data_base_folder: location of the flight info and sensor data
    :param flight_data_number: number of the dataset
    :param STFT_frequency: frequency at which the STFT is done
    :param recording_start_time: time at which the stft starts to be done. A value higher than 0 is used in order to
    remove all initial transients
    :return: the number of the next figure, the minimum flight duration in the dataset and a list with the duration of
    all the flights
    """
    # Creates a Generator object without the camera (runs faster and it is not needed)
    stft_generator = FlightDataGenerator(data_base_folder, flight_data_number, STFT_frequency,
                                         recording_start_time=recording_start_time, switch_include_camera=False)

    # Obtain the number of labels for each flight which equals the number of time steps
    n_flights = len(stft_generator.flight_names)
    stft_generator = stft_generator()
    minimum_length = float("inf")
    length_lst = []
    for i in range(n_flights):
        frames, labels = next(stft_generator)
        length_lst.append(labels.shape[0])

        # Store the shortest flight duration
        if labels.shape[0] < minimum_length:
            minimum_length = labels.shape[0]
        if i % 100 == 0:
            print(f"{i} --> current minimum = {minimum_length}")

    # Plot a histogram and cumulative histogram with the flights' duration
    plt.figure(figure_number)
    figure_number += 1
    bins = sorted(set(length_lst))
    plt.hist(length_lst, bins)
    plt.title("Histogram")

    plt.figure(figure_number)
    figure_number += 1
    plt.hist(length_lst, bins, density=True, cumulative=True)
    plt.title("Cumulative histogram")
    return figure_number, minimum_length, length_lst
