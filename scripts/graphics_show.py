import numpy as np
import matplotlib.pyplot as plt


def show_mse_epochs_dependence(mses: np.array):
    plt.plot(mses)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()


def show_linear_regression(x: np.array, y: np.array, hidden_layer_weights: np.array, output_layer_weights: np.array):
    plt.scatter(x, y)
    x_linspace = np.linspace(x.min(), x.max() + 1)
    plt.plot(x_linspace, get_neuron_sum(x_linspace, hidden_layer_weights, output_layer_weights), color='green')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def get_neuron_sum(x: np.array, hidden_layer_weights: np.array, output_layer_weights: np.array = None):
    neurons_count = hidden_layer_weights[0].shape[1]
    if neurons_count == 1:
        return hidden_layer_weights[0][0][0] * x + hidden_layer_weights[1][0]
    values = []
    for i in range(neurons_count):
        a_hidden = hidden_layer_weights[0][0][i]
        b_hidden = hidden_layer_weights[1][i]

        a_output = output_layer_weights[0][i][0]
        b_output = output_layer_weights[1][0]

        values.append(a_output * (a_hidden * x + b_hidden) + b_output / neurons_count)
    return sum(values)
