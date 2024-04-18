import numpy as np
import matplotlib.pyplot as plt


def show_mse_epochs_dependence(mses: np.array):
    plt.plot(mses)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()


def show_linear_regression(x: np.array, y: np.array, w0: float, bias: float):
    plt.scatter(x, y)
    x_for_graphic = np.linspace(x.min(), x.max() + 1)
    plt.plot(x_for_graphic, linear_function(x_for_graphic, w0, bias))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def linear_function(x: np.array, w0: float, bias: float) -> np.array:
    return w0 * x + bias
