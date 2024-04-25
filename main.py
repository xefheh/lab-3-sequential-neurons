import pandas as pd

from scripts.graphics_show import show_mse_epochs_dependence, show_linear_regression
from scripts.neural_creation import create_sequential_linear_neural_network


if __name__ == '__main__':
    df = pd.read_csv('data/data.csv')
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()

    for neurons_count in [4]:
        model = create_sequential_linear_neural_network(units=neurons_count,
                                                        input_scale=1,
                                                        use_bias=True,
                                                        learning_rate=0.01)

        history = model.fit(x=x, y=y, epochs=50)
        hidden_weights = model.get_layer('hidden_layer').get_weights()
        output_weights = None if neurons_count == 1 else model.get_layer('output_layer').get_weights()
        show_mse_epochs_dependence(history.history['loss'])
        show_linear_regression(x=x, y=y, hidden_layer_weights=hidden_weights, output_layer_weights=output_weights)
