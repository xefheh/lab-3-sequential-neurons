from tensorflow import keras


def create_sequential_linear_neural_network(units: int, input_scale: int, use_bias: bool = True,
                                            learning_rate: float = 0.1, momentum: float = 0.0,
                                            kernel_initializer: keras.initializers.Constant =
                                            keras.initializers.Constant(1),
                                            bias_initializer: keras.initializers.Constant =
                                            keras.initializers.Constant(1)) -> keras.Sequential:

    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(input_scale,),
                                      name='input_layer'))

    model.add(keras.layers.Dense(units=units,
                                 use_bias=use_bias,
                                 name='hidden_layer',
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer))

    if units > 1:
        model.add(keras.layers.Dense(units=1,
                                     name='output_layer'))

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate,
                                                 momentum=momentum),
                  loss='mse',
                  metrics=['mse'])

    return model
