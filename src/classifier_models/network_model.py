import tflearn


def create_model(input_length, output_length, activation='relu'):
    input_layer = tflearn.input_data(shape=[None, input_length])
    model = tflearn.fully_connected(input_layer,
                                    64,
                                    activation=activation)
    model = tflearn.dropout(model, 0.8)
    model = tflearn.fully_connected(input_layer,
                                    64,
                                    activation=activation)
    model = tflearn.dropout(model, 0.8)
    softmax = tflearn.fully_connected(model, output_length, activation='softmax')
    sgd = tflearn.SGD(learning_rate=0.1, decay_step=1000)
    net = tflearn.regression(softmax,
                             optimizer=sgd,
                             loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model
