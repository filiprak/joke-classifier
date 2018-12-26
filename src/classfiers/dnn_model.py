import tflearn

class DNNModel():
    def __init__(self, input_length, output_length, shape=[64, 64], dropout=0.8, activation='relu'):
        self.input_layer = tflearn.input_data(shape=[None, input_lentth])
        self.hidden_layers = []
        previous_layer = self.input_layer
        for n in shape:
            hidden = tflearn.fully_connected(previous_layer, 
                                             n, 
                                             activation=activation)
            self.hidden_layers.append(hidden)
            if dropout is not None:
                hidden = tflearn.dropout(hidden, 0.8)
                self.hidden_layers.append(hidden)
            previous_layer = hidden

        self.output_layer = tflearn.fully_connected(previous_layer, output_length, activation='softmax')
        sgd = tflearn.SGD(learning_rate=0.1, decay_step=1000)
        self.model = tflearn.regression(self.output_layer, 
                                        optimizer=sgd,
                                        loss='categorical_crossentropy')

    
