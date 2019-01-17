import tflearn
import tensorflow as tf


def test_recurrent_layers():
    X = [[1, 3, 5, 7], [2, 4, 8, 10], [1, 5, 9, 11], [2, 6, 8, 0]]
    Y = [[0., 1.], [1., 0.], [0., 1.], [1., 0.]]

    with tf.Graph().as_default():
        g = tflearn.input_data(shape=[None, 4])
        g = tflearn.embedding(g, input_dim=12, output_dim=4)
        g = tflearn.lstm(g, 6)
        g = tflearn.fully_connected(g, 2, activation='softmax')
        g = tflearn.regression(g, optimizer='sgd', learning_rate=1.)

        m = tflearn.DNN(g)
        m.fit(X, Y, n_epoch=300, snapshot_epoch=False)
        print(m.predict([[5, 9, 11, 1]])[0][1], 0.9)


test_recurrent_layers()
