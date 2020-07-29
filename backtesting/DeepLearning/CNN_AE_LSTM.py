@reset_graph
class CNN_AE_LSTM:
    """ Basic structure : CNN stacked Auto Encoder with four CNN layers(two for Encoder, two for Decoder),
                                    one LSTM layer, FFNN for last output """

    def __init__(
            self,
            learning_rate,
            input_dim,
            seq_len,
            output_len,
            final_ffnn_layer_dim,
            rnn_cell_hidden_dim,
            conv1_filters,
            conv2_filters,
            conv3_filters,  # decoding filters
            batch_size
    ):
        self.X = tf.placeholder(tf.float32, [None, seq_len, input_dim])
        self.Y = tf.placeholder(tf.float32, [None, output_len])
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_len = output_len
        self.final_ffnn_layer_dim = final_ffnn_layer_dim
        self.rnn_cell_hidden_dim = rnn_cell_hidden_dim
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.conv3_filters = conv3_filters
        self.batch_size = batch_size
        self.prediction
        self.optimize
        self.loss

    @lazy_property
    def Auto_Encoder(self):
        # CNN stacked AutoEncoder Part
        cnn_ae_layer = {'filter_1': tf.Variable(tf.random_normal([3, self.conv3_filters, self.conv2_filters])),
                        'filter_2': tf.Variable(tf.random_normal([3, self.conv2_filters, self.conv1_filters]))}

        # encoding
        conv1 = tf.layers.conv1d(self.X, filters=self.conv1_filters, kernel_size=3, strides=1, padding='SAME',
                                 activation='relu', use_bias=True, name='conv1')  # time step x conv1_filters
        maxpool1 = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2,
                                           name='pool1')  # (time step / 2) x conv1_filters
        conv2 = tf.layers.conv1d(maxpool1, filters=self.conv2_filters, kernel_size=3, strides=1, padding='SAME',
                                 activation='relu', use_bias=True, name='conv2')  # (time step / 2) x conv2_filters
        encoded = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2,
                                          name='encoding')  # (time step / 4) x conv2_filters

        # decoding
        conv3 = tf.layers.conv1d(encoded, filters=self.conv3_filters, kernel_size=3, strides=1, padding='SAME',
                                 activation='relu', use_bias=True, name='conv3')  # (time step / 4) x conv3_filters
        upsample1 = tf.nn.conv1d_transpose(conv3, filters=cnn_ae_layer['filter_1'],
                                           output_shape=(self.batch_size, int(self.seq_len / 2), self.conv2_filters),
                                           padding='SAME', strides=2,
                                           name='upsample1')  # (time step / 2) x conv2_filters
        upsample2 = tf.nn.conv1d_transpose(upsample1, filters=cnn_ae_layer['filter_2'],
                                           output_shape=(self.batch_size, self.seq_len, self.conv1_filters),
                                           padding='SAME', strides=2, name='upsample2')  # time step  x conv1_filters

        return upsample2

    @lazy_property
    def prediction(self):
        # LSTM part
        lstm_layer = {"weights1": tf.Variable(tf.random_normal(
            [self.rnn_cell_hidden_dim, self.final_ffnn_layer_dim]), name="layer1_weights"),
            "bias1": tf.Variable(tf.random_normal([self.final_ffnn_layer_dim]), name="layer1_bias"),
            "weights2": tf.Variable(tf.random_normal(
                [self.final_ffnn_layer_dim, self.output_len]), name="layer2_weights"),
            "bias2": tf.Variable(tf.random_normal([self.output_len]), name="layer2_bias")}

        lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.rnn_cell_hidden_dim)

        outputs, states = tf.nn.dynamic_rnn(lstm_cell, self.Auto_Encoder, dtype=tf.float32)
        layer1 = tf.nn.tanh(tf.matmul(outputs[:, -1, :], lstm_layer["weights1"]) + lstm_layer["bias1"])
        logits = tf.matmul(layer1, lstm_layer["weights2"]) + lstm_layer["bias2"]

        return logits

    @lazy_property
    def optimize(self):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.prediction, labels=self.Y
            )
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(cross_entropy)

    @lazy_property
    def loss(self):
        # 다른 metric 의 추이를 확인하고 싶을 시 변경 가능 (ex. MAE, WMAE ...)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.prediction, labels=self.Y
            )
        )
        return cross_entropy

def lazy_property(function):
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def reset_graph(func):
    def wrapper(*args, **kwargs):
        tf.reset_default_graph()
        result = func(*args, **kwargs)
        return result

    return wrapper


def launchGraph(
        model, train_X, train_Y, test_X, test_Y, num_epoch, batch_size, display_step=10
):
    train_loss_list = []
    test_loss_list = []

    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer())
        sess.run(init)

        for epoch in range(num_epoch):
            average_loss = 0
            average_test_loss = 0
            total_batch = int(len(train_X) / batch_size)

            for i in range(total_batch):
                X_batch, Y_batch = next_batch(train_X, train_Y, batch_size)
                X_batch = batch_normalization(X_batch)
                _, current_loss = sess.run([model.optimize, model.loss], feed_dict={model.X: X_batch, model.Y: Y_batch})
                average_loss += current_loss / total_batch

            train_loss_list.append(average_loss)

            test_batch = int(len(test_X) / batch_size)

            for j in range(test_batch):
                X_test_batch, Y_test_batch = next_batch(test_X, test_Y, batch_size)
                # X_test_batch = batch_normalization(X_test_batch)
                current_loss_test = sess.run([model.loss], feed_dict={model.X: X_test_batch, model.Y: Y_test_batch})[0]
                average_test_loss += current_loss_test / test_batch

            test_loss_list.append(average_test_loss)

            if epoch % display_step == 0:
                print("반복(Epoch): {}, 손실함수(Training Loss): {}, 손실함수(Testing Loss): {}".format(epoch + 1, average_loss,
                                                                                              average_test_loss))

        return train_loss_list, test_loss_list