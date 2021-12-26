import tensorflow as tf


class Cell(object):
    def __init__(self, length, max_sentence_len, emb_vec, emb_vec_size, rnn_size, upper_bound):
        self.length = length
        self.max_sentence_len = max_sentence_len
        self.emb_vec = emb_vec
        self.emb_vec_size = emb_vec_size
        self.rnn_size = rnn_size
        self.upper_bound = upper_bound

        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.Train_Q = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.Train_A = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.Train_F = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.Test_Q = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_sentence_len])
        self.Test_A = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_sentence_len])

        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            tf_embedding = tf.compat.v1.Variable(tf.compat.v1.to_float(self.emb_vec), trainable=True, name="W")
            questions = tf.nn.embedding_lookup(tf_embedding, self.Train_Q)
            true_answers = tf.nn.embedding_lookup(tf_embedding, self.Train_A)
            false_answers = tf.nn.embedding_lookup(tf_embedding, self.Train_F)

            test_questions = tf.nn.embedding_lookup(tf_embedding, self.Test_Q)
            test_answers = tf.nn.embedding_lookup(tf_embedding, self.Test_A)

        with tf.compat.v1.variable_scope("LSTM_scope", reuse=None):
            question1 = self.get_gradient(questions, self.rnn_size)
            question2 = tf.nn.tanh(self.down_sampling(question1))
        with tf.compat.v1.variable_scope("LSTM_scope", reuse=True):
            true_answer1 = self.get_gradient(true_answers, self.rnn_size)
            true_answer2 = tf.nn.tanh(self.down_sampling(true_answer1))
            false_answer1 = self.get_gradient(false_answers, self.rnn_size)
            false_answer2 = tf.nn.tanh(self.down_sampling(false_answer1))

            test_question1 = self.get_gradient(test_questions, self.rnn_size)
            test_question2 = tf.nn.tanh(self.down_sampling(test_question1))
            test_answer1 = self.get_gradient(test_answers, self.rnn_size)
            test_answer2 = tf.nn.tanh(self.down_sampling(test_answer1))

        self.trueCosSim = self.dot_product(question2, true_answer2)
        self.falseCosSim = self.dot_product(question2, false_answer2)
        self.loss = self.bp(self.trueCosSim, self.falseCosSim, self.upper_bound)

        self.result = self.dot_product(test_question2, test_answer2)

    def get_gradient(self, x, hidden_size):
        #计算梯度(lstm网络单元的参数计算)
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)
        cell1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        cell1 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell1, input_keep_prob=self.dropout_keep_prob,output_keep_prob=self.dropout_keep_prob)
        cell2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        cell2 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell2, input_keep_prob=self.dropout_keep_prob,output_keep_prob=self.dropout_keep_prob)
        output, _, _ = tf.compat.v1.nn.static_bidirectional_rnn(cell1, cell2, input_x, dtype=tf.float32)
        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])
        return output

    @staticmethod
    #计算两个向量的点积
    def dot_product(q, a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        dotproduct = tf.compat.v1.div(mul, tf.multiply(q1, a1))
        return dotproduct

    @staticmethod
    #下采样，缩小数据规模，此处使用pooling方法
    def down_sampling(org_matrix):
        height = int(org_matrix.get_shape()[1])
        width = int(org_matrix.get_shape()[2])
        org_matrix = tf.expand_dims(org_matrix, -1)
        reduced_matrix = tf.nn.max_pool(org_matrix, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        reduced_matrix = tf.reshape(reduced_matrix, [-1, width])
        return reduced_matrix

    @staticmethod
    #反向传播，计算模型的loss值
    def bp(org, pred, upperbound):
        zero = tf.fill(tf.shape(org), 0.0)
        tfMargin = tf.fill(tf.shape(org), upperbound)
        with tf.compat.v1.name_scope("loss"):
            # max-margin losses = max(0, margin - true + false)
            losses = tf.compat.v1.maximum(zero, tf.subtract(tfMargin, tf.subtract(org, pred)))
            loss = tf.compat.v1.reduce_sum(losses)
        return loss
