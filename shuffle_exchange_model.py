# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

import shuffle_exchange_network as network
import config as cnf


class ModelSpecific:
    """
    Task specific model structure
    """

    def cost(self, prediction) -> tuple:
        """
        :rtype: tuple (cost, per_item_cost)
        """
        pass

    def accuracy(self, prediction):
        """
        :return: Accuracy as float tensor (single)
        """
        pass

    def result(self, prediction):
        pass


class LambadaModel(ModelSpecific):

    def __init__(self, target, n_classes, label_smoothing) -> None:
        self.__target = target
        self.__n_classes = n_classes
        self.__y_one_hot = tf.one_hot(self.__target, self.__n_classes, dtype=tf.float32)
        self.__label_smoothing = label_smoothing

    def cost(self, prediction):
        labels = self.__y_one_hot[:, :, 2] / tf.reduce_sum(self.__y_one_hot[:, :, 2], axis=1, keepdims=True)
        smooth_positives = 1.0 - self.__label_smoothing
        smooth_negatives = self.__label_smoothing / labels.get_shape().as_list()[1]
        onehot_labels = labels * smooth_positives + smooth_negatives
        cost_vector = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction[:, :, 2], labels=onehot_labels)
        return tf.reduce_mean(cost_vector), cost_vector

    def accuracy(self, prediction):
        positions = tf.argmax(prediction[:, :, 2], axis=1)
        selected = self.__y_one_hot[:, :, 2]

        batch_index = tf.expand_dims(tf.range(positions.shape[0], dtype=tf.int64), axis=1)
        positions = tf.expand_dims(positions, axis=1)
        indices = tf.concat((batch_index, positions), axis=1)
        accuracy1 = tf.gather_nd(selected, indices)
        return tf.reduce_mean(accuracy1)

    def result(self, prediction):
        return tf.argmax(prediction[:, :, 2], axis=1)


class DefaultModel(ModelSpecific):
    def __init__(self, target, n_classes, label_smoothing) -> None:
        self.__target = target
        self.__n_classes = n_classes
        self.__label_smoothing = label_smoothing

    def cost(self, prediction):
        y = tf.one_hot(self.__target, self.__n_classes, dtype=tf.float32)
        smooth_positives = 1.0 - self.__label_smoothing
        smooth_negatives = self.__label_smoothing / self.__n_classes
        onehot_labels = y * smooth_positives + smooth_negatives

        # costVector = tf.reduce_sum(tf.square(prediction - y), axis=2)
        # costVector = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels = y_in)  # Softmax loss

        # prediction_mm = self.conv_linear(cur, 1, self.num_units, self.n_classes, 0.0, "output_mm")
        # prediction = tf.minimum(prediction, prediction_mm)+0.1*(prediction+prediction_mm)
        costVector = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=onehot_labels)
        cost1 = tf.reduce_mean(costVector, 1)

        return tf.reduce_mean(cost1), cost1

    @staticmethod
    def get_accuracy(prediction, y_in):
        result = tf.argmax(prediction, 2)
        correct_symbols = tf.equal(result, y_in)
        mask_y_in = tf.cast(tf.not_equal(y_in, 0), tf.float32)
        mask_out = tf.cast(tf.not_equal(result, 0), tf.float32)
        mask_2 = tf.maximum(mask_y_in, mask_out)
        correct_symbols = tf.cast(correct_symbols, tf.float32)
        correct_symbols *= mask_2
        return tf.reduce_sum(correct_symbols, 1) / tf.reduce_sum(mask_2, 1)

    def accuracy(self, prediction):
        accuracy1 = self.get_accuracy(prediction, self.__target)
        # accuracy2 = self.get_accuracy(prediction1, y_in)
        # accuracy = tf.reduce_mean(tf.maximum(accuracy1, accuracy2))
        return tf.reduce_mean(accuracy1)

    def result(self, prediction):
        return tf.argmax(prediction, axis=2)


class ShuffleExchangeModel:
    def __init__(self, num_units, bins, n_input, count_list, n_classes, dropout_keep_prob,
                 create_translation_model=False, use_two_gpus=False):
        self.translation_model = create_translation_model
        self.use_two_gpus = use_two_gpus
        self.n_classes = n_classes
        self.n_input = n_input
        self.num_units = num_units
        self.bins = bins
        self.count_list = count_list
        self.accuracy = None
        self.cost = None
        self.optimizer = None
        self.cost_list = None
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.Variable(cnf.initial_learning_rate, trainable=False)
        self.beta2_rate = tf.maximum(0.0005,
                                     tf.train.exponential_decay(0.01, self.global_step, 2000, 0.5, staircase=False))
        self.bin_losses = []
        network.dropout_keep_prob = dropout_keep_prob
        self.allMem = None
        self.x_input = []
        self.y_input = []
        self.test_x = None
        self.test_y = None
        self.lr_decay_op = self.learning_rate.assign(tf.maximum(cnf.min_learning_rate, self.learning_rate * 0.7))
        self.n_middle = 48
        network.is_training = tf.placeholder(tf.bool, name="is_training")

        if cnf.use_pre_trained_embedding:
            with open(cnf.emb_vector_file, "rb") as emb_file:
                emb = pickle.load(emb_file)  # Load binary numpy array with embeddings

            with tf.device('/cpu:0'):
                self.embedding_initializer = tf.constant_initializer(emb, verify_shape=True)
                self.embedding_shape = emb.shape

    def create_loss(self, x_in_indices, y_in, length):
        """perform loss calculation for one bin """

        batch_size = self.count_list[0]
        if cnf.use_pre_trained_embedding:
            cur = self.pre_trained_embedding(x_in_indices, length)
        else:
            cur = self.embedding(x_in_indices, length)

        if cnf.input_word_dropout_keep_prob < 1:
            cur = tf.cond(network.is_training,
                          lambda: tf.nn.dropout(cur, cnf.input_word_dropout_keep_prob,
                                                noise_shape=[batch_size, length, 1]), lambda: cur)
        if cnf.input_dropout_keep_prob < 1:
            cur = tf.cond(network.is_training, lambda: tf.nn.dropout(cur, cnf.input_dropout_keep_prob),
                          lambda: cur)

        cur, allMem = network.shuffle_exchange_network(cur, "steps", n_blocks=cnf.n_Benes_blocks)

        print(length, len(allMem))
        allMem_tensor = tf.stack(allMem)
        prediction = network.conv_linear(cur, 1, self.num_units, self.n_classes, 0.0, "output")

        if cnf.task == "lambada":
            model = LambadaModel(y_in, self.n_classes, cnf.label_smoothing)
        else:
            model = DefaultModel(y_in, self.n_classes, cnf.label_smoothing)

        cost, per_item_cost = model.cost(prediction)
        result = model.result(prediction)
        accuracy = model.accuracy(prediction)

        return cost, accuracy, allMem_tensor, prediction, per_item_cost, result

    def embedding(self, indices, length):
        with tf.device('/cpu:0'):
            emb_weights = tf.get_variable("embedding", [self.n_input, self.num_units],
                                          initializer=tf.random_uniform_initializer(-0.01, 0.01))

            cur = tf.gather(emb_weights, indices)
        return network.activation_fn(30 * cur)

    def pre_trained_embedding(self, indices, length):
        with tf.device('/cpu:0'):
            emb_weights = tf.get_variable("embedding", self.embedding_shape, tf.float32, initializer=self.embedding_initializer, trainable=False)
            cur = tf.nn.embedding_lookup(emb_weights, indices)

        cur = network.conv_linear(cur, 1, self.embedding_shape[1], self.num_units, 0.0, "embedding_linear", False)
        return network.activation_fn(cur)

    def create_test_graph(self, test_length):
        """Creates graph for accuracy evaluation"""
        with vs.variable_scope("var_lengths"):
            itemCount = self.count_list[0]
            self.test_x = tf.placeholder("int64", [itemCount, test_length])
            self.test_y = tf.placeholder("int64", [itemCount, test_length])
            _, self.test_accuracy, self.allMem, _, _, self.result = self.create_loss(self.test_x, self.test_y,
                                                                                     test_length)
            self.test_summary = tf.summary.scalar("base/test_accuracy", self.test_accuracy)

    def create_graph(self):
        """Creates graph for training"""
        self.cost = 0.0
        self.accuracy = 0
        num_sizes = len(self.bins)
        self.cost_list = []
        self.bin_losses = []

        # Create all bins and calculate losses for them

        with vs.variable_scope("var_lengths"):
            for seqLength, itemCount, ind in zip(self.bins, self.count_list, range(num_sizes)):
                x_in = tf.placeholder("int64", [itemCount, seqLength])
                y_in = tf.placeholder("int64", [itemCount, seqLength])
                self.x_input.append(x_in)
                self.y_input.append(y_in)
                network.saturation_costs = []
                network.gate_mem = []
                network.reset_mem = []
                network.candidate_mem = []
                network.prev_mem_list = []

                if self.use_two_gpus:
                    device = "/device:GPU:" + ("0" if seqLength >= self.bins[-1] else "1")
                    with tf.device(device):
                        c, a, mem1, _, perItemCost, _ = self.create_loss(x_in, y_in, seqLength)
                else:
                    c, a, mem1, _, perItemCost, _ = self.create_loss(x_in, y_in, seqLength)

                # /seqLength
                self.bin_losses.append(perItemCost)
                self.cost += c
                self.accuracy += a
                self.cost_list.append(c)
                tf.get_variable_scope().reuse_variables()

        # calculate the total loss
        self.cost /= num_sizes
        self.accuracy /= num_sizes

        # tensorboard output
        tf.summary.scalar("base/loss", self.cost)
        tf.summary.scalar("base/accuracy", self.accuracy)
        tf.summary.scalar("base/accuracy_longest", a)

        gate_img = tf.stack(network.gate_mem)
        gate_img = gate_img[:, 0:1, :, :]
        gate_img = tf.cast(gate_img * 255, dtype=tf.uint8)
        tf.summary.image("gate", tf.transpose(gate_img, [3, 0, 2, 1]), max_outputs=16)
        reset_img = tf.stack(network.reset_mem)
        reset_img = reset_img[:, 0:1, :, :]
        reset_img = tf.cast(reset_img * 255, dtype=tf.uint8)
        tf.summary.image("reset", tf.transpose(reset_img, [3, 0, 2, 1]), max_outputs=16)
        prev_img = tf.stack(network.prev_mem_list)
        prev_img = prev_img[:, 0:1, :, :]
        prev_img = tf.cast(prev_img * 255, dtype=tf.uint8)
        tf.summary.image("prev_mem", tf.transpose(prev_img, [3, 0, 2, 1]), max_outputs=16)

        candidate_img = tf.stack(network.candidate_mem)
        candidate_img = candidate_img[:, 0:1, :, :]
        candidate_img = tf.cast((candidate_img + 1.0) * 127.5, dtype=tf.uint8)
        tf.summary.image("candidate", tf.transpose(candidate_img, [3, 0, 2, 1]), max_outputs=16)

        mem1 = mem1[:, 0:1, :, :]
        tf.summary.image("mem", tf.transpose(mem1, [3, 0, 2, 1]), max_outputs=16)

        tvars = tf.trainable_variables()
        for var in tvars:
            name = var.name.replace("var_lengths", "")
            tf.summary.histogram(name + '/histogram', var)

        # gradients and optimizer
        grads = tf.gradients(self.cost, tvars, colocate_gradients_with_ops=True)

        # we use a small L2 regularization, although it is questionable if it helps
        regularizable_vars = [var for var in tvars if "CvK" in var.name]
        reg_costlist = [tf.reduce_sum(tf.square(var)) for var in regularizable_vars]
        reg_cost = tf.add_n(reg_costlist)
        tf.summary.scalar("base/regularize_loss", reg_cost)

        LazyAdamW = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.contrib.opt.LazyAdamOptimizer)
        optimizer = LazyAdamW(weight_decay = tf.cast(0.001*self.learning_rate, tf.float32), learning_rate = self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-5)
        self.optimizer = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, decay_var_list=regularizable_vars)

        # some values for printout
        max_vals = []

        for var in tvars:
            varV = optimizer.get_slot(var, "v")
            max_vals.append(varV)

        self.gnorm = tf.global_norm(max_vals)
        self.cost_list = tf.stack(self.cost_list)

    def prepare_dict(self, batch_xs_list, batch_ys_list, is_training):
        """Prepares a dictionary of input output values for all bins to do training"""
        feed_dict = {network.is_training: is_training}
        for x_in, data_x in zip(self.x_input, batch_xs_list):
            feed_dict[x_in.name] = data_x
        for y_in, data_y in zip(self.y_input, batch_ys_list):
            feed_dict[y_in.name] = data_y

        return feed_dict

    def prepare_test_dict(self, batch_xs_list, batch_ys_list):
        """Prepares a dictionary of input output values for all bins to do testing"""
        feed_dict = {network.is_training: 0}
        feed_dict[self.test_x.name] = batch_xs_list[0]
        feed_dict[self.test_y.name] = batch_ys_list[0]
        return feed_dict

    def get_all_memory(self, sess, batch_xs_list, batch_ys_list):
        """Gets an execution trace for the given inputs"""
        feed_dict = self.prepare_test_dict(batch_xs_list, batch_ys_list)
        mem = sess.run((self.allMem), feed_dict=feed_dict)
        return mem

    def get_accuracy(self, sess, batch_xs_list, batch_ys_list):
        """Gets accuracy on the given test examples"""
        feed_dict = self.prepare_test_dict(batch_xs_list, batch_ys_list)
        acc, result, summary = sess.run((self.test_accuracy, self.result, self.test_summary), feed_dict=feed_dict)
        return acc, result, summary

    def get_learning_rate(self, sess):
        rate = sess.run((self.learning_rate))
        return rate

    def print_loss(self, sess, batch_xs_list, batch_ys_list):
        """prints training loss on the given inputs"""
        feed_dict = self.prepare_dict(batch_xs_list, batch_ys_list, 0)
        acc, loss, costs, norm11, beta2, summaries = sess.run((self.accuracy, self.cost, self.cost_list,
                                                               self.gnorm, self.beta2_rate,
                                                               self.variable_summaries),
                                                              feed_dict=feed_dict)
        print("Loss= " + "{:.6f}".format(loss) + \
              ", Accuracy= " + "{:.6f}".format(acc), costs, "gnorm=", norm11)
        return summaries

    def train(self, sess, batch_xs_list, batch_ys_list, do_dropout=1):
        """do training"""
        feed_dict = self.prepare_dict(batch_xs_list, batch_ys_list, do_dropout)

        res = sess.run([self.cost, self.optimizer, self.accuracy, self.cost_list] + self.bin_losses,
                       feed_dict=feed_dict)
        loss = res[0]
        acc = res[2]
        costs = res[3]
        lossPerItem = res[5:]
        return loss, acc, lossPerItem, costs
