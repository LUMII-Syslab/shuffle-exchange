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

import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import config as cnf
import data_feeder
import data_utils as data_gen
from shuffle_exchange_model import ShuffleExchangeModel

print("Running Shuffle-Exchange trainer.....")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not cnf.use_two_gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = cnf.gpu_instance

countList = [cnf.batch_size for x in cnf.bins]
np.set_printoptions(linewidth=2000, precision=4, suppress=True)

# prepare training and test data
max_length = cnf.bins[-1]
data_gen.init()

if cnf.task in cnf.language_tasks:
    task = data_gen.find_language_task(cnf.task)
    task.prepare_data()
    data_gen.collect_bins()
    data_gen.print_bin_usage()
else:
    for l in range(1, max_length + 1):
        # n_examples = min(cnf.data_size, cnf.data_size * 32 // l)
        n_examples = cnf.data_size
        data_gen.init_data(cnf.task, l, n_examples, cnf.n_input)
    data_gen.collect_bins()
    if len(data_gen.train_set[cnf.task][cnf.forward_max]) == 0:
        data_gen.init_data(cnf.task, cnf.forward_max, cnf.test_data_size, cnf.n_input)

data_supplier = data_feeder.create_data_supplier()

# Perform training
with tf.Graph().as_default():
    learner = ShuffleExchangeModel(cnf.n_hidden, cnf.bins, cnf.n_input, countList, cnf.n_output, cnf.dropout_keep_prob,
                                   create_translation_model=cnf.task in cnf.language_tasks,
                                   use_two_gpus=cnf.use_two_gpus)
    learner.create_graph()
    learner.variable_summaries = tf.summary.merge_all()
    tf.get_variable_scope().reuse_variables()
    learner.create_test_graph(cnf.forward_max)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=cnf.tf_config) as sess:
        sess.run(tf.global_variables_initializer())

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = "{time}_{task}".format(time=current_time, task=cnf.task)
        output_dir = os.path.join(cnf.out_dir, run_name)
        train_writer = tf.summary.FileWriter(output_dir)

        if cnf.load_prev:
            saver1 = tf.train.Saver([var for var in tf.trainable_variables()])
            saver1.restore(sess, cnf.model_file)

        batch_xs, batch_ys = data_supplier.supply_validation_data(max_length, cnf.batch_size)
        step = 1
        loss = 0
        avgLoss = 0
        acc = 1
        prev_loss = [1000000] * 7
        start_time = time.time()
        batch_xs_long, batch_ys_long = data_supplier.supply_test_data(cnf.forward_max, cnf.batch_size)
        long_accuracy, _, _ = learner.get_accuracy(sess, batch_xs_long, batch_ys_long)
        print("Iter", 0, "time =", 0)
        print("accuracy on test length", cnf.forward_max, "=", long_accuracy)

        while step < cnf.training_iters:
            if step % cnf.display_step == 0:
                avgLoss /= cnf.display_step
                step_time = time.time() - start_time
                start_time = time.time()
                lr = learner.get_learning_rate(sess)
                if step % 10000 == 0: saver.save(sess, cnf.model_file)
                print("Iter", step, "time =", step_time, "lr =", lr, 'max_loss =', loss, 'min_accuracy =', acc,
                      'avgLoss =', avgLoss)
                summaries = learner.print_loss(sess, batch_xs, batch_ys)
                train_writer.add_summary(summaries, step)

                batch_xs_long, batch_ys_long = data_supplier.supply_test_data(cnf.forward_max, cnf.batch_size)
                long_accuracy, _, test_summary = learner.get_accuracy(sess, batch_xs_long, batch_ys_long)
                train_writer.add_summary(test_summary, step)
                print("accuracy on length", cnf.forward_max, "=", long_accuracy)

                # decrease learning rate if no progress is made in 4 checkpoints
                prev_loss.append(avgLoss)
                if min(prev_loss[-3:]) > min(prev_loss[-4:]):
                    prev_loss = [1000000] * 7
                    sess.run(learner.lr_decay_op)
                loss = 0
                acc = 1
                avgLoss = 0
                avgRegul = 0

            batch_xs, batch_ys = data_supplier.supply_training_data(max_length, cnf.batch_size)
            loss1, acc1, perItemCost, costList = learner.train(sess, batch_xs, batch_ys, 1)
            avgLoss += loss1

            loss = max(loss, loss1)
            acc = min(acc, acc1)
            step += 1

        print("Optimization Finished!")
        saver.save(sess, cnf.model_file)

# test generalization to longer examples
test_length = 8
data_gen.init()

while test_length < cnf.max_test_length:
    if len(data_gen.test_set[cnf.task][test_length]) == 0:
        data_gen.init_data(cnf.task, test_length, cnf.test_data_size, cnf.n_input)

    tmp_length = test_length
    while len(data_gen.test_set[cnf.task][tmp_length]) == 0 and tmp_length > 0:
        data_gen.init_data(cnf.task, tmp_length, cnf.test_data_size, cnf.n_input)
        data_gen.collect_bins()
        tmp_length -= 1

    data_gen.reset_counters()
    batchSize = 1
    if test_length < 2000: batchSize = 16
    if test_length < 800: batchSize = 128

    with tf.Graph().as_default():  # , tf.device('/cpu:0'):
        tester = ShuffleExchangeModel(cnf.n_hidden, [test_length], cnf.n_input, [batchSize], cnf.n_output,
                                      cnf.dropout_keep_prob)
        tester.create_test_graph(test_length)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=cnf.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, cnf.model_file)
            errors, seq_errors, total = 0, 0, 0
            for iter in range(cnf.test_data_size // batchSize):
                batch_xs, batch_ys = data_supplier.supply_test_data(test_length, batchSize)
                acc1, test_result, _ = tester.get_accuracy(sess, batch_xs, batch_ys)
                er, tot, seq_er = data_gen.accuracy(batch_xs[0], test_result, batch_ys[0], batchSize, 0)
                errors += er
                seq_errors += seq_er
                total += tot

            acc_real = 1.0 - float(errors) / total
            print("Testing length:", test_length, "accuracy=", acc_real, "errors =", errors, "incorrect sequences=", seq_errors)
    test_length = test_length * 2
