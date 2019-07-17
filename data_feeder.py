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

import random

import config as cnf
import data_utils as data_gen


class DataSupplier:

    def supply_training_data(self, length, batch_size) -> tuple:
        pass

    def supply_validation_data(self, length, batch_size) -> tuple:
        pass

    def supply_test_data(self, length, batch_size) -> tuple:
        pass


class DefaultSupplier(DataSupplier):
    def supply_training_data(self, length, batch_size) -> tuple:
        return self.__gen_training_data(True)

    def supply_validation_data(self, length, batch_size) -> tuple:
        return self.__gen_training_data(False)

    @staticmethod
    def __gen_training_data(for_training):
        x = []
        y = []

        for index, seq_len in enumerate(cnf.bins):
            data, labels = data_gen.get_batch(seq_len, cnf.batch_size, for_training, cnf.task)
            x += [data]
            y += [labels]

        return x, y

    def supply_test_data(self, length, batch_size):
        data, labels = data_gen.get_batch(length, batch_size, False, cnf.task)
        return [data], [labels]


def create_batch(generator, batch_size, length, for_training=False):
    qna = []
    while len(qna) < batch_size:
        question, answer = next(generator)
        if max(len(question), len(answer)) > length:
            continue

        question_and_answer = data_gen.add_padding(question, answer, length, for_training)
        qna.append(question_and_answer)

    random.shuffle(qna)
    questions, answers = tuple(zip(*qna))
    return [questions], [answers]


def create_data_supplier() -> DataSupplier:
    return DefaultSupplier()
