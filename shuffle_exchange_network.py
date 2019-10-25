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

import tensorflow as tf
from tensorflow.contrib.framework import smart_cond

is_training = None
dropout_keep_prob = 1.0

# lists to collect debug data for tensorboard
gate_mem = []
reset_mem = []
prev_mem_list = []
candidate_mem = []


def ror(x, n, p=1):
    """Bitwise rotation right"""
    return (x >> p) + ((x & ((1 << p) - 1)) << (n - p))


def rol(x, n, p=1):
    """Bitwise rotation left"""
    return ((x << p) & ((1 << n) - 1)) | (x >> (n - p))


def dropout(d, len):
    """Dropout dependent on sequence length"""
    if dropout_keep_prob < 1:
        prob = (1.0 - dropout_keep_prob) / len
        d = smart_cond(is_training, lambda: tf.nn.dropout(d, rate=prob), lambda: d)
    return d


def add_noise_mul(d, noise_scale):
    """Inject multiplicative noise"""
    d = smart_cond(is_training,
                   lambda: d * tf.random_normal(tf.shape(d), mean=1.0, stddev=noise_scale),
                   lambda: d)
    return d


def add_noise_add(d, noise_scale):
    """Inject additive noise"""
    d = smart_cond(is_training,
                   lambda: d + tf.random_normal(tf.shape(d), stddev=noise_scale),
                   lambda: d)
    return d


activation_fn = tf.tanh
gate_fn = tf.sigmoid


def conv_linear(input, kernel_width, nin, nout, bias_start, prefix, add_bias=True, init_scale=1.0):
    """Convolutional linear map"""

    with tf.variable_scope(prefix):
        if kernel_width == 1:
            inp_shape = input.get_shape().as_list()
            filter = tf.get_variable("CvK", [nin, nout], initializer=tf.variance_scaling_initializer(
                scale=init_scale, mode="fan_avg", distribution="uniform"))
            res = tf.matmul(tf.reshape(input, [inp_shape[0] * inp_shape[1], nin]), filter)
            res = tf.reshape(res, [inp_shape[0], inp_shape[1], nout])
        else:
            filter = tf.get_variable("CvK", [kernel_width, nin, nout])
            res = tf.nn.conv1d(input, filter, 1, "SAME")

        bias_term = tf.get_variable("CvB", [nout], initializer=tf.constant_initializer(0.0))
        res = res + bias_start + bias_term

        return res


def shuffle_layer(mem, do_ror=True):
    """Shuffles the elements according to bitwise left or right rotation on their indices"""
    length = mem.get_shape().as_list()[1]
    n_bits = (length - 1).bit_length()
    if do_ror:
        rev_indices = [ror(x, n_bits) for x in range(length)]
    else:
        rev_indices = [rol(x, n_bits) for x in range(length)]
    mem_shuffled = tf.gather(mem, rev_indices, axis=1)
    return mem_shuffled


def switch_layer(mem_shuffled, kernel_width, prefix, residual_input=None, perform_swap=True):
    """Computation unit for every two adjacent elements"""
    length = mem_shuffled.get_shape().as_list()[1]
    num_units = mem_shuffled.get_shape().as_list()[2]
    batch_size = mem_shuffled.get_shape().as_list()[0]
    n_bits = (length - 1).bit_length()

    def linear2(input, suffix, bias_start, in_units, out_units):
        # 2 input to 2 output linear map
        input = tf.reshape(input, [batch_size, length // 2, in_units * 2]) #reshape to form pairs. Produces a twice shorter sequence
        res = conv_linear(input, kernel_width, in_units * 2, out_units * 2, bias_start, prefix + "/" + suffix) # perform linear mapping
        res = tf.reshape(res, [batch_size, length, out_units]) # reshape back
        return res

    def gated_linear2(input, suffix, bias_start_reset, in_units, out_units):
        # linear mapping with two reset gates
        input = tf.reshape(input, [batch_size, length // 2, in_units * 2])
        reset1 = gate_fn(conv_linear(input, kernel_width, in_units * 2, in_units * 2, bias_start_reset, prefix + "/reset1/" + suffix))
        reset2 = gate_fn(conv_linear(input, kernel_width, in_units * 2, in_units * 2, bias_start_reset, prefix + "/reset2/" + suffix))
        res1 = conv_linear(input * reset1, kernel_width, in_units * 2, out_units, 0.0, prefix + "/cand1/" + suffix)
        res2 = conv_linear(input * reset2, kernel_width, in_units * 2, out_units, 0.0, prefix + "/cand2/" + suffix)
        res = tf.concat([res1, res2], axis=2)
        res = tf.reshape(res, [batch_size, length, out_units])
        return activation_fn(res), tf.reshape(reset1, [batch_size, length, in_units])

    mem_shuffled_x = mem_shuffled
    if perform_swap:
        xor_indices = [x ^ 1 for x in range(length)]
        #mem_xor = tf.gather(mem_shuffled[:, :, num_units // 2:], xor_indices, axis=1)
        #mem_shuffled_x = tf.concat([mem_shuffled[:, :, :num_units // 2], mem_xor], axis=2)
        mem_xor = tf.gather(mem_shuffled[:, :, :num_units // 2], xor_indices, axis=1)
        mem_shuffled_x = tf.concat([mem_xor, mem_shuffled[:, :, num_units // 2:]], axis=2)

    if residual_input is None:
        mem_all = mem_shuffled
    else:
        residual_scale = tf.get_variable(prefix + "/residual_scale", [num_units], initializer=tf.constant_initializer(0.5))
        prev_mem_list.append(tf.reshape(tf.clip_by_value(residual_scale, 0.0, 1.0), [1, num_units, 1]))
        mem_all = mem_shuffled + residual_input * residual_scale

    # calculate the new value
    candidate, reset = gated_linear2(mem_all, "c", 0.5, num_units, num_units)
    reset_mem.append(reset[:, :, :num_units])
    gate = gate_fn(linear2(mem_all, "g", 0.5, num_units, num_units))

    gate_mem.append(gate)
    candidate_mem.append(candidate)
    candidate = gate * mem_shuffled_x + (1 - gate) * candidate
    candidate = dropout(candidate, n_bits)  # dropout does not give observable benefit
    if dropout_keep_prob == 1.0: candidate = add_noise_mul(candidate, 0.001)

    return candidate

def shuffle_exchange_network(cur, name, kernel_width=1, n_blocks=1, tied_inner_weights=True, tied_outer_weights=False):
    """Neural Benes Network with skip connections between blocks."""
    length = cur.get_shape().as_list()[1]
    num_units = cur.get_shape().as_list()[2]
    n_bits = (length - 1).bit_length()
    allMem = []

    with tf.variable_scope(name + "_shuffle_exchange", reuse=tf.AUTO_REUSE):
        stack = []
        for k in range(n_blocks):
            outstack = []
            for i in range(n_bits - 1):
                outstack.append(cur)
                layer_name = "forward"
                prev = stack[i] if len(stack) > 0 else None
                if not tied_outer_weights: layer_name = str(k) + "_" + layer_name
                if not tied_inner_weights: layer_name += "_" + str(i)
                cur = switch_layer(cur, kernel_width, layer_name, residual_input=prev)
                allMem.append(cur)
                cur = shuffle_layer(cur, do_ror=False)

            for i in range(n_bits - 1):
                outstack.append(cur)
                layer_name = "reverse"
                prev = stack[i + n_bits - 1] if len(stack) > 0 else None
                if not tied_outer_weights: layer_name = str(k) + "_" + layer_name
                if not tied_inner_weights: layer_name += "_" + str(n_bits - 1 -1 - i)
                cur = switch_layer(cur, kernel_width, layer_name, residual_input=prev)
                allMem.append(cur)
                cur = shuffle_layer(cur, do_ror=True)
            stack = outstack

        layer_name = "last"
        prev = stack[0] if len(stack) > 0 and n_blocks>=2 else None
        cur = switch_layer(cur, kernel_width, layer_name, residual_input=prev)
        allMem.append(cur)

    return cur, allMem

def shuffle_exchange_network_less_sharing(cur, name, kernel_width=1, n_blocks=1, tied_inner_weights=True, tied_outer_weights=False):
    """Neural Benes Network with residual connections between blocks."""
    length = cur.get_shape().as_list()[1]
    n_bits = (length - 1).bit_length()
    allMem = []

    with tf.variable_scope(name + "/shuffle_exchange", reuse=tf.AUTO_REUSE):
        outstack = []
        stack = []

        def switch_and_shuffle(cur, do_ror, layer_name, block_index, layer_index, perform_swap=True):
            prev = stack[layer_index] if len(stack) > 0 else None
            if not tied_outer_weights or prev is None: layer_name = str(block_index) + "_" + layer_name
            if not tied_inner_weights: layer_name += "_" + str(layer_index)
            cur = switch_layer(cur, kernel_width, layer_name, residual_input=prev, perform_swap=perform_swap)
            cur = shuffle_layer(cur, do_ror=do_ror)
            outstack.append(cur)
            allMem.append(cur)
            return cur

        for k in range(n_blocks):
            layer_ind = 0
            outstack = []
            outstack.append(cur)
            cur = switch_and_shuffle(cur, False, "first_layer", k, layer_ind, perform_swap=False)
            layer_ind+=1

            for i in range(n_bits - 2):
                cur = switch_and_shuffle(cur, False, "forward", k, layer_ind)
                layer_ind+=1

            cur = switch_and_shuffle(cur, True, "middle_layer", k, layer_ind, perform_swap=False)
            layer_ind += 1

            for i in range(n_bits - 2):
                cur = switch_and_shuffle(cur, True, "backward", k, layer_ind)
                layer_ind+=1

            stack = outstack

        prev = stack[0] if n_blocks >= 2 else None
        cur = switch_layer(cur, kernel_width, "last_layer", residual_input=prev, perform_swap=False)
        allMem.append(cur)

    return cur, allMem
