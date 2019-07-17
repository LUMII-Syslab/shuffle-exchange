"""
    Training configuration
"""
import numpy as np
import tensorflow as tf

"""
    TensorFlow configuration
"""
tf_config = tf.ConfigProto()
tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

"""
    Model configuration
"""
dropout_keep_prob = 1.0
input_dropout_keep_prob = 1.0
input_word_dropout_keep_prob = 1.0
label_smoothing = 0.0

""" 
    Test configuration
"""
max_test_length = 5000
test_data_size = 1024

"""
    Local storage (checkpoints, etc).
"""
use_two_gpus = False
gpu_instance = "0"
out_dir = "/host-dir/gpu" + gpu_instance
model_file = out_dir + "/varWeights.ckpt"
image_path = out_dir + "/images"

"""
    Logging
"""
log_filename = ""

"""
    Training config
"""
training_iters = 20001
display_step = 100
batch_size = 32
data_size = 10000
bins = [8, 16, 32, 64]
n_Benes_blocks = 2
load_prev = False

forward_max = 128
bin_max_len = max(max_test_length, forward_max)

"""
    Lambada configuration
"""
lambada_data_dir = "/host-dir/lambada-dataset"
lambada_vocab_size = 999996

# Data preparation
use_front_padding = False  # randomly shift the starting position of the sequence in the bin
disperse_padding = False  # insert random blanks in the sequence

"""
    Embedding configuration
"""
use_pre_trained_embedding = False
base_folder = "/host-dir/embeddings/"
embedding_file = base_folder + "fast_word_embedding.vec"
emb_vector_file = base_folder + "emb_vectors.bin"
emb_word_dictionary = base_folder + "word_dict.bin"

"""
    Task configuration.
"""

all_tasks = {"sort", "kvsort", "id", "rev", "rev2", "incr", "add", "left",
             "right", "bmul", "mul", "dup",
             "badd", "qadd", "search", "qmul", "mulbcd", "shuffle", "div",
             "w_sort", "lambada"}

language_tasks = {"lambada"}

# suggested settings for binary addition
task = "badd"
n_input = 13  # range of input digits
n_output = 4  # range of output digits
n_hidden = 48 * 2  # number of maps
n_Benes_blocks = 1

# suggested settings for sequence reversal
# task = "rev"
# n_input = 12  # range of input digits
# n_output = 12  # range of output digits
# n_hidden = 48 * 2  # number of maps
# n_Benes_blocks = 1

# suggested settings for sequence duplication
# task = "dup"
# n_input = 12  # range of input digits
# n_output = 12  # range of output digits
# n_hidden = 48 * 2  # number of maps
# n_Benes_blocks = 1

# suggested settings for binary multiplication
# task = "bmul"
# n_input = 13  # range of input digits
# n_output = 4  # range of output digits
# n_hidden = 48 * 8  # number of maps
# forward_max = bins[-1]

# suggested settings for sorting numbers in range 1 to 12
# task = "sort"
# n_input = 12
# n_output = 12
# n_hidden = 48*2 # number of maps
# n_Benes_blocks = 1

initial_learning_rate = 0.0025 * np.sqrt(96 / n_hidden)
min_learning_rate = initial_learning_rate / 40


# suggested settings for LAMBADA question answering
# task = "lambada"
# bins = [128]
# batch_size = 64
# n_input = lambada_vocab_size
# n_output = 3
# n_hidden = 48 * 8
# input_word_dropout_keep_prob = 0.95
# use_front_padding = True
# use_pre_trained_embedding = False # set this flag to True and provide an embedding for the best results
# disperse_padding = True #we trained with this setting but are not sure if it helps
# label_smoothing = 0.1
# min_learning_rate = initial_learning_rate = 2e-4

if load_prev:
    initial_learning_rate = min_learning_rate
