from __future__ import print_function

import os
import re
from glob import glob
import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file


# regex for renaming the tensors to their corresponding Keras counterpart
re_repeat = re.compile(r'Repeat_[0-9_]*b')
re_block8 = re.compile(r'Block8_[A-Za-z]')


def get_filename(key):
    """Rename tensor name to the corresponding Keras layer weight name.
    # Arguments
        key: tensor name in TF (determined by tf.variable_scope)
    """
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('efficientnet-b0_', '')

    # remove "Repeat" scope from filename
    filename = re_repeat.sub('B', filename)

    if re_block8.match(filename):
        # the last block8 has different name with the previous 9 occurrences
        filename = filename.replace('Block8', 'Block8_10')
    elif filename.startswith('Logits'):
        # remove duplicate "Logits" scope
        filename = filename.replace('Logits_', '', 1)

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_checkpoint_file(filename, output_folder='weights'):
    """Extract tensors from a TF checkpoint file.
    # Arguments
        filename: TF checkpoint file
        output_folder: where to save the output numpy array files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue
        if 'stem' in key:
            print(key)
            print(reader.get_tensor(key))

        # convert tensor name into the corresponding Keras layer weight name and save
        path = os.path.join(output_folder, get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)
        # print("tensor_name: ", key)


# extract_tensors_from_checkpoint_file('efficientnet-b0/model.ckpt-109400')
