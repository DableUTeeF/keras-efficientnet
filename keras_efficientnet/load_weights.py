import os
import numpy as np
from tqdm import tqdm
from . import utils
from . import efficientnet_builder as eb
from .keras_model import keras_efficientnet

WEIGHTS_DIR = '/home/palm/tpu/models/official/efficientnet/weights'
MODEL_DIR = './models'
OUTPUT_WEIGHT_FILENAME = 'efficientnet_b0_weights_tf_dim_ordering_tf_kernels.h5'
model_name = 'efficientnet-b0'

BatchNormalization = utils.TpuBatchNormalization
print('Instantiating an empty efficientnet model...')

blocks_args, global_params = eb.get_model_params(model_name, None)
model = keras_efficientnet(blocks_args, global_params)

print('Loading weights from', WEIGHTS_DIR)
for layer in tqdm(model.layers):
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(':0', '')
            weight_file = layer.name + '_' + weight_name + '.npy'
            weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))

            # remove the "background class"
            if weight_file.startswith('Logits_bias'):
                weight_arr = weight_arr[1:]
            elif weight_file.startswith('Logits_kernel'):
                weight_arr = weight_arr[:, 1:]

            weights.append(weight_arr)
        layer.set_weights(weights)


print('Saving model weights...')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
model.save_weights(os.path.join(MODEL_DIR, OUTPUT_WEIGHT_FILENAME))
