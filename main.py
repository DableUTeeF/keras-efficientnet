from keras_model import keras_efficientnet
import efficientnet_builder as eb
import preprocessing
import tensorflow as tf
import numpy as np
import json


if __name__ == '__main__':
    model_name = 'efficientnet-b0'
    labels_map_file = 'eval_data/labels_map.txt'
    image_file = 'eval_data/panda.jpg'
    training = False
    blocks_args, global_params = eb.get_model_params(model_name, None)
    model = keras_efficientnet(blocks_args, global_params, training)
    model.load_weights('models/efficientnet_b0_weights_tf_dim_ordering_tf_kernels.h5')

    MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    image_string = tf.read_file(image_file)
    image_decoded = preprocessing.preprocess_image(image_string, training, 224)
    image = tf.cast(image_decoded, tf.float32)
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    y = model.predict(tf.expand_dims(image, 0), steps=1)[0]

    label_map = json.loads(open(labels_map_file).read())
    pred_idx = np.argsort(y)[::-1]
    print(np.argmax(y))
    print('truth: 388')
    print(pred_idx[:5])
    print([y[pid] for pid in pred_idx[:5]])
    for i in range(5):
        print('  -> top_{} ({:4.2f}%): {}  '.format(i+1, y[pred_idx[i]] * 100, label_map[str(pred_idx[i])]))
