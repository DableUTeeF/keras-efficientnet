from . import efficientnet_layers as el
from keras import models, layers


def keras_efficientnet(blocks_args, global_params, training=False, input_image=None):
    inp = input_image if input_image is not None else layers.Input((224, 224, 3))
    x = layers.Conv2D(32, 3, padding='same', strides=2, name='stem_conv2d', use_bias=False)(inp)
    x = layers.BatchNormalization(name='stem_tpu_batch_normalization')(x)
    x = layers.Lambda(lambda x: el.relu_fn(x))(x)
    # x = layers.Activation('swish')(x)
    idx = 0
    for block in blocks_args:
        x = el.mbConvBlock(x, block, global_params, idx, training=training)
        # x = MBConvBlock(block, global_params, idx)(x, training=training)
        idx += 1
        if block.num_repeat > 1:
            block = block._replace(
                input_filters=block.output_filters, strides=[1, 1])
        for _ in range(block.num_repeat - 1):
            x = el.mbConvBlock(x, block, global_params, idx, training=training)
            idx += 1
    x = layers.Conv2D(1280, 1, name='head_conv2d', use_bias=False)(x)
    x = layers.BatchNormalization(name='head_tpu_batch_normalization')(x)
    x = layers.Lambda(lambda x: el.relu_fn(x))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation='softmax', name='head_dense', )(x)
    model = models.Model(inp, x, name='efficientnet-b0')
    return model
