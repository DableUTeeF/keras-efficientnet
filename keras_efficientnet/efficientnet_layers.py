from tensorflow.contrib.keras import layers
from .import efficientnet_model as em
import tensorflow as tf


class ReduceMean(layers.Layer):
    def call(self, ip):
        spatial_dims = [1, 2]
        x = ip
        return tf.keras.backend.mean(x, spatial_dims, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape


class SigmoidMul(layers.Layer):
    def call(self, ip):
        x, se_expand = ip
        return tf.sigmoid(se_expand) * x

    def compute_output_shape(self, input_shape):
        return input_shape


def mbConvBlock(inputs, block_args, global_params, idx, training=True, drop_connect_rate=None):
    filters = block_args.input_filters * block_args.expand_ratio
    batch_norm_momentum = global_params.batch_norm_momentum
    batch_norm_epsilon = global_params.batch_norm_epsilon
    has_se = (block_args.se_ratio is not None) and (block_args.se_ratio > 0) and (block_args.se_ratio <= 1)
    x = inputs
    # block_name = 'efficientnet-b0_' + 'blocks_' + str(idx) + '_'
    block_name = 'blocks_' + str(idx) + '_'
    project_conv_name = block_name + 'conv2d'
    project_bn_name = block_name + 'tpu_batch_normalization_1'
    ndbn_name = block_name + 'tpu_batch_normalization'
    if block_args.expand_ratio != 1:
        # Expansion phase:
        expand_conv = layers.Conv2D(filters,
                                    kernel_size=[1, 1],
                                    strides=[1, 1],
                                    kernel_initializer=em.conv_kernel_initializer,
                                    padding='same',
                                    use_bias=False,
                                    name=project_conv_name
                                    )(x)
        bn0 = em.batchnorm(momentum=batch_norm_momentum,
                           epsilon=batch_norm_epsilon,
                           name=ndbn_name)(expand_conv)
        project_conv_name = block_name + 'conv2d_1'
        ndbn_name = block_name + 'tpu_batch_normalization_1'
        project_bn_name = block_name + 'tpu_batch_normalization_2'

        x = layers.Lambda(lambda x: em.relu_fn(x))(bn0)

    kernel_size = block_args.kernel_size
    # Depth-wise convolution phase:

    depthwise_conv = em.utils.DepthwiseConv2D(
        [kernel_size, kernel_size],
        strides=block_args.strides,
        depthwise_initializer=em.conv_kernel_initializer,
        padding='same',
        use_bias=False,
        name=block_name + 'depthwise_conv2d'
    )(x)
    bn1 = em.batchnorm(momentum=batch_norm_momentum,
                       epsilon=batch_norm_epsilon,
                       name=ndbn_name
                       )(depthwise_conv)
    x = layers.Lambda(lambda x: em.relu_fn(x))(bn1)

    if has_se:
        num_reduced_filters = max(
            1, int(block_args.input_filters * block_args.se_ratio))
        # Squeeze and Excitation layer.
        se_tensor = ReduceMean()(x)

        se_reduce = layers.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=em.conv_kernel_initializer,
            padding='same',
            name=block_name + 'se_' + 'conv2d',
            use_bias=True)(se_tensor)
        se_reduce = layers.Lambda(lambda x: em.relu_fn(x))(se_reduce)
        se_expand = layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=em.conv_kernel_initializer,
            padding='same',
            name=block_name + 'se_' + 'conv2d_1',
            use_bias=True)(se_reduce)
        x = SigmoidMul()([x, se_expand])

    # Output phase:
    filters = block_args.output_filters
    project_conv = layers.Conv2D(
        filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=em.conv_kernel_initializer,
        padding='same',
        name=project_conv_name,
        use_bias=False)(x)
    x = em.batchnorm(momentum=batch_norm_momentum,
                     epsilon=batch_norm_epsilon,
                     name=project_bn_name
                     )(project_conv)
    # x = layers.Lambda(lambda x: em.relu_fn(x))(bn2)
    if block_args.id_skip:
        if all(
                s == 1 for s in block_args.strides
        ) and block_args.input_filters == block_args.output_filters:
            # only apply drop_connect if skip presents.
            if drop_connect_rate:
                x = em.utils.drop_connect(x, training, drop_connect_rate)
            x = layers.add([x, inputs])
    return x
