import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from layer_utils import activation_quant, conv2d_noise, dense_noise


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 relu_decay=0.0,
                 batch_normalization=True,
                 conv_first=True,
                 activation_bits=None,
                 weight_noise_train=None,
                 weight_noise_test=None,
                 weight_bits=None,
                 trainable=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    if weight_noise_train is None and weight_noise_test is None and weight_bits is None:
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4),
                      trainable=trainable)
    else:
        conv = conv2d_noise(num_filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            noise_train=weight_noise_train,
                            noise_test=weight_noise_test,
                            num_bits=weight_bits,
                            trainable=trainable)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(trainable=trainable)(x)
        if activation is not None:
            if activation_bits is None:
                x = Activation(activation)(x)
            else:
                x = activation_quant(num_bits=activation_bits, max_value=3, decay=relu_decay, trainable=trainable)(x)
    else:
        if batch_normalization:
            x = BatchNormalization(trainable=trainable)(x)
        if activation is not None:
            if activation_bits is None:
                x = Activation(activation)(x)
            else:
                x = activation_quant(num_bits=activation_bits, max_value=3, decay=relu_decay, trainable=trainable)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10, activation_bits=None, relu_decay=0.0, weight_noise_train=None, weight_noise_test=None,
              weight_bits=None, trainable_conv=True, trainable_dense=True):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters,
                     activation_bits=activation_bits,
                     relu_decay=relu_decay,
                     weight_noise_train=weight_noise_train,
                     weight_noise_test=weight_noise_test,
                     weight_bits=weight_bits,
                     trainable=trainable_conv)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             activation_bits=activation_bits,
                             relu_decay=relu_decay,
                             weight_noise_train=weight_noise_train,
                             weight_noise_test=weight_noise_test,
                             weight_bits=weight_bits,
                             trainable=trainable_conv)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             activation_bits=activation_bits,
                             weight_noise_train=weight_noise_train,
                             weight_noise_test=weight_noise_test,
                             weight_bits=weight_bits,
                             trainable=trainable_conv)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 activation_bits=activation_bits,
                                 weight_noise_train=weight_noise_train,
                                 weight_noise_test=weight_noise_test,
                                 weight_bits=weight_bits,
                                 trainable=trainable_conv)
            x = keras.layers.add([x, y])
            if (activation_bits is None) or (stack == 2 and res_block == num_res_blocks - 1):
                x = Activation('relu')(x)
            else:
                x = activation_quant(num_bits=activation_bits, max_value=3, decay=relu_decay, trainable=trainable_conv)(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    if activation_bits is not None:
        y = activation_quant(num_bits=activation_bits, decay=relu_decay, max_value=3, trainable=trainable_dense)(y)
    if weight_noise_train is None and weight_noise_test is None and weight_bits is None:
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal',
                        trainable=trainable_dense)(y)
    else:
        outputs = dense_noise(num_classes,
                              activation='softmax',
                              noise_train=weight_noise_train,
                              noise_test=weight_noise_test,
                              num_bits=weight_bits,
                              trainable=trainable_dense)(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10, activation_bits=None, relu_decay=0.0, weight_noise_train=None, weight_noise_test=None,
              weight_bits=None, trainable_conv=True, trainable_dense=True):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths

    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True,
                     activation_bits=activation_bits,
                     relu_decay=relu_decay,
                     weight_noise_train=weight_noise_train,
                     weight_noise_test=weight_noise_test,
                     weight_bits=weight_bits,
                     trainable=trainable_conv)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             activation_bits=activation_bits,
                             relu_decay=relu_decay,
                     		 weight_noise_train=weight_noise_train,
                             weight_noise_test=weight_noise_test,
                             weight_bits=weight_bits,
                             trainable=trainable_conv)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False,
                             activation_bits=activation_bits,
                             relu_decay=relu_decay,
                     		 weight_noise_train=weight_noise_train,
                             weight_noise_test=weight_noise_test,
                             weight_bits=weight_bits,
                             trainable=trainable_conv)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False,
                             activation_bits=activation_bits,
                             relu_decay=relu_decay,
                     		 weight_noise_train=weight_noise_train,
                             weight_noise_test=weight_noise_test,
                             weight_bits=weight_bits,
                             trainable=trainable_conv)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 activation_bits=activation_bits,
                     			 weight_noise_train=weight_noise_train,
                                 weight_noise_test=weight_noise_test,
                                 weight_bits=weight_bits,
                                 trainable=trainable_conv)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization(trainable=trainable_conv)(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    if activation_bits is None:
        y = Activation('relu')(y)
    else:
        y = activation_quant(num_bits=activation_bits, max_value=3, decay=relu_decay, trainable=trainable_dense)(y)
    if weight_noise_train is None and weight_noise_test is None and weight_bits is None:
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal',
                        trainable=trainable_dense)(y)
    else:
        outputs = dense_noise(num_classes,
                              activation='softmax',
                              noise_train=weight_noise_train,
                              noise_test=weight_noise_test,
                              num_bits=weight_bits,
                              trainable=trainable_dense)(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model