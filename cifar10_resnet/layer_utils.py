import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class activation_quant(Layer):
    def __init__(self, num_bits, max_value, decay=0, **kwargs):
        super(activation_quant, self).__init__(**kwargs)
        self.num_bits = num_bits
        self.max_value = max_value
        self.decay = decay
        
    def build(self, input_shape):
        if self.num_bits is not None:
            self.relux = self.add_weight(name='relux',
                                         shape=[],
                                         regularizer=tf.keras.regularizers.l2(self.decay),
                                         initializer=keras.initializers.Constant(self.max_value),
                                         trainable=self.trainable)

    def call(self, x):
        if self.num_bits is not None:
            act = tf.maximum(tf.minimum(x, self.relux), 0)
            act = tf.quantization.fake_quant_with_min_max_vars(
                act,
                min=tf.constant(0, dtype=tf.float32),
                max=self.relux,
                num_bits=self.num_bits,
                narrow_range=False)
        else:
            act = K.relu(x)
        return act
    
    def compute_output_shape(self, input_shape):
        return input_shape

    
class conv2d_noise(Layer):
    def __init__(self, num_filter, kernel_size=3, activation=None, strides=1, padding='valid',
                 noise_train=0., noise_test=0., num_bits=None, weight_range=1., bias_range=1.,
                 range_decay=0, **kwargs):
        super(conv2d_noise, self).__init__(**kwargs)
        self.num_filter = num_filter
        self.noise_train = noise_train
        self.noise_test = noise_test
        self.kernel_size = (kernel_size, kernel_size)
        self.activation = activation
        self.strides = (strides, strides)
        self.padding = padding
        self.num_bits = num_bits
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.range_decay = range_decay
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=self.kernel_size + (int(input_shape[3]), self.num_filter),
                                      initializer='glorot_uniform')
        self.bias = self.add_weight(name='bias',
                                    shape=(self.num_filter,),
                                    initializer=keras.initializers.Zeros())
        if self.num_bits is not None:
            self.weight_range = self.add_weight(name='weight_range',
                                                shape=[],
                                                regularizer=tf.keras.regularizers.l2(self.range_decay),
                                                initializer=keras.initializers.Constant(self.weight_range))
            # self.bias_range = self.add_weight(name='bias_range',
            #                                   shape=[],
            #                                   regularizer=tf.keras.regularizers.l2(self.range_decay),
            #                                   initializer=keras.initializers.Constant(self.bias_range))
        
    def call(self, x, training=None):
        weights = self.kernel
        bias = self.bias
        if self.num_bits is not None:
            weights = tf.maximum(tf.minimum(weights, self.weight_range), -self.weight_range)
            bias = tf.maximum(tf.minimum(bias, self.bias_range), -self.bias_range)
            weights = tf.quantization.fake_quant_with_min_max_vars(
                weights,
                min=-self.weight_range,
                max=self.weight_range,
                num_bits=self.num_bits,
                narrow_range=True)
            # bias = tf.quantization.fake_quant_with_min_max_vars(
            #     bias,
            #     min=-self.bias_range,
            #     max=self.bias_range,
            #     num_bits=self.num_bits,
            #     narrow_range=True)
        noise_magnitude = self.noise_train if training else self.noise_test
        if noise_magnitude is not None and noise_magnitude > 0:
            w_max = K.max(K.abs(weights))
            weights = weights + tf.random.normal(self.kernel.shape, mean=0, stddev=w_max * noise_magnitude)
            bias = bias + tf.random.normal(self.bias.shape, stddev=w_max * noise_magnitude)
        act = K.conv2d(x, weights, strides=self.strides, padding=self.padding)
        act = K.bias_add(act, bias)
        if self.activation == 'relu':
            act = K.relu(act)
        return act
    
    def compute_output_shape(self, input_shape):
        hei = conv_utils.conv_output_length(input_shape[1], self.kernel_size[0], self.padding, self.strides[0])
        wid = conv_utils.conv_output_length(input_shape[2], self.kernel_size[1], self.padding, self.strides[1])
        return (int(input_shape[0]), hei, wid, self.num_filter)

    
class dense_noise(Layer):
    def __init__(self, output_dim, activation=None, noise_train=0., noise_test=0.,
                 num_bits=None, weight_range=1., bias_range=1., range_decay=0, **kwargs):
        super(dense_noise, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.noise_train = noise_train
        self.noise_test = noise_test
        self.activation = activation
        self.num_bits = num_bits
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.range_decay = range_decay
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=[int(input_shape[1]), int(self.output_dim)],
                                      initializer='glorot_uniform')
        self.bias = self.add_weight(name='bias',
                                    shape=[int(self.output_dim)],
                                    initializer=keras.initializers.Zeros())
        if self.num_bits is not None:
            self.weight_range = self.add_weight(name='weight_range',
                                                shape=[],
                                                regularizer=tf.keras.regularizers.l2(self.range_decay),
                                                initializer=keras.initializers.Constant(self.weight_range))
            # self.bias_range = self.add_weight(name='bias_range',
            #                                   shape=[],
            #                                   regularizer=tf.keras.regularizers.l2(self.range_decay),
            #                                   initializer=keras.initializers.Constant(self.bias_range))
        
    def call(self, x, training=None):
        weights = self.kernel
        bias = self.bias
        if self.num_bits is not None:
            weights = tf.maximum(tf.minimum(weights, self.weight_range), -self.weight_range)
            bias = tf.maximum(tf.minimum(bias, self.bias_range), -self.bias_range)
            weights = tf.quantization.fake_quant_with_min_max_vars(
                weights,
                min=-self.weight_range,
                max=self.weight_range,
                num_bits=self.num_bits,
                narrow_range=True)
            # bias = tf.quantization.fake_quant_with_min_max_vars(
            #     bias,
            #     min=-self.bias_range,
            #     max=self.bias_range,
            #     num_bits=self.num_bits,
            #     narrow_range=True)
        noise_magnitude = self.noise_train if training else self.noise_test
        if noise_magnitude is not None and noise_magnitude > 0:
            w_max = K.max(K.abs(weights))
            weights = weights + tf.random.normal(self.kernel.shape, mean=0, stddev=w_max * noise_magnitude)
            bias = bias + tf.random.normal(self.bias.shape, stddev=w_max * noise_magnitude)
        act = K.dot(x, weights) + bias
        if self.activation == 'relu':
            act = K.relu(act)
        elif self.activation == 'softmax':
            act = K.softmax(act)
        return act
    
    def compute_output_shape(self, input_shape):
        return (int(input_shape[0]), self.output_dim)


class noise_injection(Layer):
    def __init__(self, noise_std, **kwargs):
        super(noise_injection, self).__init__(**kwargs)
        self.noise_std = noise_std

    def call(self, x, training=None):
        if training:    
            x = x + tf.random.normal(tf.shape(x), mean=0, stddev=self.noise_std)
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape