from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, Input, Flatten
from layer_utils import activation_quant, conv2d_noise, dense_noise
from layers_numpy import quantize_unsigned, quantize_rescale
import numpy as np
import os


batch_size = 128  # orig paper trained all networks with batch_size=128
epochs = 200
weight_noise_train = 0.2
weight_noise_test = 0.0
activation_bits = 3
num_classes = 10
num_segment_previous = 1
num_segment_current = 2

model_name = 'filter16_act%db_wnoise%.2f_input0.95' % (activation_bits, weight_noise_train)
model_type = 'ResNet%dv%d_%s' % (20, 1, model_name)
orig_model_name = 'fwd_finetune_conv2d_noise_2'#'cifar10_%s_model' % model_type

print('Model name: %s' % model_type)

finetune_previous_layer = 'add'
finetune_current_layer = 'conv2d_noise_3'
print('Finetuning input from %s' % finetune_current_layer)
ft_model_name = 'fwd_finetune_%s' % finetune_current_layer



def resnet20_finetune(activation_bits, weight_noise_train, weight_noise_test):
	# Build model for finetuning

	layer_input_0 = Input(shape=(32, 32, 16)) # a new input tensor to be able to feed the desired layer
	layer_input_1 = Input(shape=(32, 32, 16))

	# create the new nodes for each layer in the path

	# y = conv2d_noise(16, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise')(layer_input_0)
	# y = BatchNormalization(name='batch_normalization')(y)
	# y = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant')(layer_input_0)

	# Stack 1, block 1
	# x = conv2d_noise(16, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_1')(y)
	# x = BatchNormalization(name='batch_normalization_1')(x)
	# x = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_1')(layer_input_1)
	# x = conv2d_noise(16, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_2')(x)
	# x = BatchNormalization(name='batch_normalization_2')(x)
	# y = keras.layers.add([y, layer_input_1])
	y = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_2')(layer_input_0)

	# Stack 1, block 2
	# x = conv2d_noise(16, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_3')(y)
	# x = BatchNormalization(name='batch_normalization_3')(x)
	x = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_3')(layer_input_1)
	x = conv2d_noise(16, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_4')(x)
	x = BatchNormalization(name='batch_normalization_4')(x)
	y = keras.layers.add([y, x])
	y = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_4')(y)

	# Stack 1, block 3
	x = conv2d_noise(16, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_5')(y)
	x = BatchNormalization(name='batch_normalization_5')(x)
	x = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_5')(x)
	x = conv2d_noise(16, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_6')(x)
	x = BatchNormalization(name='batch_normalization_6')(x)
	y = keras.layers.add([y, x])
	y = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_6')(y)

	# Stack 2, block 1
	x = conv2d_noise(32, strides=2, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_7')(y)
	x = BatchNormalization(name='batch_normalization_7')(x)
	x = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_7')(x)
	x = conv2d_noise(32, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_8')(x)
	y = conv2d_noise(32, kernel_size=1, strides=2, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_9')(y)
	x = BatchNormalization(name='batch_normalization_8')(x)
	y = keras.layers.add([y, x])
	y = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_8')(y)

	# Stack 2, block 2
	x = conv2d_noise(32, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_10')(y)
	x = BatchNormalization(name='batch_normalization_9')(x)
	x = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_9')(x)
	x = conv2d_noise(32, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_11')(x)
	x = BatchNormalization(name='batch_normalization_10')(x)
	y = keras.layers.add([y, x])
	y = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_10')(y)

	# Stack 2, block 3
	x = conv2d_noise(32, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_12')(y)
	x = BatchNormalization(name='batch_normalization_11')(x)
	x = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_11')(x)
	x = conv2d_noise(32, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_13')(x)
	x = BatchNormalization(name='batch_normalization_12')(x)
	y = keras.layers.add([y, x])
	y = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_12')(y)

	# Stack 3, block 1
	x = conv2d_noise(64, strides=2, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_14')(y)
	x = BatchNormalization(name='batch_normalization_13')(x)
	x = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_13')(x)
	x = conv2d_noise(64, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_15')(x)
	y = conv2d_noise(64, kernel_size=1, strides=2, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_16')(y)
	x = BatchNormalization(name='batch_normalization_14')(x)
	y = keras.layers.add([y, x])
	y = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_14')(y)

	# Stack 3, block 2
	x = conv2d_noise(64, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_17')(y)
	x = BatchNormalization(name='batch_normalization_15')(x)
	x = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_15')(x)
	x = conv2d_noise(64, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_18')(x)
	x = BatchNormalization(name='batch_normalization_16')(x)
	y = keras.layers.add([y, x])
	y = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_16')(y)

	# Stack 3, block 3
	x = conv2d_noise(64, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_19')(y)
	x = BatchNormalization(name='batch_normalization_17')(x)
	x = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_17')(x)
	x = conv2d_noise(64, strides=1, padding='same', noise_train=weight_noise_train, noise_test=weight_noise_test, name='conv2d_noise_20')(x)
	x = BatchNormalization(name='batch_normalization_18')(x)
	y = keras.layers.add([y, x])

	y = AveragePooling2D(pool_size=8)(y)
	y = Flatten()(y)
	y = activation_quant(num_bits=activation_bits, max_value=3, name='activation_quant_18')(y)

	outputs = dense_noise(10, activation='softmax', noise_train=weight_noise_train, noise_test=weight_noise_test, name='dense_noise')(y)

	# create the model
	model = Model([layer_input_0, layer_input_1], outputs)
	model.compile(loss='categorical_crossentropy',
	              optimizer=Adam(1e-4),
	              metrics=['accuracy'])
	model.summary()
	return model


def build_model(activation_bits, weight_noise_train, weight_noise_test, weights_map):
	K.clear_session()
	model = resnet20_finetune(activation_bits, weight_noise_train, weight_noise_test)
	for klayer in model.layers:
	    if klayer.name in weights_map:
	        klayer.set_weights(weights_map[klayer.name])
	return model


# Load original model weights
# filepath = os.path.join(os.getcwd(), '%s.npy' % model_type)
# weights = np.load(filepath, allow_pickle=True).item()
ckpt_dir = os.path.join(os.getcwd(), model_type)
orig_path = os.path.join(ckpt_dir, orig_model_name)
orig_model = load_model(orig_path)
weights = {}
for olayer in orig_model.layers:
    weights[olayer.name] = olayer.get_weights()


# Load finetuning inputs
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

DATA_PATH = '/scratch/users/weierwan/forward_finetune/'
train_previous_inputs = 0
test_previous_inputs = 0
for i in range(num_segment_previous):
	train_previous_inputs = np.load(os.path.join(DATA_PATH, '%s_%d_train.npy' % (finetune_previous_layer, i)))
	# train_previous_inputs += (tmp['out_chip'] - tmp['intercept']) / tmp['slope']
	test_previous_inputs = np.load(os.path.join(DATA_PATH, '%s_%d_test.npy' % (finetune_previous_layer, i)))
	# test_previous_inputs += (tmp['out_chip'] - tmp['intercept']) / tmp['slope']
train_current_inputs = 0
test_current_inputs = 0
for i in range(num_segment_current):
	tmp = np.load(os.path.join(DATA_PATH, '%s_%d_train.npz' % (finetune_current_layer, i)))
	train_current_inputs += (tmp['out_chip'] - tmp['intercept']) / tmp['slope']
	tmp = np.load(os.path.join(DATA_PATH, '%s_%d_test.npz' % (finetune_current_layer, i)))
	test_current_inputs += (tmp['out_chip'] - tmp['intercept']) / tmp['slope']


# Evaluate the model before the fine-tuning (with noise injection)

ITERATION = 10
accuracy_train = np.zeros(ITERATION)
accuracy_test = np.zeros(ITERATION)
model = build_model(activation_bits, weight_noise_train, 0.1, weights)
for i in range(ITERATION):
	scores = model.evaluate([train_previous_inputs, train_current_inputs], y_train, verbose=0)
	print('Train loss:', scores[0])
	print('Train accuracy:', scores[1])
	accuracy_train[i] = scores[1]
	scores = model.evaluate([test_previous_inputs, test_current_inputs], y_test, verbose=0)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	accuracy_test[i] = scores[1]


model = build_model(activation_bits, weight_noise_train, weight_noise_test, weights)
accuracy_train_nf = model.evaluate([train_previous_inputs, train_current_inputs], y_train, verbose=0)[1]
accuracy_test_nf = model.evaluate([test_previous_inputs, test_current_inputs], y_test, verbose=0)[1]

print('Noise-free train accuracy:', accuracy_train_nf)
print('Noise-free test accuracy:', accuracy_test_nf)
print('10%-noise train accuracy:', accuracy_train.mean())
print('10%-noise test accuracy:', accuracy_test.mean())


# Fine-tune

ckpt_dir = os.path.join(os.getcwd(), model_type)
ft_path = os.path.join(ckpt_dir, ft_model_name)
checkpoint = ModelCheckpoint(filepath=ft_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

model.fit([train_previous_inputs, train_current_inputs], y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([test_previous_inputs, test_current_inputs], y_test),
          shuffle=True,
          callbacks=[checkpoint])


# Score fine-tuned model.
model = load_model(ft_path)
weights_ft = {}
for klayer in model.layers:
    weights_ft[klayer.name] = klayer.get_weights()

accuracy_train_nf = model.evaluate([train_previous_inputs, train_current_inputs], y_train, verbose=0)[1]
accuracy_test_nf = model.evaluate([test_previous_inputs, test_current_inputs], y_test, verbose=0)[1]

accuracy_train = np.zeros(ITERATION)
accuracy_test = np.zeros(ITERATION)
model = build_model(activation_bits, weight_noise_train, 0.1, weights_ft)
for i in range(ITERATION):
	scores = model.evaluate([train_previous_inputs, train_current_inputs], y_train, verbose=0)
	print('Train loss:', scores[0])
	print('Train accuracy:', scores[1])
	accuracy_train[i] = scores[1]
	scores = model.evaluate([test_previous_inputs, test_current_inputs], y_test, verbose=0)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	accuracy_test[i] = scores[1]

print('Noise-free train accuracy:', accuracy_train_nf)
print('Noise-free test accuracy:', accuracy_test_nf)
print('10%-noise train accuracy:', accuracy_train.mean())
print('10%-noise test accuracy:', accuracy_test.mean())