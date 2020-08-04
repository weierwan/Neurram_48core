# Matmul
# Author: Weier Wan
#

import numpy as np
import matmul


def conv_filter_2_matrix(W, b):
    HH, WW, C, F = W.shape
    W_transform = W.reshape(HH*WW*C, F)
    W_transform = np.vstack([W_transform, b])
    return W_transform


def weight_mapping(W, g_max, scheme='sign'):
	w_max = np.abs(W).max()
	G = np.zeros([W.shape[0]*2, W.shape[1]])
	if scheme == 'sign':
		scale = g_max / w_max
		G[0::2, :] = (W>0) * W * scale
		G[1::2, :] = -((W<0) * W) * scale
	else:
		scale = g_max / w_max / 2
		G[0::2, :] = (w_max + W) * scale
		G[1::2, :] = (w_max - W) * scale
	return G


def conv_unsigned(dev, x, x_addr, y_addr, core_row, core_col, height, width, input_num_bits, stride=1, pad=(0,0), bias=True, pulse_multiplier=1, write_y_addr=True, verbose=False):
    N, H, W, C = x.shape
    F = len(y_addr)
    HC = int(1 + (H + pad[0] + pad[1] - height) / stride)
    WC = int(1 + (W + pad[0] + pad[1] - width) / stride)

    out = np.zeros([N,HC,WC,F])
    x_pad = np.pad(x, ((0,0),(pad[0],pad[1]),(pad[0],pad[1]),(0,0)), mode='constant')
    for i, xi in enumerate(x_pad):
        for row in range(HC):
            for column in range(WC):
                tmp = xi[stride*row : stride*row+height, stride*column : stride*column+width, :]
                out[i, row, column, :] = matmul.matmul_unsigned(dev, tmp.flatten(), x_addr, y_addr, bias, core_row, True, input_num_bits, core_col, pulse_multiplier=pulse_multiplier, write_y_addr=write_y_addr)
        if verbose:
            print('Finished input %d' % i)
    return out


def dense_unsigned(dev, x, x_addr, y_addr, core_row, core_col, input_num_bits, bias=False, pulse_multiplier=1, verbose=False):
    N, H = x.shape
    F = len(y_addr)
    
    out = np.zeros([N, F])
    for i, xi in enumerate(x):
        out[i, :] = matmul.matmul_unsigned(dev, xi, x_addr, y_addr, bias, core_row, True, input_num_bits, core_col, pulse_multiplier=pulse_multiplier)
        if verbose:
            print('Finished input %d' % i)
    return out