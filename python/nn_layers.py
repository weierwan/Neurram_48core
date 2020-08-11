# Neural network layers weight mapping and inferencing
# Author: Weier Wan
#

import numpy as np
import matmul


def conv_filter_2_matrix(W, b, relux=1):
    HH, WW, C, F = W.shape
    W_transform = W.reshape(HH*WW*C, F)
    segment = int(np.ceil(W_transform.shape[0] / 100.0))
    seg_length = int(np.ceil(W_transform.shape[0] / segment))
    W_transform_list = []
    for i in range(segment):
        W_transform_list.append(W_transform[seg_length * i : seg_length * (i+1), :])
    W_transform_list[segment-1] = np.vstack([W_transform_list[segment-1], b/relux])
    return W_transform_list


def weight_mapping(W, g_max, scheme='sign', w_max_percentile=100):
	w_max = np.percentile(np.abs(W), w_max_percentile)
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


def conv_unsigned(dev, x, x_addr, y_addr, core_row, core_col, height, width, input_num_bits, segment_index=None, stride=1, pad=(0,0), bias=True, batch_size=1, pulse_multiplier=1, verbose=False):
    N, H, W, C = x.shape
    F = len(y_addr)
    HC = int(1 + (H + pad[0] + pad[1] - height) / stride)
    WC = int(1 + (W + pad[0] + pad[1] - width) / stride)

    out = np.zeros([N,HC,WC,F])
    x_pad = np.pad(x, ((0,0),(pad[0],pad[1]),(pad[0],pad[1]),(0,0)), mode='constant')
    for i in range(N // batch_size):
        xi = x_pad[i*batch_size : (i+1)*batch_size, :, :, :]
        for row in range(HC):
            for column in range(WC):
                tmp = xi[:, stride*row : stride*row+height, stride*column : stride*column+width, :]
                tmp = tmp.reshape([batch_size, -1])
                if segment_index is not None:
                    tmp = tmp[:, segment_index]
                out[i*batch_size : (i+1)*batch_size, row, column, :] = matmul.matmul_unsigned(
                	dev, tmp, x_addr, y_addr, bias, core_row, True, input_num_bits, core_col,
                	pulse_multiplier=pulse_multiplier, prep=(i==0 and row==0 and column==0))
        if verbose:
            print('Finished batch %d' % i)
    matmul.disable_inference(dev)
    return out


def dense_unsigned(dev, x, x_addr, y_addr, core_row, core_col, input_num_bits, segment_index=None, bias=False, batch_size=1, pulse_multiplier=1, verbose=False):
    N, H = x.shape
    F = len(y_addr)
    
    out = np.zeros([N, F])
    for i in range(N // batch_size):
        xi = x_pad[i*batch_size : (i+1)*batch_size, :]
        if segment_index is not None:
            xi = xi[:, segment_index]
        out[i*batch_size : (i+1)*batch_size, :] = matmul.matmul_unsigned(
            dev, xi, x_addr, y_addr, bias, core_row, True, input_num_bits, core_col,
            pulse_multiplier=pulse_multiplier, prep=(i==0))
        if verbose:
            print('Finished batch %d' % i)
    matmul.disable_inference(dev)
    return out