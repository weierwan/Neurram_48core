# Neural network layers weight mapping and inferencing
# Author: Weier Wan
#

import numpy as np
import matmul


def conv_filter_2_matrix(W, b, relux=1, distribute_bias=True, num_bias=1, seg_length=100):
    if len(W.shape) == 4:
        HH, WW, C, F = W.shape
        W_transform = W.reshape(HH*WW*C, F)
    else:
        HH, F = W.shape
        W_transform = W
    segment = int(np.ceil(W_transform.shape[0] / seg_length))
    seg_length = int(np.ceil(W_transform.shape[0] / segment))
    W_transform_list = []
    for i in range(segment):
        W_segment = W_transform[seg_length * i : seg_length * (i+1), :]
        if distribute_bias:
            b_segment = np.tile(b/relux/segment/num_bias, [num_bias, 1])
            W_segment = np.vstack([W_segment, b_segment])
        W_transform_list.append(W_segment)
    if not distribute_bias:
        b_segment = np.tile(b/relux/num_bias, [num_bias, 1])
        W_transform_list[segment-1] = np.vstack([W_transform_list[segment-1], b_segment])
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


def merge_conv_batchnorm(W, b, gamma, beta, mean, variance, eps=0.001):
    std = np.sqrt(variance + eps)
    w_scaling = gamma / std
    W_folded = W * w_scaling
    b_folded = (b - mean) * w_scaling + beta
    return (W_folded, b_folded)


def conv_unsigned(dev, x, x_addr, y_addr, core_row, core_col, height, width, input_num_bits, segment_index=None, stride=1, pad=(0,0), bias=True, batch_size=1, pulse_multiplier=1, verbose=False):
    N, H, W, C = x.shape
    HC = int(1 + (H + pad[0] + pad[1] - height) / stride)
    WC = int(1 + (W + pad[0] + pad[1] - width) / stride)

    if type(x_addr) is np.ndarray:
        F = len(y_addr)
        out = np.zeros([N,HC,WC,F], dtype=np.float16)
    elif type(x_addr) is list:
        F = len(y_addr[0][0])
        out = np.zeros([len(x_addr),len(x_addr[0]),N,HC,WC,F], dtype=np.float16)
    x_pad = np.pad(x, ((0,0),(pad[0],pad[1]),(pad[0],pad[1]),(0,0)), mode='constant')
    for i in range(N // batch_size):
        xi = x_pad[i*batch_size : (i+1)*batch_size, :, :, :]
        for row in range(HC):
            for column in range(WC):
                tmp = xi[:, stride*row : stride*row+height, stride*column : stride*column+width, :]
                tmp = tmp.reshape([batch_size, -1])
                if segment_index is not None:
                    if type(segment_index) is np.ndarray:
                        tmp = tmp[:, segment_index]
                    elif type(segment_index) is list:
                        tmp_r = []
                        for seg_r in segment_index:
                            tmp_c = []
                            for seg_c in seg_r:
                                tmp_c.append(tmp[:, seg_c])
                            tmp_r.append(tmp_c)
                        tmp = tmp_r
                out_tmp = matmul.matmul_unsigned(
                	dev, tmp, x_addr, y_addr, bias, core_row, True, input_num_bits, core_col,
                	pulse_multiplier=pulse_multiplier, prep=(i==0 and row==0 and column==0))
                if type(x_addr) is np.ndarray:
                    out[i*batch_size : (i+1)*batch_size, row, column, :] = out_tmp
                elif type(x_addr) is list:
                    out[:, :, i*batch_size : (i+1)*batch_size, row, column, :] = np.array(out_tmp)
        if verbose:
            print('Finished batch %d' % i)
    matmul.disable_inference(dev)
    return out


def dense_unsigned(dev, x, x_addr, y_addr, core_row, core_col, input_num_bits, segment_index=None, bias=False, batch_size=1, pulse_multiplier=1, verbose=False):
    N, H = x.shape
    F = len(y_addr)
    
    out = np.zeros([N, F], dtype=np.float16)
    for i in range(N // batch_size):
        xi = x[i*batch_size : (i+1)*batch_size, :]
        if segment_index is not None:
            xi = xi[:, segment_index]
        out[i*batch_size : (i+1)*batch_size, :] = matmul.matmul_unsigned(
            dev, xi, x_addr, y_addr, bias, core_row, True, input_num_bits, core_col,
            pulse_multiplier=pulse_multiplier, prep=(i==0))
        if verbose:
            print('Finished batch %d' % i)
    matmul.disable_inference(dev)
    return out


def quantize_unsigned(x, num_bits, max_value):
    y = np.maximum(np.minimum(x, max_value), 0.0)
    y = np.round(y / max_value * (2**num_bits-1))
    return y


def quantize_unsigned_rescale(x, num_bits, max_value):
    return x / (2**num_bits-1) * max_value


def quantize_signed(x, num_bits, max_value):
    step = 2.0**(num_bits-1)
    x = x / max_value
    y = np.clip(x, -1+(1/step), 1-(1/step))
    y = np.round(y*step)
    return y


def quantize_signed_rescale(x, num_bits, max_value):
    return x / 2**(num_bits-1) * max_value
