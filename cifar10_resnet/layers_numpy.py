import numpy as np


def quantize_unsigned(x, num_bits, max_value):
    x = np.maximum(np.minimum(x, max_value), 0.0)
    x = np.round(x / max_value * (2**num_bits-1))
    return x


def quantize_rescale(x, num_bits, max_value):
    return x / (2**num_bits-1) * max_value



def conv(x, w, b, stride=1, pad=(0,0), segment_index=None, bias=True):
  N, H, W, C = x.shape
  HH, WW, _, F = w.shape
  HC = int(1 + (H + pad[0] + pad[1] - HH) / stride)
  WC = int(1 + (W + pad[0] + pad[1] - WW) / stride)
  
  out = np.zeros([N,HC,WC,F], dtype=np.float32)
  x_pad = np.pad(x, ((0,0),(pad[0],pad[1]),(pad[0],pad[1]),(0,0)), mode='constant')

  for row in range(HC):
      for column in range(WC):
          tmp = x_pad[:, stride*row : stride*row+HH, stride*column : stride*column+WW, :].reshape([N, -1])
          w_tmp = w.reshape([-1, F])
          if segment_index is not None:
            tmp = tmp[:, segment_index]
            w_tmp = w_tmp[segment_index, :]
          out[:, row, column, :] = tmp.dot(w_tmp)
  if bias:
    out += b
  return out


def max_pool(x, pool_size, stride=None, pad=(0,0)):
  HH = pool_size
  WW = pool_size
  if stride is None:
    stride = pool_size

  x = np.pad(x, ((0,0),(pad[0],pad[1]),(pad[0],pad[1]),(0,0)), mode='constant')
  N, H, W, C = x.shape
  HP = int((H - HH) / stride + 1)
  WP = int((W - WW) / stride + 1)

  out = np.zeros([N,HP,WP,C])
  for i, xi in enumerate(x):
    for c in range(C):
        ci = xi[:,:,c]
        for h in range(HP):
            for w in range(WP):
                out[i,h,w,c] = np.max(ci[h*stride:h*stride+HH, w*stride:w*stride+WW])
  return out


def avg_pool_flatten(x):
  return np.mean(x, axis=(1,2))


def relu(x):
  return x * (x>0)


def flatten(x):
  N, H, W, C = x.shape
  return x.reshape(N, -1)


def dense(x, w, b, segment_index=None):
  if segment_index is None:
    out = x.dot(w) + b
  else:
    out = x[:, segment_index].dot(w[segment_index, :]) + b
  return out


def batch_normalization(x, gamma, beta, mean, variance, eps=0.001):
	out = (x - mean) / np.sqrt(variance + eps)
	return out * gamma + beta


def merge_conv_batchnorm(W, b, gamma, beta, mean, variance, eps=0.001):
	std = np.sqrt(variance + eps)
	w_scaling = gamma / std
	W_folded = W * w_scaling
	b_folded = (b - mean) * w_scaling + beta
	return (W_folded, b_folded)
	