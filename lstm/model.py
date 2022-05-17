import glob, os, argparse, re, hashlib, uuid, collections, math, time

from itertools import islice
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from dataloader import SpeechCommandsGoogle

torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda")    
    torch.backends.cudnn.benchmark=True 
else:
    device = torch.device("cpu")

max_w = 1#.1
pact_a = True

def pre_processing(x, y, device, mfcc_cuda):
    batch_size = x.shape[0]

    x =  mfcc_cuda(x.to(device))
    x =  x.permute(2,0,1)
    y =  y.view((-1)).to(device)

    return x,y


class LSTMLayer(nn.Module):
    def __init__(self, cell, drop_p, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)
        self.drop_p = drop_p

        limit1 = 1.0 / math.sqrt(cell_args[1])
        limit2 = 1.0 / math.sqrt(cell_args[0])

        limit1 = w_init(limit1, cell_args[2])
        limit2 = w_init(limit2, cell_args[2])

        torch.nn.init.uniform_(self.cell.weight_hh, a = -limit1, b = limit1)
        torch.nn.init.uniform_(self.cell.weight_ih, a = -limit2, b = limit2)

        #http://proceedings.mlr.press/v37/jozefowicz15.pdf
        torch.nn.init.uniform_(self.cell.bias_ih, a = -0, b = 0)
        torch.nn.init.uniform_(self.cell.bias_hh, a = 1, b = 1)

    def forward(self, input, state):
        if input.shape[0] == self.cell.n_blocks:
            inputs = input.unbind(1)
        else:
            inputs = input.unbind(0)
        # outputs = []

        w_mask = torch.bernoulli(torch.ones_like(self.cell.weight_hh)*(1-self.drop_p))

        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, w_mask)
            # outputs += [out]

        return out, state


def step_d(bits):
    return 2.0 ** (bits - 1)

class QuantFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, x_range):
        if (x is None) or (bits is None) or (bits == 0):
            return x
        
        step_d = 2.0 ** (bits - 1)

        if len(x_range) > 1:
            x_range = x_range.unsqueeze(1).unsqueeze(1).expand(x.shape)

            x_scaled = x/x_range

            x01 = torch.clamp(x_scaled,-1+(1./step_d),1-(1./step_d))

            x01q =  torch.round(x01 * step_d ) / step_d

            x = x01q*x_range

        else:
            x_scaled = x/x_range

            x01 = torch.clamp(x_scaled,-1+(1./step_d),1-(1./step_d))

            x01q =  torch.round(x01 * step_d ) / step_d

            x = x01q*x_range

        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

quant_pass = QuantFunc.apply

def pact_af(x, a):
    if not pact_a:
        return x
    return torch.sign(x) * .5*(torch.abs(x) - torch.abs(torch.abs(x) - a) + a)

def pact_a_bmm(x, a):
    if not pact_a:
        return x
    if len(x.shape) == 2:
        a = a.unsqueeze(1).expand(x.shape)
    elif len(x.shape) == 3:
        a = a.unsqueeze(1).unsqueeze(1).expand(x.shape)
    else:
        raise Exception("Unsupported dimensionality")
    return torch.sign(x) * .5 * (torch.abs(x) - torch.abs(torch.abs(x) - a) + a)

def w_init(fp, wb):
    fp = float(fp)
    if (wb is None) or (wb == 0):
        return fp

    Wm = 1.5/step_d(torch.tensor([float(wb)])).item()
    return Wm if Wm > fp else fp


class CustomMM_bmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, nl, wb, bias_r):
        #noise_w = torch.randn(weight.shape, device = input.device) * weight.max() * nl
        #bias_w  = torch.randn(bias.shape, device = bias.device) * bias.max() * nl

        noise_w = torch.randn(weight.shape, device = input.device) * max_w * nl
        bias_w  = torch.randn(bias.shape, device = bias.device) * max_w * nl
        weight = torch.clamp(weight, -max_w, max_w)
        bias = torch.clamp(bias, -bias_r, bias_r)

        wq = quant_pass(weight, wb, torch.tensor([1.]).to(input.device))
        bq = quant_pass(bias, wb, torch.tensor([1.]).to(input.device))

        ctx.save_for_backward(input, weight, bias)
        output = input.bmm(wq + noise_w) + bq + bias_w
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.bmm(weight.permute(0,2,1))
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.permute(0,2,1).bmm(input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(1)

        return grad_input, grad_weight.permute(0,2,1), grad_bias.unsqueeze(1), None, None, None


class LSTMCellQ_bmm(nn.Module):
    def __init__(self, input_size, hidden_size, wb, ib, abMVM, abNM, noise_level, n_blocks, pact_a, bias_r):
        super(LSTMCellQ_bmm, self).__init__()
        self.wb = wb
        self.ib = ib
        self.abMVM = abMVM
        self.abNM  = abNM
        self.noise_level = noise_level
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.bias_r = bias_r
        self.weight_ih = nn.Parameter(torch.randn(n_blocks, input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(n_blocks, hidden_size, 4 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(n_blocks, 1, 4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(n_blocks, 1, 4 * hidden_size))

        self.a1 = nn.Parameter(torch.tensor([128.] * n_blocks), requires_grad = pact_a)
        self.a2 = nn.Parameter(torch.tensor([16.] * n_blocks), requires_grad = pact_a)
        self.a3 = nn.Parameter(torch.tensor([1.] * n_blocks), requires_grad = pact_a)
        self.a4 = nn.Parameter(torch.tensor([1.] * n_blocks), requires_grad = pact_a)
        self.a5 = nn.Parameter(torch.tensor([1.] * n_blocks), requires_grad = pact_a)
        self.a6 = nn.Parameter(torch.tensor([1.] * n_blocks), requires_grad = pact_a)
        self.a7 = nn.Parameter(torch.tensor([4.] * n_blocks), requires_grad = pact_a)
        self.a8 = nn.Parameter(torch.tensor([1.] * n_blocks), requires_grad = pact_a)
        self.a9 = nn.Parameter(torch.tensor([4.] * n_blocks), requires_grad = pact_a)
        self.a10 = nn.Parameter(torch.tensor([1.] * n_blocks), requires_grad = pact_a)
        self.a11 = nn.Parameter(torch.tensor([4.] * n_blocks), requires_grad = pact_a)

        self.a12 = nn.Parameter(torch.tensor([4.] * n_blocks), requires_grad = pact_a)
        self.a13 = nn.Parameter(torch.tensor([4.] * n_blocks), requires_grad = pact_a)
        self.a14 = nn.Parameter(torch.tensor([4.] * n_blocks), requires_grad = pact_a)



    def forward(self, input, state, w_mask):
        hx, cx = state

        # MVM
        if input.shape[0] == self.n_blocks:
            part1 = CustomMM_bmm.apply(quant_pass(pact_a_bmm(input, self.a1), self.ib, self.a1), self.weight_ih, self.bias_ih, self.noise_level, self.wb, self.bias_r)
        else:
            part1 = CustomMM_bmm.apply(quant_pass(pact_a_bmm(input.repeat(self.n_blocks, 1, 1), self.a1), self.ib, self.a1), self.weight_ih, self.bias_ih, self.noise_level, self.wb, self.bias_r)
        part2 = CustomMM_bmm.apply(quant_pass(pact_a_bmm(hx, self.a11), self.ib, self.a11), self.weight_hh * w_mask, self.bias_hh, self.noise_level, self.wb, self.bias_r)

        # gates = quant_pass(pact_a_bmm( quant_pass(pact_a_bmm(part1, self.a12), self.abMVM, self.a12) + quant_pass(pact_a_bmm(part2, self.a13), self.abMVM, self.a13), self.a14), self.abNM, self.a14)
        gates = pact_a_bmm(pact_a_bmm(part1, self.a12) + pact_a_bmm(part2, self.a13), self.a14)
        # gates = part1 + part2

        #gates = (CustomMM_bmm.apply(quant_pass(pact_a_bmm(input.repeat(self.n_blocks, 1, 1), self.a1), self.ib, self.a1), self.weight_ih, self.bias_ih, self.noise_level, self.wb) + CustomMM_bmm.apply(quant_pass(pact_a_bmm(hx, self.a11), self.ib, self.a11), self.weight_hh * w_mask, self.bias_hh, self.noise_level, self.wb))

        #import pdb; pdb.set_trace()
        #i, j, f, o
        i, j, f, o = gates.chunk(4, 2)
        
        # 
        # forget_gate_out = quant_pass(pact_a_bmm(torch.sigmoid(f), self.a3), self.abNM, self.a3)
        # input_gate_out = quant_pass(pact_a_bmm(torch.sigmoid(i), self.a4), self.abNM, self.a4)
        # activation_out = quant_pass(pact_a_bmm(torch.tanh(j), self.a5), self.abNM, self.a5)
        # output_gate_out = quant_pass(pact_a_bmm(torch.sigmoid(o), self.a6), self.abNM, self.a6)
        forget_gate_out = pact_a_bmm(torch.sigmoid(f), self.a3)
        input_gate_out = pact_a_bmm(torch.sigmoid(i), self.a4)
        activation_out = pact_a_bmm(torch.tanh(j), self.a5)
        output_gate_out = pact_a_bmm(torch.sigmoid(o), self.a6)
        #
        # gated_cell = quant_pass(pact_a_bmm(cx * forget_gate_out, self.a7), self.abNM, self.a7)
        # activated_input = quant_pass(pact_a_bmm(input_gate_out * activation_out, self.a8), self.abNM, self.a8)
        # new_c = quant_pass(pact_a_bmm(gated_cell + activated_input, self.a9), self.abNM, self.a9)
        # activated_cell = quant_pass(pact_a_bmm(torch.tanh(new_c), self.a10), self.abNM, self.a10)
        # new_h = quant_pass(pact_a_bmm(activated_cell * output_gate_out, self.a11), self.abNM, self.a11)
        gated_cell = pact_a_bmm(cx * forget_gate_out, self.a7)
        activated_input = pact_a_bmm(input_gate_out * activation_out, self.a8)
        new_c = pact_a_bmm(gated_cell + activated_input, self.a9)
        activated_cell = pact_a_bmm(torch.tanh(new_c), self.a10)
        new_h = pact_a_bmm(activated_cell * output_gate_out, self.a11)

        return new_h, (new_h, new_c)

class LinLayer_bmm(nn.Module):
    def __init__(self, inp_dim, out_dim, noise_level, abMVM, ib, wb, n_blocks, pact_a, bias_r):
        super(LinLayer_bmm, self).__init__()
        self.abMVM = abMVM
        self.ib = ib
        self.wb = wb
        self.noise_level = noise_level
        self.bias_r = bias_r

        self.weights = nn.Parameter(torch.randn(int(n_blocks), int(inp_dim), int(out_dim)))
        self.bias = nn.Parameter(torch.randn(int(n_blocks), 1, int(out_dim)))


        limit = np.sqrt(6/inp_dim)
        limit = w_init(limit, wb)

        torch.nn.init.uniform_(self.weights, a = -limit, b = limit)
        torch.nn.init.uniform_(self.bias, a = -0, b = 0)

        self.a1 = nn.Parameter(torch.tensor([4.]*n_blocks), requires_grad = pact_a)
        self.a2 = nn.Parameter(torch.tensor([16.]*n_blocks), requires_grad = pact_a)

    def forward(self, input):
        return quant_pass(pact_a_bmm(CustomMM_bmm.apply(quant_pass(pact_a_bmm(input, self.a1), self.ib, self.a1), self.weights, self.bias, self.noise_level, self.wb, self.bias_r), self.a2), self.abMVM, self.a2)


class KWS_LSTM_bmm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, wb, abMVM, abNM, ib, noise_level, drop_p, n_msb, pact_a, bias_r):
        super(KWS_LSTM_bmm, self).__init__()
        self.device = device
        self.noise_level = noise_level
        self.wb = wb
        self.abMVM = abMVM
        self.abNM = abNM
        self.ib = ib
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_p = drop_p
        self.n_msb = n_msb
        self.bias_r = bias_r

        # LSTM layer
        self.lstmBlocks = LSTMLayer(LSTMCellQ_bmm, self.drop_p, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.n_msb, pact_a, bias_r)

        # final FC layer
        self.finFC = LinLayer_bmm(self.hidden_dim, np.ceil(output_dim/n_msb), noise_level, abMVM, ib, wb, self.n_msb, pact_a, bias_r)


    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros(self.n_msb, inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(self.n_msb, inputs.shape[1], self.hidden_dim, device = self.device))
        
        # LSTM blocks
        lstm_out, _ = self.lstmBlocks(inputs, self.hidden_state)

        # final FC blocks
        output = self.finFC(lstm_out[-1,:,:,:])

        #output = torch.stack([output[0,:,0] + output[0,:,1], output[1,:,0] + output[1,:,1], output[2,:,0] + output[2,:,1], output[3,:,0] + output[3,:,1], output[4,:,0], output[4,:,1], output[5,:,0], output[5,:,1], output[6,:,0], output[6,:,1], output[7,:,0], output[7,:,1]],0).t()

        if self.n_msb == 12:
            output = torch.stack([output[0,:,0], output[11,:,0], output[1,:,0], output[10,:,0], output[2,:,0], output[9,:,0], output[3,:,0], output[8,:,0], output[4,:,0], output[7,:,0], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 11:
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[10,:,0], output[2,:,0], output[9,:,0], output[3,:,0], output[8,:,0], output[4,:,0], output[7,:,0], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 10:
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[1,:,1], output[2,:,0], output[9,:,0], output[3,:,0], output[8,:,0], output[4,:,0], output[7,:,0], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 9:
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[1,:,1], output[2,:,0], output[2,:,1], output[3,:,0], output[8,:,0], output[4,:,0], output[7,:,0], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 8:
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[1,:,1], output[2,:,0], output[2,:,1], output[3,:,0], output[3,:,1], output[4,:,0], output[7,:,0], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 7:
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[1,:,1], output[2,:,0], output[2,:,1], output[3,:,0], output[3,:,1], output[4,:,0], output[4,:,1], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 6:
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[1,:,1], output[2,:,0], output[2,:,1], output[3,:,0], output[3,:,1], output[4,:,0], output[4,:,1], output[5,:,0], output[5,:,1]],0).t()
        elif self.n_msb == 5:
            output = torch.stack([output[0,:,0], output[0,:,1], output[0,:,2], output[1,:,0], output[1,:,1], output[1,:,2], output[2,:,0], output[2,:,1], output[3,:,0], output[3,:,1], output[4,:,0], output[4,:,1] ],0).t()
        elif self.n_msb == 4:
            output = torch.stack([output[0,:,0], output[0,:,1], output[0,:,2], output[1,:,0], output[1,:,1], output[1,:,2], output[2,:,0], output[2,:,1], output[2,:,2], output[3,:,0], output[3,:,1], output[3,:,2]],0).t()
        elif self.n_msb == 3:
            output = torch.stack([output[0,:,0], output[0,:,1], output[0,:,2], output[0,:,3], output[1,:,0], output[1,:,1], output[1,:,2], output[1,:,3], output[2,:,0], output[2,:,1], output[2,:,2], output[2,:,3]],0).t()
        elif self.n_msb == 2:
            output = torch.stack([output[0,:,0], output[0,:,1], output[0,:,2], output[0,:,3], output[0,:,4], output[0,:,5], output[1,:,0], output[1,:,1], output[1,:,2], output[1,:,3], output[1,:,4], output[1,:,5]],0).t()
        elif self.n_msb == 1:
            output = output[0,:,:]
        else:
            raise Exception("Too many blocks")

        return output

    def set_noise(self, nl):
        self.noise_level = nl
        self.lstmBlocks.cell.noise_level = nl
        self.finFC.noise_level = nl

    def cosine_sim(self):
        return 0

    def set_drop_p(self, drop_p):
        self.drop_p = drop_p
        self.lstmBlocks.drop_p = drop_p

    def get_a(self):
        return torch.cat([self.lstmBlocks.cell.a1, self.lstmBlocks.cell.a3, self.lstmBlocks.cell.a2,  self.lstmBlocks.cell.a4, self.lstmBlocks.cell.a5, self.lstmBlocks.cell.a6, self.lstmBlocks.cell.a7, self.lstmBlocks.cell.a8, self.lstmBlocks.cell.a9, self.lstmBlocks.cell.a10,  self.lstmBlocks.cell.a11, self.lstmBlocks.cell.a12, self.lstmBlocks.cell.a13, self.lstmBlocks.cell.a14, self.finFC.a1, self.finFC.a2])/(16*self.n_msb)



# orthogonality training
class KWS_LSTM_cs(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, wb, abMVM, abNM, ib, noise_level, drop_p, n_msb, pact_a, bias_r):
        super(KWS_LSTM_cs, self).__init__()
        self.device = device
        self.noise_level = noise_level
        self.wb = wb
        self.abMVM = abMVM
        self.abNM = abNM
        self.ib = ib
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_p = drop_p
        self.n_msb = n_msb

        self.c_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

        # LSTM layer
        self.lstmBlocks = LSTMLayer(LSTMCellQ_bmm, self.drop_p, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.n_msb, pact_a, bias_r)

        # final FC layer
        self.finFC = LinLayer_bmm(self.hidden_dim, 12, noise_level, abMVM, ib, wb, self.n_msb, pact_a, bias_r)

        self.a2 = nn.Parameter(torch.tensor([16.*n_msb]), requires_grad = pact_a)

    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros(self.n_msb, inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(self.n_msb, inputs.shape[1], self.hidden_dim, device = self.device))
        
        # LSTM blocks
        lstm_out, _ = self.lstmBlocks(inputs, self.hidden_state)

        # final FC blocks
        # quantize this
        output = quant_pass(pact_a_bmm(self.finFC(lstm_out).sum(0), self.a2), self.abNM, self.a2)
        
        return output

    def cosine_sim(self):
        sims = torch.zeros([self.n_msb, self.n_msb])

        weights = torch.cat([self.lstmBlocks.cell.weight_ih.reshape(self.n_msb,-1), self.lstmBlocks.cell.weight_hh.reshape(self.n_msb,-1), self.lstmBlocks.cell.bias_ih.reshape(self.n_msb,-1), self.lstmBlocks.cell.bias_hh.reshape(self.n_msb,-1), self.finFC.weights.reshape(self.n_msb,-1), self.finFC.bias.reshape(self.n_msb,-1)], dim=1)

        for i in range(self.n_msb):
            for j in range(self.n_msb):
                sims[i,j] = self.c_sim(weights[i,:], weights[j,:])

        return (sims.sum().sum()-self.n_msb)/(self.n_msb**2)

    def set_noise(self, nl):
        self.noise_level = nl
        self.lstmBlocks.cell.noise_level = nl
        self.finFC.noise_level = nl

    def set_drop_p(self, drop_p):
        self.drop_p = drop_p
        self.lstmBlocks.drop_p = drop_p

    def get_a(self):
        return torch.cat([self.lstmBlocks.cell.a1, self.lstmBlocks.cell.a3, self.lstmBlocks.cell.a2,  self.lstmBlocks.cell.a4, self.lstmBlocks.cell.a5, self.lstmBlocks.cell.a6, self.lstmBlocks.cell.a7, self.lstmBlocks.cell.a8, self.lstmBlocks.cell.a9, self.lstmBlocks.cell.a10,  self.lstmBlocks.cell.a11, self.lstmBlocks.cell.a12, self.lstmBlocks.cell.a13, self.lstmBlocks.cell.a14, self.finFC.a1, self.finFC.a2])/(16*self.n_msb)



# mixed but using double the blocks as given
class KWS_LSTM_mix(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, wb, abMVM, abNM, ib, noise_level, drop_p, n_msb, pact_a, bias_r, gain_blocks):
        super(KWS_LSTM_mix, self).__init__()
        self.device = device
        self.noise_level = noise_level
        self.wb = wb
        self.abMVM = abMVM
        self.abNM = abNM
        self.ib = ib
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_p = drop_p
        self.n_msb = n_msb
        self.gain_blocks = gain_blocks

        self.c_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

        # LSTM layer
        self.lstmBlocks = LSTMLayer(LSTMCellQ_bmm, self.drop_p, self.input_dim, self.hidden_dim, self.wb, self.ib, self.abMVM, self.abNM, self.noise_level, self.n_msb * gain_blocks, pact_a, bias_r)

        # final FC layer
        self.finFC = LinLayer_bmm(self.hidden_dim, np.ceil(output_dim/n_msb), noise_level, abMVM, ib, wb, self.n_msb * gain_blocks, pact_a, bias_r)

        self.a2 = nn.Parameter(torch.tensor([16.*n_msb]), requires_grad = pact_a)


    def forward(self, inputs):
        # init states with zero
        self.hidden_state = (torch.zeros(self.n_msb * self.gain_blocks, inputs.shape[1], self.hidden_dim, device = self.device), torch.zeros(self.n_msb * self.gain_blocks, inputs.shape[1], self.hidden_dim, device = self.device))
        
        # LSTM blocks
        lstm_out, _ = self.lstmBlocks(inputs, self.hidden_state)

        # final FC blocks
        output = self.finFC(lstm_out[-1,:,:,:])

        if self.n_msb == 12:
            import pdb; pdb.set_trace()
            output = torch.stack([output[0,:,0], output[11,:,0], output[1,:,0], output[10,:,0], output[2,:,0], output[9,:,0], output[3,:,0], output[8,:,0], output[4,:,0], output[7,:,0], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 11:
            import pdb; pdb.set_trace()
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[10,:,0], output[2,:,0], output[9,:,0], output[3,:,0], output[8,:,0], output[4,:,0], output[7,:,0], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 10:
            import pdb; pdb.set_trace()
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[1,:,1], output[2,:,0], output[9,:,0], output[3,:,0], output[8,:,0], output[4,:,0], output[7,:,0], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 9:
            import pdb; pdb.set_trace()
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[1,:,1], output[2,:,0], output[2,:,1], output[3,:,0], output[8,:,0], output[4,:,0], output[7,:,0], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 8:
            import pdb; pdb.set_trace()
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[1,:,1], output[2,:,0], output[2,:,1], output[3,:,0], output[3,:,1], output[4,:,0], output[7,:,0], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 7:
            import pdb; pdb.set_trace()
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[1,:,1], output[2,:,0], output[2,:,1], output[3,:,0], output[3,:,1], output[4,:,0], output[4,:,1], output[5,:,0], output[6,:,0]],0).t()
        elif self.n_msb == 6:
            import pdb; pdb.set_trace()
            output = torch.stack([output[0,:,0], output[0,:,1], output[1,:,0], output[1,:,1], output[2,:,0], output[2,:,1], output[3,:,0], output[3,:,1], output[4,:,0], output[4,:,1], output[5,:,0], output[5,:,1]],0).t()
        elif self.n_msb == 5:
            import pdb; pdb.set_trace()
            output = torch.stack([output[0,:,0], output[0,:,1], output[0,:,2], output[1,:,0], output[1,:,1], output[1,:,2], output[2,:,0], output[2,:,1], output[3,:,0], output[3,:,1], output[4,:,0], output[4,:,1] ],0).t()
        elif self.n_msb == 4:
            output = torch.stack([output[0,:,0], output[0,:,1], output[0,:,2], output[1,:,0], output[1,:,1], output[1,:,2], output[2,:,0], output[2,:,1], output[2,:,2], output[3,:,0], output[3,:,1], output[3,:,2]],0).t()
        elif self.n_msb == 3:
            import pdb; pdb.set_trace()
            output = torch.stack([output[0,:,0], output[0,:,1], output[0,:,2], output[0,:,3], output[1,:,0], output[1,:,1], output[1,:,2], output[1,:,3], output[2,:,0], output[2,:,1], output[2,:,2], output[2,:,3]],0).t()
        elif self.n_msb == 2 and self.gain_blocks == 2:
            output = torch.stack([output[0,:,0] + output[2,:,0], output[0,:,1] + output[2,:,1], output[0,:,2] + output[2,:,2], output[0,:,3] + output[2,:,3], output[0,:,4] + output[2,:,4], output[0,:,5] + output[2,:,5], output[1,:,0] + output[3,:,0], output[1,:,1] + output[3,:,1], output[1,:,2] + output[3,:,2], output[1,:,3] + output[3,:,3], output[1,:,4] + output[3,:,4], output[1,:,5] + output[3,:,5]],0).t()
        elif self.n_msb == 1:
            import pdb; pdb.set_trace()
            output = torch.stack([output[0,:,:]],0).t()
        else:
            raise Exception("Configuration does not have specified mapping")
        
        return output

    def cosine_sim(self):
        sims = torch.zeros([self.n_msb, self.n_msb])

        cols = self.n_msb * self.gain_blocks

        weights = torch.cat([self.lstmBlocks.cell.weight_ih.reshape(cols, -1), self.lstmBlocks.cell.weight_hh.reshape(cols,-1), self.lstmBlocks.cell.bias_ih.reshape(cols,-1), self.lstmBlocks.cell.bias_hh.reshape(cols,-1), self.finFC.weights.reshape(cols,-1), self.finFC.bias.reshape(cols,-1)], dim=1)

        if self.n_msb == 2 and self.gain_blocks == 2:
            out = self.c_sim(weights[0,:], weights[2,:])
            out += self.c_sim(weights[1,:], weights[3,:])
            out /= self.gain_blocks
        else:
            raise Exception("Configuration does not have specified mapping")

        return out

    def set_noise(self, nl):
        self.noise_level = nl
        self.lstmBlocks.cell.noise_level = nl
        self.finFC.noise_level = nl

    def set_drop_p(self, drop_p):
        self.drop_p = drop_p
        self.lstmBlocks.drop_p = drop_p

    def get_a(self):
        return torch.cat([self.lstmBlocks.cell.a1, self.lstmBlocks.cell.a3, self.lstmBlocks.cell.a2,  self.lstmBlocks.cell.a4, self.lstmBlocks.cell.a5, self.lstmBlocks.cell.a6, self.lstmBlocks.cell.a7, self.lstmBlocks.cell.a8, self.lstmBlocks.cell.a9, self.lstmBlocks.cell.a10,  self.lstmBlocks.cell.a11, self.lstmBlocks.cell.a12, self.lstmBlocks.cell.a13, self.lstmBlocks.cell.a14, self.finFC.a1, self.finFC.a2])/(16*self.n_msb)
