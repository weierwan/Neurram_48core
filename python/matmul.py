# Matmul
# Author: Weier Wan
#

import numpy as np
import ok
import dac_control as dac
import spi_control as spi

VREF = 0.9
BIAS10 = VREF
IGATE_BIAS = 0.7
VCOMP2 = VREF + 0.02

def dac_setup(dev, vpos_bl=0.0, vneg_bl=0.0, vpos_sl=0.0, vneg_sl=0.0, vreset_plus=0.1, vreset_minus=0.06, vcomp_offset=0.02):
    dac.dac_program_single_daisy(dev, 0, 0, VREF + vcomp_offset)
    dac.dac_program_single_daisy(dev, 0, 1, VREF + vcomp_offset)
    dac.dac_program_single_daisy(dev, 0, 2, VREF)
    dac.dac_program_single_daisy(dev, 1, 1, BIAS10)
    dac.dac_program_single_daisy(dev, 1, 4, BIAS10)
    dac.dac_program_single_daisy(dev, 2, 0, IGATE_BIAS)
    dac.dac_program_single_daisy(dev, 2, 2, VREF - vreset_minus)
    dac.dac_program_single_daisy(dev, 0, 4, VREF + vreset_plus)
    
    dac.dac_program_single_daisy(dev, 1, 0, VREF - vneg_bl)
    dac.dac_program_single_daisy(dev, 1, 2, VREF + vpos_bl)
    dac.dac_program_single_daisy(dev, 1, 3, VREF - vneg_sl)
    dac.dac_program_single_daisy(dev, 1, 5, VREF + vpos_sl)


def _populate_nparray(x, x_addr, bias, fwd):
    if bias:
        assert (2*(len(x)+1) == len(x_addr))
    else:
        assert (2*(len(x)) == len(x_addr))
        
    x_reg = np.zeros(256)
    x_reg[x_addr[0:2*len(x):2]] = x
    x_reg[x_addr[1:2*len(x):2]] = -x
    if bias:
        x_reg[x_addr[2*len(x)]] = 1
        x_reg[x_addr[2*len(x)+1]] = -1
    if not fwd:
        if bias:
            x_reg[x_addr[-1]+1:] = np.tile([1,-1], 128-len(x)-1)
        else:
            x_reg[x_addr[-1]+1:] = np.tile([1,-1], 128-len(x))
    return x_reg


def _encode_input(x, x_addr, bias, fwd):
    if type(x) is np.ndarray and len(x.shape) == 1:
        return _populate_nparray(x, x_addr, bias, fwd)
    assert len(x) == len(x_addr)
    assert len(x) == len(bias)
    x_reg = []
    for x_i, addr_i, bias_i in zip(x, x_addr, bias):
        x_reg.append(_populate_nparray(x_i, addr_i, bias_i, fwd))
    return np.hstack(x_reg)


def _encode_input_01(x, x_addr, bias, fwd):
    if type(x) is np.ndarray and len(x.shape) == 1:
        x_bipolar = x*2-1
        ones_bipolar = np.ones_like(x)
    else:
        x_bipolar = [i*2-1 for i in x]
        ones_bipolar = [np.ones_like(i) for i in x]
    return np.hstack([_encode_input(x_bipolar, x_addr, bias, fwd), _encode_input(ones_bipolar, x_addr, bias, fwd)])


def _encode_input_unsigned(x, x_addr, bias, fwd, input_num_bits):
    if type(x) is np.ndarray and len(x.shape) == 1:
        assert (x < 2**input_num_bits).all(), 'x value must be less than 2**input_num_bits'
    x_binary_list = []
    for b in range(input_num_bits):
        if type(x) is np.ndarray and len(x.shape) == 1:
            x_binary = (x.astype(np.int8) & (0b1 << b)) >> b
        else:
            x_binary = [(x_i.astype(np.int8) & (0b1 << b)) >> b for x_i in x]
        x_binary_list.append(_encode_input_01(x_binary, x_addr, bias, fwd))
    return np.hstack(x_binary_list)


def _encode_multi_row(x, x_addr, bias, fwd, encode_func=_encode_input, **kwargs):
    assert len(x) == len(x_addr)
    x_rows = []
    for x_i, addr_i, bias_i in zip(x, x_addr, bias):
        x_reg = encode_func(x_i, addr_i, bias_i, fwd, **kwargs)
        x_rows.append(x_reg)
    return np.vstack(x_rows)
    
    
def _translate_outputs(outputs, y_addr, output_dim=256):
    if len(outputs.shape) == 1:
        return outputs[y_addr]
    assert outputs.shape[0] == len(y_addr)
    assert outputs.shape[1] == len(y_addr[0]) * output_dim
    out_list = []
    for r, y_addr_r in enumerate(y_addr):
        out_row = []
        for c, y_addr_rc in enumerate(y_addr_r):
            out_row.append(outputs[r, output_dim*c : output_dim*(c+1)][y_addr_rc])
        out_list.append(out_row)
    return out_list


def _setup_inference(dev, fwd, num_pulses, run_all):
    dev.SetWireInValue(0x06, 0b01100000) # Enable neuron OTA and reg_controlled_wl
    if fwd:
        dev.SetWireInValue(0x05, 0b100001) # Set inference direction
    else:
        dev.SetWireInValue(0x05, 0b000001)
    if run_all:
        dev.SetWireInValue(0x09, 0b01011111111101 | (num_pulses << 14)) # RUN_ALL mode + set OTA timing       
    else:
        dev.SetWireInValue(0x09, 0b01011111111100 | (num_pulses << 14))
    dev.UpdateWireIns()
    

def _setup_neuron_control(dev, num_pulses, run_all):
    if run_all:
        dev.SetWireInValue(0x09, 0b01011111111101 | (num_pulses << 14)) # RUN_ALL mode + set OTA timing       
    else:
        dev.SetWireInValue(0x09, 0b01011111111100 | (num_pulses << 14))
    dev.UpdateWireIns()

    
def _setup_partial_reset_hw(dev, num_pulses=0):
    dev.SetWireInValue(0x09, 0b01011111111111 | (num_pulses << 14))
    dev.SetWireInValue(0x03, 0b0011000000)
    dev.SetWireInValue(0x0D, 0b00 | (11 & 0xf) << 2)
    dev.UpdateWireIns()

    
def _setup_neuron_register_readout(dev, fwd):
    if fwd:
        dev.SetWireInValue(0x03, 0b0001000000) # Register write enable
    else:
        dev.SetWireInValue(0x03, 0b0010000000)
    dev.UpdateWireIns()


def _trigger_neuron(dev, trigger):
    dev.ActivateTriggerIn(0x45, trigger)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if status & 0b1 != 0:
            break


def _disable_inference(dev, disable_neuron=False):
    dev.SetWireInValue(0x05, 0b000000) # Disable inference mode
    if disable_neuron:
        dev.SetWireInValue(0x06, 0b00100000) # Disable neurons
    dev.UpdateWireIns()

        
def send_inputs(dev, x, x_addr, bias, row_addr, fwd, encode_func, col_addr=None, trigger=True, **kwargs):
    spi.reset(dev, vert=(not fwd), horz=fwd)
    if type(x) is np.ndarray and len(x.shape) == 1:
        inputs = encode_func(x, x_addr, bias, fwd, **kwargs)
        spi.write_single_core(dev, row_addr, col_addr, vert=fwd, is_pipe_in=True, inputs=inputs, trigger=trigger)
    else:
        inputs = _encode_multi_row(x, x_addr, bias, fwd, encode_func=encode_func, **kwargs)
        spi.write_rows(dev, row_addr, vert=fwd, is_pipe_in=True, inputs=inputs, pipe_in_steps=len(bias[0]), trigger=trigger)


def _matmul_unsigned_helper_hw(dev, fwd, num_bits=1, pulse_multiplier=1, readout=True, cds=True):
    dev.SetWireInValue(0x0E, (pulse_multiplier & 0b11111) << 5 | ((num_bits-1) & 0b111) << 2 | 0b10 | cds)
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x45, 7)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if status & 0b10000 != 0:
            break
    if readout:
        _setup_neuron_register_readout(dev, fwd)
        _trigger_neuron(dev, 2)
               
    
def matmul_bipolar(dev, x, x_addr, y_addr, bias, row_addr, fwd, num_pulses, col_addr=None):
    send_inputs(dev, x, x_addr, bias, row_addr, fwd, _encode_input, col_addr=col_addr)
    _setup_inference(dev, fwd, num_pulses, True)
    _setup_neuron_register_readout(dev, fwd)

    _trigger_neuron(dev, 0)
    _disable_inference(dev)
    
    # Read register
    if type(x) is np.ndarray and len(x.shape) == 1:
        out = spi.read_single_core(dev, row_addr, col_addr, False, is_pipe_out=True)
    else:
        out = spi.read_rows(dev, row_addr, vert=False, is_pipe_out=True)[:, :len(bias[0])*256]
    return _translate_outputs(out*2+1, y_addr)


def matmul_01(dev, x, x_addr, y_addr, bias, row_addr, fwd, num_pulses, col_addr=None):
    send_inputs(dev, x, x_addr, bias, row_addr, fwd, _encode_input_01, col_addr=col_addr, trigger=False)
    _setup_inference(dev, fwd, num_pulses, False)
    _matmul_unsigned_helper_hw(dev, fwd, pulse_multiplier=num_pulses, readout=True)
    _disable_inference(dev)
    
    # Read register
    if type(x) is np.ndarray and len(x.shape) == 1:
        out = spi.read_single_core(dev, row_addr, col_addr, False, is_pipe_out=True)
    else:
        out = spi.read_rows(dev, row_addr, vert=False, is_pipe_out=True)[:, :len(bias[0])*256]
    return _translate_outputs(out+1, y_addr)


def _write_y_addr(dev, y_addr, row_addr=None, col_addr=None):
    if type(y_addr) is np.ndarray and len(y_addr.shape) == 1:
        btarray = bytearray(96)
        for y in y_addr:
            btarray[col_addr*16 + y//8] = btarray[col_addr*16 + y//8] | (0b1 << (y%8))
    else:
        assert len(y_addr) == len(row_addr)
        btarray = bytearray(96 * len(y_addr))
        for r, addr_r in enumerate(y_addr):
            for c, addr_c in enumerate(addr_r):
                for y in addr_c:
                    btarray[r*96 + c*16 + y//8] = btarray[r*96 + c*16 + y//8] | (0b1 << (y%8))
    data = dev.WriteToPipeIn(0x81, btarray)
    dev.ActivateTriggerIn(0x45, 6)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if (status & 0b1000 != 0) and (status & (0b1 << 7) != 0):
            break
        

def _decode_row_output(datain):
    assert len(datain) == 768
    num_step = np.zeros(768)
    out_init = np.zeros(768)
    for i, bt in enumerate(datain):
        num_step[i] = bt & 0x7f
        out_init[i] = bt >> 7
    return (num_step, out_init)


def _read_num_step(dev, y_addr, row_addr=None, col_addr=None):
    if type(row_addr) is not list and col_addr is not None:
        while True:
            dev.UpdateWireOuts()
            status = dev.GetWireOutValue(0x28)
            if status & (0b1 << 13) == 0:
                break
        datain = bytearray(768)
        data = dev.ReadFromPipeOut(0xA2, datain)
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        assert status & (0b1 << 8) != 0
        assert status & (0b1 << 13) != 0
        num_step, out_init = _decode_row_output(datain)
        num_step = num_step[128*col_addr : 128*(col_addr+1)]
        out_init = out_init[128*col_addr : 128*(col_addr+1)]
    else:
        num_row = len(y_addr)
        num_col = len(y_addr[0])
        while True:
            dev.UpdateWireOuts()
            status = dev.GetWireOutValue(0x28)
            if status & (0b1 << 13) == 0:
                break
        datain = bytearray(768 * num_row)
        data = dev.ReadFromPipeOut(0xA2, datain)
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        assert status & (0b1 << 8) != 0
        assert status & (0b1 << 13) != 0
        num_step = np.zeros([num_row, 128*num_col])
        out_init = np.zeros([num_row, 128*num_col])
        for r in range(num_row):
            num_step_r, out_init_r = _decode_row_output(datain[768*r : 768*(r+1)])
            num_step[r, :] = num_step_r[:128*num_col]
            out_init[r, :] = out_init_r[:128*num_col]
    outputs = (num_step - 0.5) * (1 - out_init * 2)
    return _translate_outputs(outputs, y_addr, output_dim=128)
    

def _matmul_partial_reset_helper_hw(dev, y_addr, row_addr, col_addr=None, write_y_addr=True):
    _setup_partial_reset_hw(dev)
    if write_y_addr:
        _write_y_addr(dev, y_addr, row_addr=row_addr, col_addr=col_addr)
    dev.ActivateTriggerIn(0x45, 5)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if status & 0b1000 != 0:
            break
    return _read_num_step(dev, y_addr, row_addr=row_addr, col_addr=col_addr)
    

def matmul_bipolar_partial_reset(dev, x, x_addr, y_addr, bias, row_addr, fwd, num_pulses, col_addr=None, write_y_addr=True):
    send_inputs(dev, x, x_addr, bias, row_addr, fwd, _encode_input, col_addr=col_addr)
    _setup_inference(dev, fwd, num_pulses, False)
    
    _trigger_neuron(dev, 0)
    _trigger_neuron(dev, 1)
    output = _matmul_partial_reset_helper_hw(dev, y_addr, row_addr=row_addr, col_addr=col_addr, write_y_addr=write_y_addr)
        
    _disable_inference(dev)
    return output


def matmul_01_partial_reset(dev, x, x_addr, y_addr, bias, row_addr, fwd, num_pulses, col_addr=None, write_y_addr=True):
    send_inputs(dev, x, x_addr, bias, row_addr, fwd, _encode_input_01, col_addr=col_addr, trigger=False)
    _setup_inference(dev, fwd, num_pulses, False)
    _matmul_unsigned_helper_hw(dev, fwd, pulse_multiplier=num_pulses, readout=False)
    output = _matmul_partial_reset_helper_hw(dev, y_addr, row_addr=row_addr, col_addr=col_addr, write_y_addr=write_y_addr)
    _disable_inference(dev)
    return output


def matmul_unsigned_num_pulse(dev, x, x_addr, y_addr, bias, row_addr, fwd, input_num_bits, col_addr=None, pulse_multiplier=1, write_y_addr=True):
    send_inputs(dev, x, x_addr, bias, row_addr, fwd, _encode_input_unsigned, col_addr=col_addr, trigger=False, input_num_bits=input_num_bits)
    _setup_inference(dev, fwd, 0, False)
    _matmul_unsigned_helper_hw(dev, fwd, num_bits=input_num_bits, pulse_multiplier=pulse_multiplier, readout=False)
    output = _matmul_partial_reset_helper_hw(dev, y_addr, row_addr=row_addr, col_addr=col_addr, write_y_addr=write_y_addr)
    _disable_inference(dev)
    return output


matmul_unsigned = matmul_unsigned_num_pulse


def calibrate_voltage(dev, vpos, vneg, rows, cols, fwd, num_pulse, tolerance=0.05, increment=0.001, iteration=1000, max_cycle=10, vcomp_offset=0.02, verbose=False):
    M = int(len(rows)/2)
    N = int(len(cols))
    cycle = 0
    while True:
        dac_setup(dev, vpos, vneg, vcomp_offset=vcomp_offset)
        outputs = np.zeros([iteration, N])
        inputs = np.random.binomial(n=1, p=0.5, size=(iteration, M)) * 2 - 1
        for i in xrange(iteration):
            outputs[i, :] = matmul_bipolar(dev, inputs[i, :], rows, cols, fwd, num_pulse, False)
        one_pct = np.sum(outputs == 1) / float(outputs.size)
        if verbose:
            print('vpos=%.4f, vneg=%.4f, percentage(output=1)=%.3f' % (vpos, vneg, one_pct))
        if cycle >= max_cycle:
            print('fail to calibrate, vpos=%.4f, vneg=%.4f, percentage(output=1) = %.3f' % (vpos, vneg, one_pct))
            return (vpos, vneg)
        if one_pct < 0.5 - tolerance:
            vpos += increment
            vneg -= increment
        elif one_pct > 0.5 + tolerance:
            vpos -= increment
            vneg += increment
        else:
            print('calibrate vpos=%.4f, vneg=%.4f, percentage(output=1)=%.3f' % (vpos, vneg, one_pct))
            return (vpos, vneg)
        cycle += 1
