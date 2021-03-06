# Matmul
# Author: Weier Wan
#

import numpy as np
import ok
import time
import dac_control as dac
import spi_control as spi

VREF = 0.9
BIAS10 = VREF
IGATE_BIAS = 0.7
VCOMP2 = VREF + 0.02
NMLO_LENGTH = 128
NMLO_CORE = 3
NMLO_TOTAL_LENGTH = NMLO_LENGTH * NMLO_CORE

def dac_setup(dev, vpos_bl=None, vneg_bl=None, vpos_sl=None, vneg_sl=None, vreset_plus=0.1, vreset_minus=0.06, vcomp_offset=0.02):
    dac.dac_program_single_daisy(dev, 0, 0, VREF + vcomp_offset)
    dac.dac_program_single_daisy(dev, 0, 1, VREF + vcomp_offset)
    dac.dac_program_single_daisy(dev, 0, 2, VREF)
    dac.dac_program_single_daisy(dev, 1, 1, BIAS10)
    dac.dac_program_single_daisy(dev, 1, 4, BIAS10)
    dac.dac_program_single_daisy(dev, 2, 0, IGATE_BIAS)
    dac.dac_program_single_daisy(dev, 2, 2, VREF - vreset_minus)
    dac.dac_program_single_daisy(dev, 0, 4, VREF + vreset_plus)
    
    if vneg_bl is not None:
        dac.dac_program_single_daisy(dev, 1, 0, VREF - vneg_bl)
    if vpos_bl is not None:
        dac.dac_program_single_daisy(dev, 1, 2, VREF + vpos_bl)
    if vneg_sl is not None:
        dac.dac_program_single_daisy(dev, 1, 3, VREF - vneg_sl)
        # dac.ramp_up_voltage(dev, 1, 3, VREF - vneg_sl)
    if vpos_sl is not None:
        dac.dac_program_single_daisy(dev, 1, 5, VREF + vpos_sl)
        # dac.ramp_up_voltage(dev, 1, 5, VREF + vpos_sl)


def _populate_nparray(x, x_addr, bias, fwd, reverse=False):
    transform = len(x.shape) == 1
    if transform:
        x = x.reshape([1, -1])
    N, L = x.shape
    if bias:
        if type(bias) is bool:
            bias = 1
        assert (2*(L + bias) == len(x_addr))
    else:
        assert (2*L == len(x_addr))
        
    x_reg = np.zeros([N, 256])
    x_reg[:, x_addr[0:2*L:2]] = x
    x_reg[:, x_addr[1:2*L:2]] = -x
    if bias:
        x_reg[:, x_addr[2*L : 2*(L+bias)]] = np.tile([1, -1], bias)
    if not fwd:
        if reverse:
            padding = [-1, 1]
        else:
            padding = [1, -1]
        padding_idx = np.delete(np.arange(256), x_addr)
        if bias:
            x_reg[:, padding_idx] = np.tile(padding, 128-L-bias)
        else:
            x_reg[:, padding_idx] = np.tile(padding, 128-L)
    if transform:
        x_reg = x_reg.reshape([-1,])
    return x_reg


def _encode_input(x, x_addr, bias, fwd, reverse=False):
    if type(x) is np.ndarray:
        return _populate_nparray(x, x_addr, bias, fwd, reverse=reverse)
    assert len(x) == len(x_addr)
    assert len(x) == len(bias)
    x_reg = []
    for x_i, addr_i, bias_i in zip(x, x_addr, bias):
        x_reg.append(_populate_nparray(x_i, addr_i, bias_i, fwd, reverse=reverse))
    return np.hstack(x_reg)


def _encode_input_01(x, x_addr, bias, fwd, neg=False):
    if type(x) is np.ndarray:
        x_bipolar = x*2-1
        ones_bipolar = np.ones_like(x)
        if neg:
            x_bipolar = -x_bipolar
            ones_bipolar = -ones_bipolar
    else:
        x_bipolar = [i*2-1 for i in x]
        ones_bipolar = [np.ones_like(i) for i in x]
        if neg:
            x_bipolar = [-i for i in x_bipolar]
            ones_bipolar = [-i for i in ones_bipolar]
    return np.hstack([_encode_input(x_bipolar, x_addr, bias, fwd), _encode_input(ones_bipolar, x_addr, bias, fwd, reverse=True)])


def _encode_input_101(x, x_addr, bias, fwd):
    if type(x) is np.ndarray:
        x_1 = (x>0) * 2 - 1
        x_2 = (x>=0) * 2 - 1
    else:
        x_1 = [(i>0)*2-1 for i in x]
        x_2 = [(i>=0)*2-1 for i in x]
    return np.hstack([_encode_input(x_1, x_addr, bias, fwd), _encode_input(x_2, x_addr, bias, fwd)])


def _encode_input_unsigned(x, x_addr, bias, fwd, input_num_bits):
    if type(x) is np.ndarray:
        assert (x < 2**input_num_bits).all(), 'x value must be less than 2**input_num_bits'
    x_binary_list = []
    for b in range(input_num_bits):
        if type(x) is np.ndarray:
            x_binary = (x.astype(np.int8) & (0b1 << b)) >> b
        else:
            x_binary = [(x_i.astype(np.int8) & (0b1 << b)) >> b for x_i in x]
        x_binary_list.append(_encode_input_01(x_binary, x_addr, bias, fwd))
    return np.hstack(x_binary_list)


def _encode_input_2complement(x, x_addr, bias, fwd, input_num_bits):
    if type(x) is np.ndarray:
        assert (x < 2**(input_num_bits-1)).all(), 'x value must be less than 2**(input_num_bits-1)'
        assert (x >= -2**(input_num_bits-1)).all(), 'x value must be greater than or equal to -2**(input_num_bits-1)'
    x_binary_list = []
    for b in range(input_num_bits-1):
        if type(x) is np.ndarray:
            x_binary = (x.astype(np.int8) & (0b1 << b)) >> b
        else:
            x_binary = [(x_i.astype(np.int8) & (0b1 << b)) >> b for x_i in x]
        x_binary_list.append(_encode_input_01(x_binary, x_addr, bias, fwd))
    # Process the sign bit (2's complement)
    if type(x) is np.ndarray:
        x_binary = (x < 0).astype(np.int8)
    else:
        x_binary = [(x < 0).astype(np.int8) for x_i in x]
    x_binary_list.append(_encode_input_01(x_binary, x_addr, bias, fwd, neg=True))
    return np.hstack(x_binary_list)


def _encode_input_signed(x, x_addr, bias, fwd, input_num_bits):
    if type(x) is np.ndarray:
        assert (x < 2**(input_num_bits-1)).all(), 'x value must be less than 2**(input_num_bits-1)'
        assert (x > -2**(input_num_bits-1)).all(), 'x value must be greater than -2**(input_num_bits-1)'
    x_binary_list = []
    for b in range(input_num_bits-1):
        if type(x) is np.ndarray:
            x_binary = (np.abs(x).astype(np.int8) >> b & 0b1) * np.sign(x)
        else:
            x_binary = [(np.abs(x_i).astype(np.int8) >> b & 0b1) * np.sign(x_i) for x_i in x]
        x_binary_list.append(_encode_input_101(x_binary, x_addr, bias, fwd))
    return np.hstack(x_binary_list)



def _encode_multi_row(x, x_addr, bias, fwd, encode_func=_encode_input, **kwargs):
    assert len(x) == len(x_addr)
    x_rows = []
    for x_i, addr_i, bias_i in zip(x, x_addr, bias):
        x_reg = encode_func(x_i, addr_i, bias_i, fwd, **kwargs).flatten()
        x_rows.append(x_reg)
    return np.vstack(x_rows)
    
    
def _translate_outputs(outputs, y_addr, batch_size=None):
    if type(y_addr) is np.ndarray and len(y_addr.shape) == 1:
        if batch_size is None:
            return outputs[y_addr]
        else:
            return outputs[:, y_addr]
    else:
        if batch_size is None:
            assert outputs.shape[0] == len(y_addr)
            assert outputs.shape[1] == len(y_addr[0])
            out_list = []
            for r, y_addr_r in enumerate(y_addr):
                out_row = []
                for c, y_addr_rc in enumerate(y_addr_r):
                    out_row.append(outputs[r, c, :][y_addr_rc])
                out_list.append(out_row)
        else:
            assert outputs.shape[0] == batch_size
            assert outputs.shape[1] == len(y_addr)
            assert outputs.shape[2] == len(y_addr[0])
            out_list = []
            for r, y_addr_r in enumerate(y_addr):
                out_row = []
                for c, y_addr_rc in enumerate(y_addr_r):
                    out_row.append(outputs[:, r, c, :][:, y_addr_rc])
                out_list.append(out_row)
    return out_list


def _setup_inference(dev, fwd, num_pulses, run_all, partial_reset=False, col_addr=None, ota_time=3, sample_time=2, num_bits=None, reset_reg=True, iteration=1, max_steps=128,
                     cds_time=3, comp_time=2, reset_time=0):
    dev.SetWireInValue(0x06, 0b01100000) # Enable neuron OTA and reg_controlled_wl
    dev.SetWireInValue(0x09, run_all | partial_reset << 1 | 0b11 << 2 | (ota_time & 0xff) << 4 | 0b01 << 12 | (num_pulses << 14) | 0b01 << 23 | cds_time << 25)
    dev.SetWireInValue(0x10, sample_time << 4 | comp_time << 12 | reset_time << 20)
    if partial_reset:
        _setup_partial_reset_hw(dev, col_addr, fwd, max_steps)
    if num_bits is not None:
        dev.SetWireInValue(0x0E, (iteration & 0xff) << 10 | (num_pulses & 0b11111) << 5 | ((num_bits-1) & 0b111) << 2 | reset_reg << 1 | 0b1)
    dev.UpdateWireIns()
    dev.SetWireInValue(0x05, 0b1 | fwd << 5) # Set inference direction
    dev.UpdateWireIns()
    
    
def _setup_partial_reset_hw(dev, col_addr=None, fwd=True, max_steps=128):
    if type(col_addr) is list:
        num_core = len(col_addr)
        shift_multiplier = 2 * num_core - 1
        single_core = False
    else:
        num_core = 1
        if fwd:
            shift_multiplier = 2 * col_addr + 1
        else:
            shift_multiplier = 2 * (col_addr + 1)
        single_core = True
    dev.SetWireInValue(0x0F, (num_core & 0b111) | (shift_multiplier & 0xf) << 3 | single_core << 7 | max_steps << 8 | 0b0 << 16)


def _trigger_neuron(dev, trigger):
    dev.ActivateTriggerIn(0x45, trigger)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if status & 0b1 != 0:
            break


def disable_inference(dev, disable_neuron=False):
    dev.SetWireInValue(0x05, 0b100000) # Disable inference mode
    if disable_neuron:
        dev.SetWireInValue(0x06, 0b00100000) # Disable neurons
    dev.UpdateWireIns()

        
def send_inputs(dev, x, x_addr, bias, row_addr, fwd, encode_func, col_addr=None, trigger=True, prep=True, **kwargs):
    # spi.reset(dev, vert=(not fwd), horz=fwd)
    if type(x) is np.ndarray:
        inputs = encode_func(x, x_addr, bias, fwd, **kwargs).flatten()
        spi.write_single_core(dev, row_addr, col_addr, vert=fwd, is_pipe_in=True, inputs=inputs, trigger=trigger, prep=prep)
    else:
        inputs = _encode_multi_row(x, x_addr, bias, fwd, encode_func=encode_func, **kwargs)
        spi.write_rows(dev, row_addr, vert=fwd, is_pipe_in=True, inputs=inputs, col_addr=col_addr,
                       trigger=trigger, prep=prep)


def _matmul_unsigned_helper_hw(dev, readout=True):
    dev.ActivateTriggerIn(0x45, 7)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if status & 0b10000 != 0:
            break
    if readout:
        _trigger_neuron(dev, 2)
               
    
def matmul_bipolar(dev, x, x_addr, y_addr, bias, row_addr, fwd, num_pulses, col_addr=None):
    send_inputs(dev, x, x_addr, bias, row_addr, fwd, _encode_input, col_addr=col_addr)
    _setup_inference(dev, fwd, num_pulses, run_all=True)

    _trigger_neuron(dev, 0)
    # disable_inference(dev)
    
    # Read register
    if type(x) is np.ndarray and len(x.shape) == 1:
        out = spi.read_single_core(dev, row_addr, col_addr, not fwd, read_shift_regs=[True, False], is_pipe_out=True)
    else:
        out = spi.read_rows(dev, row_addr, vert=not fwd, read_shift_regs=[True, False], is_pipe_out=True, col_addr=col_addr)
    return _translate_outputs(out*2+1, y_addr)


def matmul_01(dev, x, x_addr, y_addr, bias, row_addr, fwd, num_pulses, col_addr=None, prep=True):
    send_inputs(dev, x, x_addr, bias, row_addr, fwd, _encode_input_01, col_addr=col_addr, trigger=False)
    _setup_inference(dev, fwd, num_pulses, run_all=False, num_bits=1)
    _matmul_unsigned_helper_hw(dev, readout=True)
    # disable_inference(dev)
    
    # Read register
    if type(x) is np.ndarray and len(x.shape) == 1:
        out = spi.read_single_core(dev, row_addr, col_addr, not fwd, read_shift_regs=[True, False], is_pipe_out=True, prep=prep)
    else:
        out = spi.read_rows(dev, row_addr, vert=not fwd, read_shift_regs=[True, False], is_pipe_out=True, col_addr=col_addr, prep=prep)
    return _translate_outputs(out+1, y_addr)


def _write_y_addr(dev, y_addr, row_addr=None, col_addr=None):
    btarray_length = NMLO_TOTAL_LENGTH // 8
    if type(y_addr) is np.ndarray and len(y_addr.shape) == 1:
        btarray = bytearray(btarray_length)
        for y in y_addr:
            btarray[y//8] = btarray[y//8] | (0b1 << (y%8))
    else:
        assert len(y_addr) == len(row_addr)
        assert len(y_addr[0]) == len(col_addr)
        start_i = NMLO_CORE-len(y_addr[0])
        btarray = bytearray(btarray_length * len(y_addr))
        for r, addr_r in enumerate(y_addr):
            for c, addr_c in enumerate(addr_r):
                for y in addr_c:
                    btarray[r*btarray_length + (c+start_i)*NMLO_LENGTH//8 + y//8] = btarray[r*btarray_length + (c+start_i)*NMLO_LENGTH//8 + y//8] | (0b1 << (y%8))
    data = dev.WriteToPipeIn(0x81, btarray)
    dev.ActivateTriggerIn(0x45, 6)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if (status & 0b1000 != 0) and (status & (0b1 << 7) != 0):
            break
        

def _decode_row_output(dev, num_row, num_col, batch_size, row_length=NMLO_LENGTH):
    if batch_size is None:
        batch_size = 1
    ROW_ARRAY_SIZE = num_col * row_length
    BATCH_ARRAY_SIZE = ROW_ARRAY_SIZE * num_row
    L = BATCH_ARRAY_SIZE * batch_size
    
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if status & (0b1 << 13) == 0:
            break
    datain = bytearray(L)
    data = dev.ReadFromPipeOut(0xA2, datain)
    dev.UpdateWireOuts()
    status = dev.GetWireOutValue(0x28)
    assert status & (0b1 << 8) != 0
    assert status & (0b1 << 13) != 0

    np_bytes = np.frombuffer(datain, dtype=np.uint8)
    np_bytes = np_bytes.reshape([batch_size, num_row, num_col, row_length])[:, :, ::-1, :]
    num_step = np_bytes & 0x7f
    out_init = np_bytes >> 7
    output = (num_step - 0.5) * (1.0 - out_init * 2.0)
    
    return output


def _read_num_step(dev, y_addr, row_addr=None, col_addr=None, batch_size=None):
    if type(row_addr) is not list and col_addr is not None:
        outputs = _decode_row_output(dev, 1, 1, batch_size, row_length=256)
        if batch_size is None:
            outputs = outputs.flatten()
        else:
            outputs = outputs.reshape([batch_size, -1])
    else:
        num_row = len(y_addr)
        num_col = len(y_addr[0])
        outputs = _decode_row_output(dev, num_row, num_col, batch_size, row_length=NMLO_LENGTH)
        if batch_size is None:
            outputs = outputs.reshape([num_row, num_col, NMLO_LENGTH])
    return _translate_outputs(outputs, y_addr, batch_size=batch_size)
    

def _matmul_partial_reset_helper_hw(dev, y_addr, row_addr, col_addr=None, write_y_addr=True, readout=True):
    if write_y_addr:
        _write_y_addr(dev, y_addr, row_addr=row_addr, col_addr=col_addr)
    dev.ActivateTriggerIn(0x45, 5)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if status & 0b1000 != 0:
            break
    if readout:
        return _read_num_step(dev, y_addr, row_addr=row_addr, col_addr=col_addr)
    

def _matmul_dac2adc_helper(dev):
    dev.ActivateTriggerIn(0x45, 8)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if status & (0b1 << 9) != 0:
            break


def matmul_bipolar_partial_reset(dev, x, x_addr, y_addr, bias, row_addr, fwd, num_pulses, col_addr=None, prep=True):
    send_inputs(dev, x, x_addr, bias, row_addr, fwd, _encode_input, col_addr=col_addr, prep=prep)
    if prep:
        _setup_inference(dev, fwd, num_pulses, run_all=False, partial_reset=True, col_addr=col_addr)
    
    _trigger_neuron(dev, 0)
    _trigger_neuron(dev, 1)
    output = _matmul_partial_reset_helper_hw(dev, y_addr, row_addr=row_addr, col_addr=col_addr, write_y_addr=prep)
        
    # disable_inference(dev)
    return output


def matmul_01_partial_reset(dev, x, x_addr, y_addr, bias, row_addr, fwd, num_pulses, col_addr=None, prep=True):
    iteration = 1
    batch_size = None
    if type(x) is np.ndarray and len(x.shape) == 2:
        iteration = x.shape[0]
        batch_size = x.shape[0]
    send_inputs(dev, x, x_addr, bias, row_addr, fwd, _encode_input_01, col_addr=col_addr, trigger=False, prep=prep)
    if prep:
        _setup_inference(dev, fwd, num_pulses, run_all=False, partial_reset=True, col_addr=col_addr, num_bits=1, iteration=iteration)
        _write_y_addr(dev, y_addr, row_addr=row_addr, col_addr=col_addr)
    _matmul_dac2adc_helper(dev)
    return _read_num_step(dev, y_addr, row_addr=row_addr, col_addr=col_addr, batch_size=batch_size)


def matmul(dev, x, x_addr, y_addr, bias, row_addr, fwd, input_num_bits, signed, col_addr, pulse_multiplier=1, prep=True):
    iteration = 1
    batch_size = None
    if type(x) is np.ndarray:
        if len(x.shape) == 2:
            iteration = x.shape[0]
            batch_size = x.shape[0]
    else:
        if len(x[0][0].shape) == 2:
            iteration = x[0][0].shape[0]
            batch_size = x[0][0].shape[0]
    if signed:
        encode_func = _encode_input_signed
    else:
        encode_func = _encode_input_unsigned
    send_inputs(dev, x, x_addr, bias, row_addr, fwd, encode_func, col_addr=col_addr, trigger=False, prep=prep, input_num_bits=input_num_bits)
    if prep:
        _setup_inference(dev, fwd, pulse_multiplier, run_all=False, partial_reset=True, col_addr=col_addr, num_bits=input_num_bits-1 if signed else input_num_bits, iteration=iteration)
        _write_y_addr(dev, y_addr, row_addr=row_addr, col_addr=col_addr)
    _matmul_dac2adc_helper(dev)
    return _read_num_step(dev, y_addr, row_addr=row_addr, col_addr=col_addr, batch_size=batch_size)


def matmul_unsigned(dev, x, x_addr, y_addr, bias, row_addr, fwd, input_num_bits, col_addr, pulse_multiplier=1, prep=True):
    return matmul(dev, x, x_addr, y_addr, bias, row_addr, fwd, input_num_bits, False, col_addr, pulse_multiplier, prep)


def calibrate_voltage(dev, vpos, vneg, rows, cols, core_row, core_col, fwd, num_pulse, tolerance=0.05, increment=0.001, iteration=1000, max_cycle=10, vcomp_offset=0.02, verbose=False):
    M = int(len(rows)/2)
    N = int(len(cols))
    cycle = 0
    while True:
        dac_setup(dev, vpos, vneg, vcomp_offset=vcomp_offset)
        outputs = np.zeros([iteration, N])
        inputs = np.random.binomial(n=1, p=0.5, size=(iteration, M)) * 2 - 1
        for i in range(iteration):
            outputs[i, :] = matmul_bipolar(dev, inputs[i, :], rows, cols, False, core_row, fwd, num_pulse, col_addr=core_col)
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
