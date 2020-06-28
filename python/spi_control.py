# SPI control
import numpy as np
import time

def reset(dev, horz=True, vert=True):
    dev.SetWireInValue(0x03, (horz | vert << 1))
    dev.UpdateWireIns()
    dev.SetWireInValue(0x03, 0b00)
    dev.UpdateWireIns()

    
def random_write(dev, vert, addr_horz, addr_vert, addr, din, enable_core=False):
    if enable_core:
        enable_single_core(dev, addr_horz, addr_vert)
    dev.SetWireInValue(0x13, swap_addr(addr_horz) | swap_addr(addr_vert) << 3)
    dev.SetWireInValue(0x04, ((din & 0b11) << 8) | (addr & 0xff))
    if vert:
        dev.SetWireInValue(0x03, 0b1010111100)
    else:
        dev.SetWireInValue(0x03, 0b1001111100)
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x44, 1)
    # dev.SetWireInValue(0x03, 0b100000)
    # dev.UpdateWireIns()

    
def translate_bytearray(btarray, fwd):
    output = np.zeros(len(btarray) * 4, dtype=np.int8)
    for i, bt in enumerate(btarray):
        for j in range(4):
            tmp = bt >> (j*2) & 0b11
            if tmp == 0b10:
                output[i*4+j] = 1
            elif tmp == 0b00:
                output[i*4+j] = 0
            elif tmp == 0b01:
                output[i*4+j] = -1
            else:
                output[i*4+j] = 3
    if fwd:
        output = output[::-1]
    return output


def encode_bytearray(nparray, fwd, pipe_in_steps):
    L = len(nparray)
    L_bt = L/4
    M = 256 * pipe_in_steps
    M_bt = M/4
    btarray = bytearray(L_bt)
    for i, n in enumerate(nparray):
        if n == 1:
            bits = 0b10
        elif n == 0:
            bits = 0b00
        elif n == -1:
            bits = 0b01
        else:
            raise ValueError('Element can only take value in [-1, 0, 1].')
        
        if fwd:
            mask = 0b11 << (2*(3-i%4))
            btarray[i//M*M_bt + M_bt-1-(i%M)//4] = (btarray[i//M*M_bt + M_bt-1-(i%M)//4] & (~mask)) | (bits << (2*(3-i%4)))
        else:
            mask = 0b11 << (2*(i%4))
            btarray[i/4] = (btarray[i/4] & (~mask)) | (bits << (2*(i%4)))
    return btarray


def _compose_row_mask(row_addr):
    if type(row_addr) is list:
        row_mask = 0b0
        for r in row_addr:
            row_mask = row_mask | (0b1 << r)
    else:
        row_mask = 0b1 << row_addr
    return row_mask


def _swap_btarray_order(datain, num_words):
    num_bytes = num_words * 4
    num_rows = len(datain) // num_bytes
    data_swapped = bytearray(len(datain))
    for r in range(num_rows):
        r_swap = num_rows - 1 - r
        data_swapped[r*num_bytes : (r+1)*num_bytes] = datain[r_swap*num_bytes : (r_swap+1)*num_bytes]
    return data_swapped


def pipe_out(dev, row_addr, forward, num_words, setup=True):
    if setup:
        dev.SetWireInValue(0x15, (_compose_row_mask(row_addr) & 0xff) | (num_words & 0xff) << 18)
        dev.UpdateWireIns()
    datain = bytearray(num_words*4*len(row_addr))
    # time.sleep(0.01)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if status & (0b1 << 11) == 0:
            break
    data = dev.ReadFromPipeOut(0xA0, datain)
    dev.UpdateWireOuts()
    status = dev.GetWireOutValue(0x28)
    assert status & (0b1 << 6) != 0
    assert status & (0b1 << 11) != 0
    if forward:
        datain = _swap_btarray_order(datain, num_words)
    return translate_bytearray(datain, forward).reshape([len(row_addr), -1])

    
def spi_read(dev, row_addr, forward, overwrite=True, shift_multiplier=1, pipe_out_steps=1, is_pipe_out=True, num_words=16, trigger=True):
    dev.SetWireInValue(0x15, (_compose_row_mask(row_addr) & 0xff) | (num_words & 0xff) << 18)
    if forward:
        if overwrite:
            dev.SetWireInValue(0x03, 0b1111000000)
        else:
            dev.SetWireInValue(0x03, 0b1110000000)
    else:
        if overwrite:
            dev.SetWireInValue(0x03, 0b0111000000)
        else:
            dev.SetWireInValue(0x03, 0b0101000000)
    dev.SetWireInValue(0x0D, 0b01 | (shift_multiplier & 0xf) << 2 | (pipe_out_steps & 0xf) << 10)
    dev.UpdateWireIns()
    
    if trigger:
        dev.ActivateTriggerIn(0x44, 0)
        while True:
            dev.UpdateWireOuts()
            status = dev.GetWireOutValue(0x28)
            if status & 0b100 != 0:
                break

    if pipe_out:
        return pipe_out(dev, row_addr, forward, num_words, setup=False)


def pipe_in(dev, inputs, row_addr, forward, pipe_in_steps=1):
    if len(inputs.shape) == 1:
        inputs = np.array([inputs])
        row_addr = [row_addr]
    # Check the input dimension
    assert inputs.shape[0] == len(row_addr)
    num_words = inputs.shape[1] // 16
    dev.SetWireInValue(0x15, (_compose_row_mask(row_addr) & 0xff) | (num_words & 0x3ff) << 8)
    dev.UpdateWireIns()

    nparray = np.empty([0,])
    for inp in inputs:
        nparray = np.hstack([nparray, inp])

    dataout = encode_bytearray(nparray, forward, pipe_in_steps=pipe_in_steps)
    data = dev.WriteToPipeIn(0x80, dataout)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x28)
        if status & 0b100000 != 0:
            break       


def spi_write(dev, row_addr, forward, inputs=None, overwrite=True, shift_multiplier=1, pipe_in_steps=1, is_pipe_in=True, trigger=True):
    if pipe_in:
        pipe_in(dev, inputs, row_addr, forward, pipe_in_steps=pipe_in_steps)
    else:
        dev.SetWireInValue(0x15, (_compose_row_mask(row_addr) & 0xff))
    
    if forward:
        if overwrite:
            dev.SetWireInValue(0x03, 0b1111000000)
        else:
            dev.SetWireInValue(0x03, 0b1101000000)
    else:
        if overwrite:
            dev.SetWireInValue(0x03, 0b0111000000)
        else:
            dev.SetWireInValue(0x03, 0b0110000000)
    dev.SetWireInValue(0x0D, 0b10 | (shift_multiplier & 0xf) << 2 | (pipe_in_steps & 0xf) << 6)
    dev.UpdateWireIns()
    if trigger:
        dev.ActivateTriggerIn(0x44, 0)
        while True:
            dev.UpdateWireOuts()
            status = dev.GetWireOutValue(0x28)
            if status & 0b100 != 0:
                break       


def write_single_core(dev, row_addr, col_addr, vert, is_pipe_in=False, inputs=None, trigger=True):
    if vert:
        shift_multiplier = 2 * (col_addr + 1)
    else:
        shift_multiplier = 2 * col_addr + 1
    spi_write(dev, [row_addr], True, inputs=np.array([inputs]), shift_multiplier=shift_multiplier, is_pipe_in=is_pipe_in, trigger=trigger)


def read_single_core(dev, row_addr, col_addr, vert, is_pipe_out=False, trigger=True):
    if vert:
        shift_multiplier = 2 * (col_addr + 1)
    else:
        shift_multiplier = 2 * col_addr + 1
    return spi_read(dev, [row_addr], False, shift_multiplier=shift_multiplier, is_pipe_out=is_pipe_out, num_words=16, trigger=trigger)[0]


def write_rows(dev, row_addr, vert, is_pipe_in=False, inputs=None, pipe_in_steps=6, trigger=True):
    # if inputs is not None:
    #     assert inputs.shape[1] == 256 * pipe_in_steps
    spi_write(dev, row_addr, vert, inputs=inputs, shift_multiplier=2, pipe_in_steps=pipe_in_steps, is_pipe_in=is_pipe_in, trigger=trigger)


def read_rows(dev, row_addr, vert, is_pipe_out=False, pipe_out_steps=6, trigger=True):
    return spi_read(dev, row_addr, not vert, shift_multiplier=2, pipe_out_steps=pipe_out_steps, is_pipe_out=is_pipe_out, num_words=16*pipe_out_steps, trigger=trigger)


def enable_core(dev, addr_horz, addr_vert, dec_enable=0b11):
    dev.SetWireInValue(0x13, compose_addr(addr_horz, addr_vert, dec_enable=dec_enable))
    dev.SetWireInValue(0x14, 0b100)
    dev.UpdateWireIns()
    dev.SetWireInValue(0x14, 0b110)
    dev.UpdateWireIns()
    dev.SetWireInValue(0x14, 0b000)
    dev.UpdateWireIns()


def disable_core(dev, addr_horz, addr_vert, dec_enable=0b11):
    dev.SetWireInValue(0x13, compose_addr(addr_horz, addr_vert, dec_enable=dec_enable))
    dev.SetWireInValue(0x14, 0b000)
    dev.UpdateWireIns()
    dev.SetWireInValue(0x14, 0b010)
    dev.UpdateWireIns()
    dev.SetWireInValue(0x14, 0b000)
    dev.UpdateWireIns()


def enable_single_core(dev, addr_horz, addr_vert):
    reset_core_enable_reg(dev)
    enable_core(dev, swap_addr(addr_horz), swap_addr(addr_vert))
    disable_core(dev, swap_addr(addr_horz), addr_vert)
    disable_core(dev, addr_horz, swap_addr(addr_vert))


def enable_single_row(dev, addr_horz):
    reset_core_enable_reg(dev)
    enable_core(dev, swap_addr(addr_horz), 0b000, dec_enable=0b01)
    enable_core(dev, swap_addr(addr_horz), 0b100, dec_enable=0b01)
    disable_core(dev, addr_horz, 0b000, dec_enable=0b01)
    disable_core(dev, addr_horz, 0b100, dec_enable=0b01)


def enable_3_rows(dev, exclude_row):
    reset_core_enable_reg(dev)
    enable_core(dev, exclude_row, 0b000, dec_enable=0b01)
    

def compose_addr(addr_horz, addr_vert, dec_enable):
    return (addr_horz & 0b111) | ((addr_vert & 0b111) << 3) | (dec_enable << 6)


def swap_addr(addr):
    if addr == 3 or addr == 7:
        return addr - 3
    else:
        return addr + 1


def reset_core_enable_reg(dev):
    dev.SetWireInValue(0x14, 0b001)
    dev.UpdateWireIns()
    dev.SetWireInValue(0x14, 0b000)
    dev.UpdateWireIns()