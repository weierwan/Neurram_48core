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
    dev.SetWireInValue(0x04, vert << 10 | ((din & 0b11) << 8) | (addr & 0xff))
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x44, 1)

    
def translate_bytearray(btarray, fwd, pipe_out_steps, read_shift_regs=[True, True]):
    np_bytes = np.frombuffer(btarray, dtype=np.uint8)
    np_bits = np.unpackbits(np_bytes, bitorder='little')
    L = len(np_bits) // 2
    output = np.zeros(L)
    if read_shift_regs[0]:
        mask = (np_bits[0::2] == 1)
        output[mask] = -1
    if read_shift_regs[1]:
        mask = (np_bits[1::2] == 1)
        output[mask] = 1
    if read_shift_regs[0] and read_shift_regs[1]:
        mask = (np_bits[0::2] == 1) & (np_bits[1::2] == 1)
        output[mask] = 3
    if fwd:
        output = output.reshape([-1, pipe_out_steps * 256])
        output = output[:, ::-1]
        output = output.flatten()
    return output


def translate_bytearray_depracated(btarray, fwd):
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
    if fwd:
        nparray = nparray.reshape([-1, 256 * pipe_in_steps])
        nparray = nparray[:, ::-1]
        nparray = nparray.flatten()
    bit_array = np.zeros(L*2, dtype=bool)
    bit_array[0::2] = (nparray == -1)
    bit_array[1::2] = (nparray == 1)
    return bytearray(np.packbits(bit_array, bitorder='little'))


# Deprecated
def encode_bytearray_deprecated(nparray, fwd, pipe_in_steps):
    L = len(nparray)
    L_bt = L//4
    M = 256 * pipe_in_steps
    M_bt = M//4
    btarray = bytearray(L_bt)
    for i, n in enumerate(nparray):
        if n == 0:
            continue
        if n == 1:
            bits = 0b10
        elif n == -1:
            bits = 0b01
        else:
            raise ValueError('Element can only take value in [-1, 0, 1].')
        
        if fwd:
            mask = 0b11 << (2*(3-i%4))
            btarray[i//M*M_bt + M_bt-1-(i%M)//4] = (btarray[i//M*M_bt + M_bt-1-(i%M)//4] & (~mask)) | (bits << (2*(3-i%4)))
        else:
            mask = 0b11 << (2*(i%4))
            btarray[i//4] = (btarray[i//4] & (~mask)) | (bits << (2*(i%4)))
    return btarray


def _compose_row_mask(row_addr):
    if type(row_addr) is list:
        row_mask = 0b0
        for r in row_addr:
            row_mask = row_mask | (0b1 << r)
    else:
        row_mask = 0b1 << row_addr
    return row_mask


# Deprecated
def _swap_btarray_order(datain, num_words):
    num_bytes = num_words * 4
    num_rows = len(datain) // num_bytes
    data_swapped = bytearray(len(datain))
    for r in range(num_rows):
        r_swap = num_rows - 1 - r
        data_swapped[r*num_bytes : (r+1)*num_bytes] = datain[r_swap*num_bytes : (r_swap+1)*num_bytes]
    return data_swapped


def pipe_out(dev, row_addr, forward, num_words, pipe_out_steps, read_shift_regs=[True, True]):
    datain = bytearray(num_words*4*len(row_addr))
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
    return translate_bytearray(datain, forward, pipe_out_steps, read_shift_regs).reshape([len(row_addr), -1])

    
def spi_read(dev, row_addr, forward, overwrite=True, shift_multiplier=1, pipe_out_steps=1, read_shift_regs=[True, True], is_pipe_out=True,
             num_words=16, extra_shift_cycles=0, trigger=True, prep=True):
    if prep:
        dev.SetWireInValue(0x15, (_compose_row_mask(row_addr) & 0xff) | (num_words & 0xff) << 18)
        dev.SetWireInValue(0x0D, 0b01 | (shift_multiplier & 0xf) << 2 | (pipe_out_steps & 0xf) << 10 | forward << 14 | (extra_shift_cycles & 0xf) << 15)
        dev.UpdateWireIns()
    
    if trigger:
        dev.ActivateTriggerIn(0x44, 0)
        while True:
            dev.UpdateWireOuts()
            status = dev.GetWireOutValue(0x28)
            if status & 0b100 != 0:
                break

    if is_pipe_out:
        return pipe_out(dev, row_addr, forward, num_words, pipe_out_steps, read_shift_regs)


def pipe_in(dev, inputs, row_addr, forward, pipe_in_steps=1, check_done=False):
    dataout = encode_bytearray(inputs.flatten(), forward, pipe_in_steps=pipe_in_steps)
    data = dev.WriteToPipeIn(0x80, dataout)
    if check_done:
        while True:
            dev.UpdateWireOuts()
            status = dev.GetWireOutValue(0x28)
            if status & 0b100000 != 0:
                break       


def spi_write(dev, row_addr, forward, inputs=None, overwrite=True, shift_multiplier=1, pipe_in_steps=1, extra_shift_cycles=0,
              is_pipe_in=True, trigger=True, prep=True):
    if prep:
        num_words = 0
        if inputs is not None:
            if len(inputs.shape) == 1:
                num_words = inputs.shape[0] // 16
            else:
                num_words = inputs.shape[1] // 16
        dev.SetWireInValue(0x15, (_compose_row_mask(row_addr) & 0xff) | (num_words & 0x3ff) << 8)
        dev.SetWireInValue(0x0D, 0b10 | (shift_multiplier & 0xf) << 2 | (pipe_in_steps & 0xf) << 6 | forward << 14 | (extra_shift_cycles & 0xf) << 15)
        dev.UpdateWireIns()

    if is_pipe_in:
        pipe_in(dev, inputs, row_addr, forward, pipe_in_steps=pipe_in_steps)
    
    if trigger:
        dev.ActivateTriggerIn(0x44, 0)
        while True:
            dev.UpdateWireOuts()
            status = dev.GetWireOutValue(0x28)
            if status & 0b100 != 0:
                break       


def write_single_core(dev, row_addr, col_addr, vert, is_pipe_in=False, inputs=None, trigger=True, prep=True):
    if col_addr <= 2:
        fwd = True
        if vert:
            shift_multiplier = 2 * (col_addr + 1)
        else:
            shift_multiplier = 2 * col_addr + 1
    else:
        fwd = False
        if vert:
            shift_multiplier = 2 * (5 - col_addr) + 1
        else:
            shift_multiplier = 2 * (6 - col_addr)
    spi_write(dev, [row_addr], fwd, inputs=np.array([inputs]), shift_multiplier=shift_multiplier, is_pipe_in=is_pipe_in, trigger=trigger, prep=prep)


def read_single_core(dev, row_addr, col_addr, vert, read_shift_regs=[True, True], is_pipe_out=False, batch_size=1, trigger=True, prep=True):
    if col_addr <= 2:
        fwd = False
        if vert:
            shift_multiplier = 2 * (col_addr + 1)
        else:
            shift_multiplier = 2 * col_addr + 1
    else:
        fwd = True
        if vert:
            shift_multiplier = 2 * (5 - col_addr) + 1
        else:
            shift_multiplier = 2 * (6 - col_addr)
    return spi_read(dev, [row_addr], fwd, shift_multiplier=shift_multiplier, read_shift_regs=read_shift_regs,
                    is_pipe_out=is_pipe_out, num_words=16*batch_size, trigger=trigger, prep=prep)[0]


def write_rows(dev, row_addr, vert, is_pipe_in=False, inputs=None, col_addr=range(6), trigger=True, prep=True):
    if vert:
        extra_shift_cycles = 2 * col_addr[0]
    else:
        extra_shift_cycles = 2 * (5 - col_addr[-1])
    spi_write(dev, row_addr, vert, inputs=inputs, shift_multiplier=2, pipe_in_steps=len(col_addr), extra_shift_cycles=extra_shift_cycles,
              is_pipe_in=is_pipe_in, trigger=trigger, prep=prep)


def read_rows(dev, row_addr, vert, read_shift_regs=[True, True], is_pipe_out=False, col_addr=range(6), trigger=True, prep=True):
    if vert:
        extra_shift_cycles = 2 * col_addr[0]
    else:
        extra_shift_cycles = 2 * (5 - col_addr[-1])
    return spi_read(dev, row_addr, not vert, shift_multiplier=2, pipe_out_steps=len(col_addr), read_shift_regs=read_shift_regs,
                    is_pipe_out=is_pipe_out, num_words=16*len(col_addr), extra_shift_cycles=extra_shift_cycles, trigger=trigger, prep=prep)


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


def enable_single_row(dev, addr_horz, first_half=True, second_half=True):
    reset_core_enable_reg(dev)
    if first_half:
        enable_core(dev, swap_addr(addr_horz), 0b000, dec_enable=0b01)
        disable_core(dev, addr_horz, 0b000, dec_enable=0b01)
    if second_half:
        enable_core(dev, swap_addr(addr_horz), 0b100, dec_enable=0b01)
        disable_core(dev, addr_horz, 0b100, dec_enable=0b01)


def enable_single_col(dev, addr_vert, first_half=True, second_half=True):
    reset_core_enable_reg(dev)
    if first_half:
        enable_core(dev, 0b000, swap_addr(addr_vert), dec_enable=0b10)
        disable_core(dev, 0b000, addr_vert, dec_enable=0b10)
    if second_half:
        enable_core(dev, 0b100, swap_addr(addr_vert), dec_enable=0b10)
        disable_core(dev, 0b100, addr_vert, dec_enable=0b10)


def enable_2_cores_vert(dev, addr_horz, addr_vert):
    reset_core_enable_reg(dev)
    enable_core(dev, swap_addr(addr_horz), swap_addr(addr_vert))
    disable_core(dev, swap_addr(addr_horz), addr_vert)
    disable_core(dev, addr_horz, swap_addr(addr_vert))
    enable_core(dev, swap_addr(addr_horz+4), swap_addr(addr_vert))
    disable_core(dev, swap_addr(addr_horz+4), addr_vert)
    disable_core(dev, addr_horz+4, swap_addr(addr_vert))


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