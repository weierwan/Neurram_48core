# DAC control
# Author: Weier Wan
#

import numpy as np

DAC_NUM = 4
DAC_NUM_CHANNEL = 8

VCOMP1 = 0.9
VCOMP2 = 0.9
VREF = 0.9
VPLUS = 1
VMINUS = 0.8
VPLFSR = 1
VNLFSR = 0.8
VPULSE_BIAS = 0.6
BIAS06 = 0.6
BIAS10 = 0.9
BIAS14 = 1.2
VSET_BL = 1.8
VSET_WL = 1.5
VRESET_SL = 1.8

DAC_VOLTAGES = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]

DAC_VREFS = [2.048, 2.048, 2.048, 3.3]


def dac_program_single(dev, chnl_idx, voltage, vref, no_op=False):
    if no_op:
        cmd = dac_translate_cmd(chnl_idx, voltage, vref, cmd=0b1111)
    else:
        cmd = dac_translate_cmd(chnl_idx, voltage, vref)
    dev.UpdateWireOuts()
    status = dev.GetWireOutValue(0x20)
    if status & 0b01 != 0: # Check whether the FIFO is full
        print('DAC FIFO is full.')
        return
    dev.SetWireInValue(0x01, cmd)   # command
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x40, 1)


def dac_program_single_daisy(dev, dac_idx, chnl_idx, voltage, vrefs=DAC_VREFS, verbose=False):
    for i in range(DAC_NUM-1, -1, -1):
        if i == dac_idx:
            dac_program_single(dev, chnl_idx, voltage, vrefs[i])
        else:
            dac_program_single(dev, chnl_idx, 0, vrefs[i], no_op=True)
        while True:
            dev.UpdateWireOuts()
            status = dev.GetWireOutValue(0x20)
            if status & 0b10 != 0:
                break
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x20)
        if status & 0b1000 != 0:
            break       
    dev.ActivateTriggerIn(0x41, 0)
    i_to = 0
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x20)
        if status & 0b100000 != 0:
            break
        if i_to == 1e6:
            print('DAC programming timeout.')
            break
        i_to += 1
    if verbose:
        print('Updated DAC %d channel %d voltage to %fV.' % (dac_idx, chnl_idx, voltage))


def dac_program_all(dev, voltages=DAC_VOLTAGES, vrefs=DAC_VREFS):
    for c in range(DAC_NUM_CHANNEL):
        for i in range(DAC_NUM):
            volt = voltages[DAC_NUM-1-i][c]
            dac_program_single(dev, c, volt, vrefs[DAC_NUM-1-i])
            while True:
                dev.UpdateWireOuts()
                status = dev.GetWireOutValue(0x20)
                if status & 0b10 != 0:
                    print('Programed DAC %d Channel %d to %fV.' % (DAC_NUM-1-i, c, volt))
                    break
        dev.ActivateTriggerIn(0x41, 0)
        print('Updated channel %d output' % c)


def dac_translate_cmd(chnl_idx, voltage, vref, cmd=0b0011):
    if chnl_idx < 0 or chnl_idx > 7:
        print('DAC channel index is out of range.')
        cmd = 0b1111
    elif voltage < 0:
        raise ValueError('DAC voltage cannot be negative.')
    elif voltage > vref:
        if (voltage - vref)  < 1e-6:
            voltage = vref
        else:
            raise ValueError('DAC voltage %.32f is out of range %.32f.' % (voltage, vref))
    if voltage == vref:
        data = int(voltage / vref * 2**16 - 1)
    else:
        data = int(voltage / vref * 2**16)
    return (cmd << 20) | (chnl_idx << 16) | data & 0x00ffffff


def ramp_up_voltage(dev, adc, channel, v_target):
    for v in np.arange(0, v_target, 0.1):
        dac_program_single_daisy(dev, adc, channel, v)