# ADC & TIA control
# Author: Weier Wan
#

import time
import numpy as np
import dac_control as dac

ADC_RRAM_NUM = 1
ADC_RRAM_VREF = 2.048

CLK_PERIOD = 1.28e-6
C_INTEG = 1e-10

ADDR_RRAM_TRIGGER = 0x43
VALID_BIT_RRAM = 8


def adc_reading_translate(dout0, dout1, adc_num, vref=ADC_RRAM_VREF):
    adc0_read = read_bin2fp((dout1 & 0x3ffff0) >> 4, vref)
    adc1_read = read_bin2fp(((dout1 & 0xf) << 14) | ((dout0 & 0xfffc0000) >> 18), vref)
    adc2_read = read_bin2fp(dout0 & 0x3ffff, vref)
    if adc_num == 2:
        return (adc1_read, adc2_read)
    return (adc0_read, adc1_read, adc2_read)


def read_bin2fp(reading, vref=ADC_RRAM_VREF):
    return reading * vref / 2**18


def adc_reading_encode(voltage, vref):
    return int(voltage * 2**18 / vref) & 0x3ffff



def read_current_auto_range(dev, t_shld, t_delta, vref=0.9, output_t=False, verbose=False):
    t_setting = ((int(t_delta) & 0xffff) << 16) | (int(t_shld) & 0xffff)
    dev.SetWireInValue(0x02, t_setting)
    dev.SetWireInValue(0x12, adc_reading_encode(vref, ADC_RRAM_VREF))
    dev.UpdateWireIns()

    dev.ActivateTriggerIn(ADDR_RRAM_TRIGGER, 2)
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x20)
        if status & 0b100 != 0:
            d_start = dev.GetWireOutValue(0x29)
            d_end = dev.GetWireOutValue(0x2a)
            timing = dev.GetWireOutValue(0x2b)
            dev.ActivateTriggerIn(ADDR_RRAM_TRIGGER, 3) # ack
            break

    d_start = read_bin2fp(d_start, ADC_RRAM_VREF)
    d_end = read_bin2fp(d_end, ADC_RRAM_VREF)
    delta_v = d_start - d_end
    t_shld = timing & 0xffff
    t_delta = (timing & 0xffff0000) >> 16

    current = C_INTEG * delta_v / (t_delta * CLK_PERIOD)
    if verbose:
        print(d_start)
        print(d_end)
        print(t_delta)
    if output_t:
        return (current, t_shld, t_delta)
    return current



def adc_setup(dev, vread=1.0, vref=0.9, vread_wl=3.3):
    dac.dac_program_single_daisy(dev, 1, 6, vread)
    dac.dac_program_single_daisy(dev, 1, 7, vref)

    
def read_average_resistance(dev, vread, vref, t_shld, t_delta, read_cycles, ignore_cycles, adc_idx=0, dac_setup=False, current=False,
                           verbose=False, output_raw=False, vread_wl=3.3):
    if dac_setup:
        adc_setup(dev, vread, vref, vread_wl)
    
    dev.SetWireInValue(0x0B, 0b11)
    dev.UpdateWireIns()
    
    currents = np.zeros(read_cycles)
    for i in range(read_cycles):
        currents[i], t_shld, t_delta = read_current_auto_range(dev, t_shld, t_delta, vref=vref,
                                                    output_t=True, verbose=verbose)
    
    dev.SetWireInValue(0x0B, 0b00)
    dev.UpdateWireIns()
    
    if output_raw:
        return (vread-vref) / currents[ignore_cycles:]
    currents_avg = np.mean(currents[ignore_cycles:])
    if currents_avg == 0:
        print('All voltage readigns are identical')
        return
    # Note that this compute geometric mean since what we care is conductance
    if current:
        return currents_avg
    return (vread - vref) / currents_avg
