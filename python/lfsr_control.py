# SPI control
# import binascii
import numpy as np
import dac_control as dac

def set_reg(dev):
    dev.SetWireInValue(0x0A, 0b1)
    dev.UpdateWireIns()
    dev.SetWireInValue(0x0A, 0b0)
    dev.UpdateWireIns()
    
def shift_reg(dev):
    datain = bytearray(64)
    dev.ActivateTriggerIn(0x44, 4)
    data = dev.ReadFromPipeOut(0xA1, datain)
#     print binascii.hexlify(datain)
    return datain
    
def enable_lfsr_mode(dev):
    dev.SetWireInValue(0x05, 0b000100)
    dev.UpdateWireIns()        
    
def pulse(dev, vpos=None, vneg=None):
    if vpos is not None:
        dac.dac_program_single_daisy(dev, 0, 5, vpos)
    if vneg is not None:
        dac.dac_program_single_daisy(dev, 0, 6, vneg)
    dev.ActivateTriggerIn(0x44, 5)