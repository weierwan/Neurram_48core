# Neuron control
# Author: Weier Wan
#
import dac_control as dac
import spi_control as spi


VREF = 0.9

def update_bias_voltages(dev, vref=None, vcomp1=None, vcomp2=None, ibias=None, vlfsr_pos=None, vlfsr_neg=None, vreset_plus=None, vreset_minus=None, vcomp_offset=0.02):
	if vref is not None:
		dac.dac_program_single_daisy(dev, 0, 2, vref)
		dac.dac_program_single_daisy(dev, 1, 1, vref)
		dac.dac_program_single_daisy(dev, 1, 4, vref)
	else:
		vref = VREF
	if vcomp1 is not None:
		dac.dac_program_single_daisy(dev, 0, 0, vcomp1 + vcomp_offset)
	if vcomp2 is not None:
		dac.dac_program_single_daisy(dev, 0, 1, vcomp2 + vcomp_offset)
	if ibias is not None:
		dac.dac_program_single_daisy(dev, 2, 0, ibias)
	if vlfsr_pos is not None:
		dac.dac_program_single_daisy(dev, 0, 5, vlfsr_pos)
	if vlfsr_neg is not None:
		dac.dac_program_single_daisy(dev, 0, 6, vlfsr_neg)
	if vreset_plus is not None:
		dac.dac_program_single_daisy(dev, 0, 4, vref + vreset_plus)
	if vreset_minus is not None:
		dac.dac_program_single_daisy(dev, 2, 2, vref - vreset_minus)


def update_pulse_voltages(dev, bias06=None, bias14=None, is_bl=True):
	if is_bl:
		if bias06 is not None:
			dac.dac_program_single_daisy(dev, 1, 0, bias06)
		# dac.dac_program_single_daisy(dev, 1, 1, bias10)
		if bias14 is not None:
			dac.dac_program_single_daisy(dev, 1, 2, bias14)
	else:
		if bias06 is not None:
			dac.dac_program_single_daisy(dev, 1, 3, bias06)
		# dac.dac_program_single_daisy(dev, 1, 4, bias10)
		if bias14 is not None:
			dac.dac_program_single_daisy(dev, 1, 5, bias14)


def setup(dev, mode, forward, run_all, partial_reset, comp_phase, ota_time, num_pulses):
	dev.SetWireInValue(0x06, 0b01100000) # Enable neuron OTA and reg_controlled_wl
	if mode == 'INF':
		if forward:
			dev.SetWireInValue(0x05, 0b100001)
		else:
			dev.SetWireInValue(0x05, 0b000001)
	elif mode == 'IFAT':
		if forward:
			dev.SetWireInValue(0x05, 0b100010)
		else:
			dev.SetWireInValue(0x05, 0b000010)
	
	controls = run_all | (partial_reset << 1) | 0b1100 | ((ota_time & 0xff) << 4) | (
		(comp_phase & 0b11) << 12) | ((num_pulses & 0xff) << 14)
	dev.SetWireInValue(0x09, controls)

	dev.UpdateWireIns()



# Deprecated, now the registers are directly configured by the FSM
def setup_reg(dev, forward):
	# spi.reset(dev, horz=forward, vert=not forward)
	if forward:
		dev.SetWireInValue(0x03, 0b0001000000)
	else:
		dev.SetWireInValue(0x03, 0b0010000000)
	dev.UpdateWireIns()


def check_idle(dev):
	while True:
		dev.UpdateWireOuts()
		status = dev.GetWireOutValue(0x28)
		if status & 0b1 != 0:
			return


def trigger_cds(dev):
	dev.ActivateTriggerIn(0x45, 0)
	check_idle(dev)


def trigger_sample_integ(dev):
	dev.ActivateTriggerIn(0x45, 1)
	check_idle(dev)


def trigger_compare_write(dev):
	dev.ActivateTriggerIn(0x45, 2)
	check_idle(dev)


def trigger_integ(dev):
	dev.ActivateTriggerIn(0x45, 3)
	check_idle(dev)


def trigger_partial_reset(dev):
	dev.ActivateTriggerIn(0x45, 4)
	check_idle(dev)
