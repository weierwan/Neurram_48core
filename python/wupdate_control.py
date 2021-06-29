# Weight update control
# Author: Weier Wan

import time
import numpy as np
import spi_control as spi
import adc_control as adc
import dac_control as dac

SYS_CLK_PERIOD = 10e-9

def write_reg(dev, row, col, core_row, core_col, enable_core=False):
    row = int(row)
    col = int(col)
    core_row = int(core_row)
    core_col = int(core_col)
    if enable_core:
        spi.enable_single_core(dev, core_row, core_col)
    spi.reset(dev)
    spi.random_write(dev, True, core_row, core_col, row, 0b01)
    spi.random_write(dev, False, core_row, core_col, col, 0b01)

    
def enable_wupdate_mode(dev):
    dev.SetWireInValue(0x05, 0b000000)
    dev.UpdateWireIns()


def disable_wupdate_mode(dev):
    dev.SetWireInValue(0x05, 0b000000)
    dev.UpdateWireIns()

    
def read_single_rram(dev, row, col, vread, vref, t_shld, t_delta):
    write_reg(dev, row, col)
    enable_wupdate_mode(dev)
    res = adc.read_resistance(dev, vread, vref, t_shld, t_delta)
    disable_wupdate_mode(dev)
    spi.reset(dev)
    return res


# Deprecated. Replaced by read_average_resistance in adc_control
def read_average(dev, vread, vref, t_shld, t_delta, read_cycles, ignore_cycles):
    readings = np.zeros(read_cycles)
    for i in range(read_cycles):
        readings[i] = adc.read_resistance(dev, vread, vref, t_shld, t_delta, verbose=False)
    reading_avg = np.mean(readings[ignore_cycles:])
    return reading_avg


def ramp_up_voltage(dev, adc, channel, v_target):
    for v in np.arange(0, v_target, 0.1):
        dac.dac_program_single_daisy(dev, adc, channel, v)


def wupdate_setup(dev, vset_bl, vset_wl, vreset_sl, pulse_width, vreset_wl=5.0, wl_pulse_width=None):
    dac.ramp_up_voltage(dev, 3, 0, vset_bl)
    dac.ramp_up_voltage(dev, 3, 1, vset_wl)
    dac.ramp_up_voltage(dev, 3, 2, vreset_sl)
    dac.ramp_up_voltage(dev, 3, 3, vreset_wl)
    dev.SetWireInValue(0x07, int(pulse_width/SYS_CLK_PERIOD) & 0xffffffff)
    if wl_pulse_width is None:
        wl_pulse_width = pulse_width
    dev.SetWireInValue(0x0C, int(wl_pulse_width/SYS_CLK_PERIOD) & 0xffffffff)
    dev.UpdateWireIns()

    
def apply_set_pulse(dev, vset_bl, vset_wl, tset, prep=True):
    if prep:
        # dac.dac_program_single_daisy(dev, 3, 0, 1.0)
        dac.ramp_up_voltage(dev, 3, 0, vset_bl)
        # dac.dac_program_single_daisy(dev, 3, 1, 1.0)
        dac.ramp_up_voltage(dev, 3, 1, vset_wl)
        dev.SetWireInValue(0x07, int(tset/SYS_CLK_PERIOD) & 0xffffffff)
        dev.UpdateWireIns()
    dev.SetWireInValue(0x0B, 0b10)
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x40, 2) # WL, BL and SL all get pulsed
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x20)
        if status & 0b10000 != 0:
            dev.ActivateTriggerIn(0x40, 4) # ack
            break
    dev.SetWireInValue(0x0B, 0b00)
    dev.UpdateWireIns()

    
def apply_reset_pulse(dev, vreset_sl, treset, vreset_wl=5.0, prep=True):
    if prep:
        # dac.dac_program_single_daisy(dev, 3, 2, 1.0)
        dac.ramp_up_voltage(dev, 3, 2, vreset_sl)
        # dac.dac_program_single_daisy(dev, 3, 3, 1.0)
        dac.ramp_up_voltage(dev, 3, 3, vreset_wl)
        dev.SetWireInValue(0x07, int(treset/SYS_CLK_PERIOD) & 0xffffffff)
        dev.UpdateWireIns()
    dev.SetWireInValue(0x0B, 0b01)
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x40, 3) # WL, BL and SL all get pulsed
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x20)
        if status & 0b10000 != 0:
            dev.ActivateTriggerIn(0x40, 4) # ack
            break
    dev.SetWireInValue(0x0B, 0b00)
    dev.UpdateWireIns()

    
def activate_wl(dev, set_op=None, v_wl=3.3, pulse_width=1e-6, prep=False):
    if prep:
        dev.SetWireInValue(0x0C, int(pulse_width/SYS_CLK_PERIOD) & 0xffffffff)
        dev.UpdateWireIns()
        if set_op:
            dac.dac_program_single_daisy(dev, 2, 3, v_wl)
        else:
            dac.dac_program_single_daisy(dev, 2, 1, v_wl)
    if set_op == 'SET':
        dev.SetWireInValue(0x0B, 0b01)
    elif set_op == 'RESET':
        dev.SetWireInValue(0x0B, 0b10)
    else:
        dev.SetWireInValue(0x0B, 0b11)
    dev.UpdateWireIns()
    dev.ActivateTriggerIn(0x40, 5) # WL get pulsed
    while True:
        dev.UpdateWireOuts()
        status = dev.GetWireOutValue(0x20)
        if status & 0b10000 != 0:
            dev.ActivateTriggerIn(0x40, 4) # ack
            break
    dev.SetWireInValue(0x0B, 0b00)
    dev.UpdateWireIns()
        
    
def find_vset_wl(dev, row, col, core_row, core_col, vset_bl, vset_wl_start, vset_wl_end, tset, r_target, v_incr=0.1, tread=200, pulse_first=False, verbose=False):
    write_reg(dev, row, col, core_row, core_col)
    # enable_wupdate_mode(dev)
    if pulse_first:
        res = np.inf
    else:
        res = adc.read_average_resistance(dev, 1.0, 0.9, tread/2, tread, 10, 5, dac_setup=False)
    if verbose:
        print(res)
    if res > 0 and res <= r_target:
        print('The cell has already been formed.')
        return (0.0, res)
    vset_wl_range = np.arange(vset_wl_start, vset_wl_end, v_incr)
    for vset_wl in vset_wl_range:
        apply_set_pulse(dev, vset_bl, vset_wl, tset)
        res = adc.read_average_resistance(dev, 1.0, 0.9, tread/2, tread, 10, 5, dac_setup=False)
        if verbose:
            print(res)
        if res > 0 and res <= r_target:
            res = adc.read_average_resistance(dev, 1.0, 0.9, tread/2, tread, 10, 5, dac_setup=False)
            if res > 0 and res <= r_target:
                print('Vset_wl = %f, Rset = %f' % (vset_wl, res))
                # disable_wupdate_mode(dev)
                return (vset_wl, res)
    # disable_wupdate_mode(dev)
    print('The RRAM is not SET')
    return (0, 0)


def find_vset_bl(dev, row, col, core_row, core_col, vset_wl, vset_bl_start, vset_bl_end, tset, r_target, v_incr=0.1, tread=200, pulse_first=False, verbose=False):
    write_reg(dev, row, col, core_row, core_col)
    # enable_wupdate_mode(dev)
    if pulse_first:
        res = np.inf
    else:
        res = adc.read_average_resistance(dev, 1.0, 0.9, tread/2, tread, 10, 5, dac_setup=False)
    if verbose:
        print(res)
    if res > 0 and res <= r_target:
        print('The cell has already been formed.')
        return (0.0, res)
    vset_bl_range = np.arange(vset_bl_start, vset_bl_end, v_incr)
    for vset_bl in vset_bl_range:
        apply_set_pulse(dev, vset_bl, vset_wl, tset)
        res = adc.read_average_resistance(dev, 1.0, 0.9, tread/2, tread, 10, 5, dac_setup=False)
        if verbose:
            print(res)
        if res > 0 and res <= r_target:
            res = adc.read_average_resistance(dev, 1.0, 0.9, tread/2, tread, 10, 5, dac_setup=False)
            print('Vset = %f, Rset = %f' % (vset_bl, res))
            # disable_wupdate_mode(dev)
            return (vset_bl, res)
    # disable_wupdate_mode(dev)
    print('The RRAM is not SET')
    return (0, 0)
    
    
def program_increment(dev, row, col, core_row, core_col,
                      r_low, r_high,
                      vset_start, vset_end, vset_wl, vreset_start, vreset_end, vreset_wl=5.0,
                      pulse_width=1e-7, iteration_limit=10, max_pulse_limit=10, vread=1.0, vref=0.9, t_delta=20,
                      read_cycles=8, ignore_cycles=3,
                      incremental=True, adc_setup=False, verbose=1, record_history=False, final_pulse=None):

    write_reg(dev, int(row), int(col), int(core_row), int(core_col))
    readings = []
    pulses = []
    pulse_count = 0
    last_pulse = 0
    # enable_wupdate_mode(dev)
    t_shld = t_delta/2
    read = adc.read_average_resistance(dev, vread=vread, vref=vref, t_shld=t_shld, t_delta=t_delta,
                                       read_cycles=read_cycles, ignore_cycles=ignore_cycles, dac_setup=adc_setup)
    if verbose > 1:
        print(read)
    if record_history:
        readings.append(read)
    iter_count = 0

    while True:
        v_reset = vreset_start
        max_pulse_count_reset = 0
        while read < r_low and max_pulse_count_reset < max_pulse_limit:
            apply_reset_pulse(dev, v_reset, pulse_width, vreset_wl=vreset_wl)
            last_pulse = -v_reset
            pulse_count += 1
            read = adc.read_average_resistance(dev, vread=vread, vref=vref, t_shld=t_shld, t_delta=t_delta,
                                               read_cycles=read_cycles, ignore_cycles=ignore_cycles, dac_setup=adc_setup)
            if verbose > 1:
                print(read)
            if record_history:
                pulses.append(-v_reset)
                readings.append(read)
            if v_reset >= vreset_end:
                max_pulse_count_reset += 1
            elif incremental:
                v_reset += 0.1
        v_set = vset_start # v_set_wl = VSETWL
        max_pulse_count_set = 0
        while read > r_high and max_pulse_count_set < max_pulse_limit: #v_set_wl <= VSETWL_LIMIT:
            apply_set_pulse(dev, v_set, vset_wl, pulse_width) #VSET, v_set_wl, 1e-7)
            last_pulse = v_set
            pulse_count += 1
            read = adc.read_average_resistance(dev, vread=vread, vref=vref, t_shld=t_shld, t_delta=t_delta,
                                               read_cycles=read_cycles, ignore_cycles=ignore_cycles, dac_setup=adc_setup)
            if verbose > 1:
                print(read)
            if record_history:
                pulses.append(v_set)
                readings.append(read)
            if v_set >= vset_end:
                max_pulse_count_set += 1
            elif incremental:
                v_set += 0.1 # v_set_wl += 0.1
        iter_count += 1
        if read > r_low and read < r_high:
            if (final_pulse is None) or (last_pulse * final_pulse >= 0):
                read = adc.read_average_resistance(dev, vread=vread, vref=vref, t_shld=t_shld, t_delta=t_delta, 
                                                   read_cycles=read_cycles, ignore_cycles=ignore_cycles, dac_setup=adc_setup)
                if read > r_low and read < r_high:
                    if verbose > 0:
                        print('RRAM is programmed within the range.')
                    return (True, read, pulse_count, pulses, readings)
        if iter_count > iteration_limit or max_pulse_count_reset >= max_pulse_limit or max_pulse_count_set >= max_pulse_limit:
            if verbose > 0:
                print('Programming exceeds maximum iteration.')
            return (False, read, pulse_count, pulses, readings)


def outer_product_update_manual(dev, x, y, rows, cols, dac_setup=False, charge_based_set=False):
    assert len(x) == len(rows)
    assert len(y) == len(cols)

    spi.reset(dev)
    x_reg_p = np.zeros(256)
    x_reg_n = np.zeros(256)
    y_reg_p = np.zeros(256)
    y_reg_n = np.zeros(256)
    x_reg_p[rows[x==1]] = -1
    x_reg_n[rows[x==-1]] = -1
    y_reg_p[cols[y==1]] = -1
    y_reg_n[cols[y==-1]] = -1

    spi.spi_write(dev, x_reg_p, forward=False, overwrite=False)
    spi.spi_write(dev, y_reg_p, forward=True, overwrite=False)
    apply_set_pulse(dev, 1.5, 3.0, 1e-6, prep=dac_setup)
    if charge_based_set:
        activate_wl(dev)

    spi.spi_write(dev, y_reg_n, forward=True, overwrite=False)
    apply_reset_pulse(dev, 1.5, 1e-6, prep=dac_setup)

    spi.spi_write(dev, x_reg_n, forward=False, overwrite=False)
    apply_set_pulse(dev, 1.5, 3.0, 1e-6, prep=dac_setup)
    if charge_based_set:
        activate_wl(dev)

    spi.spi_write(dev, y_reg_p, forward=True, overwrite=False)
    apply_reset_pulse(dev, 1.5, 1e-6, prep=dac_setup)

    spi.reset(dev)
    
    
def outer_product_update_reset(dev, x, y, rows, cols, dac_setup=False):
    assert len(x) == len(rows)
    assert len(y) == len(cols)

    spi.reset(dev)
    x_reg_p = np.zeros(256)
    x_reg_n = np.zeros(256)
    y_reg_p = np.zeros(256)
    y_reg_n = np.zeros(256)
    x_reg_p[rows[x==1]] = -1
    x_reg_n[rows[x==-1]] = -1
    y_reg_p[cols[y==1]] = -1
    y_reg_n[cols[y==-1]] = -1

    spi.spi_write(dev, x_reg_p, forward=False, overwrite=False)
    spi.spi_write(dev, y_reg_n, forward=True, overwrite=False)
    apply_reset_pulse(dev, 1.5, 1e-6, prep=dac_setup)

    spi.spi_write(dev, x_reg_n, forward=False, overwrite=False)
    spi.spi_write(dev, y_reg_p, forward=True, overwrite=False)
    apply_reset_pulse(dev, 1.5, 1e-6, prep=dac_setup)

    spi.reset(dev)
    
    
def outer_product_update_set(dev, x, y, rows, cols, dac_setup=False):
    assert len(x) == len(rows)
    assert len(y) == len(cols)

    spi.reset(dev)
    x_reg_p = np.zeros(256)
    x_reg_n = np.zeros(256)
    y_reg_p = np.zeros(256)
    y_reg_n = np.zeros(256)
    x_reg_p[rows[x==1]] = -1
    x_reg_n[rows[x==-1]] = -1
    y_reg_p[cols[y==1]] = -1
    y_reg_n[cols[y==-1]] = -1

    spi.spi_write(dev, x_reg_p, forward=False, overwrite=False)
    spi.spi_write(dev, y_reg_p, forward=True, overwrite=False)
    apply_set_pulse(dev, 1.5, 0.0, 1e-6, prep=dac_setup)
    dev.SetWireInValue(0x0B, 0b11)
    dev.SetWireInValue(0x05, 0b010000)
    dev.UpdateWireIns()
    dev.SetWireInValue(0x0B, 0b00)
    dev.SetWireInValue(0x05, 0b000000)
    dev.UpdateWireIns()

    spi.spi_write(dev, x_reg_n, forward=False, overwrite=False)
    spi.spi_write(dev, y_reg_n, forward=True, overwrite=False)
    apply_set_pulse(dev, 1.5, 0.0, 1e-6, prep=dac_setup)
    dev.SetWireInValue(0x0B, 0b11)
    dev.SetWireInValue(0x05, 0b010000)
    dev.UpdateWireIns()
    dev.SetWireInValue(0x0B, 0b00)
    dev.SetWireInValue(0x05, 0b000000)
    dev.UpdateWireIns()

    spi.reset(dev)
    

def conductance_remap_reset(g_current, g_max):
    g_diff = g_current[0::2, :] - g_current[1::2, :]
    g_remap = np.ones_like(g_current) * g_max
    g_remap[1::2, :][g_diff > 0] = g_max - g_diff[g_diff > 0]
    g_remap[0::2, :][g_diff < 0] = g_max + g_diff[g_diff < 0]
    return g_remap


def program_array(dev, rows, cols, core_row, core_col, G, g_min, g_tol, num_epoch, g_max=np.inf, vset_wl=2.5, end_voltage=3.3, vread=1.0, vref=0.9, iteration_limit=20,
                  pulse_width=1e-6, record_pulse=False, verbose=1):
    M = len(rows)
    N = len(cols)
    program_sucess = np.zeros((M, N), dtype=bool)
    final_readings = np.zeros((M, N))
    num_pulses = np.zeros((M, N))
    pulses = []
    start = time.time()

    for epoch in range(num_epoch):
        for ri, r in enumerate(rows):
            for ci, c in enumerate(cols):
                if G[ri,ci] < g_min:
                    program_sucess[ri,ci], final_readings[ri,ci], num_pulse, pulse_train, _ = program_increment(
                        dev, row=r, col=c, core_row=core_row, core_col=core_col,
                        r_low=1/g_min, r_high=np.inf,
                        vset_start=1.2, vset_end=end_voltage, vset_wl=vset_wl,
                        vreset_start=2.5, vreset_end=end_voltage,
                        pulse_width=pulse_width, iteration_limit=1, max_pulse_limit=30, vread=vread, vref=vref,
                        read_cycles=3, ignore_cycles=1,
                        verbose=verbose, record_history=record_pulse)
                elif G[ri,ci] > g_max:
                    program_sucess[ri,ci], final_readings[ri,ci], num_pulse, pulse_train, _ = program_increment(
                        dev, row=r, col=c, core_row=core_row, core_col=core_col,
                        r_low=1/(G[ri,ci]+g_tol), r_high=1/np.minimum(g_max, (G[ri,ci]-g_tol)),
                        vset_start=2.0, vset_end=end_voltage, vset_wl=vset_wl,
                        vreset_start=1.5, vreset_end=end_voltage,
                        pulse_width=pulse_width, iteration_limit=1, max_pulse_limit=30, vread=vread, vref=vref,
                        verbose=verbose, record_history=record_pulse)
                else:
                    g_lower = G[ri,ci]-g_tol
                    if g_lower <= 0:
                        r_higher = np.inf
                    else:
                        r_higher = 1/g_lower
                    program_sucess[ri,ci], final_readings[ri,ci], num_pulse, pulse_train, _ = program_increment(
                        dev, row=r, col=c, core_row=core_row, core_col=core_col,
                        r_low=1/(G[ri,ci]+g_tol), r_high=r_higher,
                        vset_start=1.2, vset_end=end_voltage, vset_wl=vset_wl,
                        vreset_start=1.2, vreset_end=end_voltage,
                        pulse_width=pulse_width, iteration_limit=iteration_limit, max_pulse_limit=30, vread=vread, vref=vref,
                        verbose=verbose, record_history=record_pulse)
                    
                num_pulses[ri,ci] += num_pulse
                if record_pulse:
                    pulses.append(pulse_train)
                if verbose > 0:
                    print('Finished programming %d row %d col' % (r, c))
        print('Finished programming epoch %d.' % epoch)
    end = time.time()
    print('Time elapsed: %fs' % (end - start))
    return (program_sucess, 1/final_readings, num_pulses, pulses)


def read_array(dev, rows, cols, core_row, core_col, read_cycle=20, ignore_cycle=10, vread=1.0, vref=0.9, verbose=1):
    M = len(rows)
    N = len(cols)
    readings_post = np.zeros((M, N))
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            write_reg(dev, r, c, core_row, core_col)
            readings_post[ri, ci] = adc.read_average_resistance(dev, vread, vref, 20, 40, read_cycle, ignore_cycle)
            if verbose > 0:
               print('Finished reading %d row %d col' % (r, c))
    return 1/readings_post
