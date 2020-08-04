`timescale 1ns / 1ps
//------------------------------------------------------------------------
// neuron_control.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module neuron_control(
	input wire clk,
	input wire rst,

    input wire cds_trigger,
    input wire pulse_trigger,
    input wire compare_write_trigger,
    input wire integ_trigger,
	input wire partial_reset_trigger,

    input wire run_all,
    input wire run_all_reset,
    input wire keep_integ_on,
    input wire use_ext_1n,
    input wire use_ext_3n,
    input wire [1:0] comparison_phase,
	input wire [1:0] partial_reset_phase,
    input wire [7:0] num_pulses,
    output reg idle,

    input wire [7:0] t_opamp,
    input wire [7:0] t_sample,

    output reg sw_cds,
    output reg sw_cds_vref,
    output reg sw_in_sample,
    output reg sw_integ,
    output reg sw_ota_loop,
    output reg sw_feedback,
    output reg sw_latch_en,
    output reg [1:0] vcomp_config,
    output reg [1:0] v_level1_config,
    output reg [1:0] v_level2_config,
    output reg ns_pulse_trig,
    output reg ext_1n,
    output reg ext_3n,
    output reg precharge_off,
    output reg [1:0] neuron_read_trigger,
    output reg register_mode,
    output reg sampling,
    output reg compare_write,
	output reg partial_reset,
    output reg ext_inf_on,
	output reg sel_vfb
    );


reg [5:0] state, next_state;
reg [10:0] clk_counter;
reg [7:0] pulse_cnt;

parameter [5:0] STATE_IDLE = 6'd0;
parameter [5:0] STATE_CDS1 = 6'd1;
parameter [5:0] STATE_CDS2 = 6'd2;
parameter [5:0] STATE_CDS3 = 6'd3;
parameter [5:0] STATE_CDS4 = 6'd4;
parameter [5:0] STATE_SAMPLING = 6'd5;
parameter [5:0] STATE_SAMPLING1 = 6'd6;
parameter [5:0] STATE_SAMPLING2 = 6'd7;
parameter [5:0] STATE_INTEG1 = 6'd8;
parameter [5:0] STATE_COMPARE = 6'd9;
parameter [5:0] STATE_COMPARE_WRITE_MINUS1 = 6'd10;
parameter [5:0] STATE_COMPARE_WRITE_MINUS2 = 6'd11;
parameter [5:0] STATE_COMPARE_WRITE_MINUS2_1 = 6'd12;
parameter [5:0] STATE_COMPARE_WRITE_MINUS3 = 6'd13;
parameter [5:0] STATE_COMPARE_WRITE_MINUS4 = 6'd14;
parameter [5:0] STATE_COMPARE_WRITE_MINUS5 = 6'd15;
parameter [5:0] STATE_WRITE_MINUS_DONE0 = 6'd16;
parameter [5:0] STATE_WRITE_MINUS_DONE = 6'd17;
parameter [5:0] STATE_WRITE_MINUS_DONE2 = 6'd18;
parameter [5:0] STATE_COMPARE_WRITE_PLUS1 = 6'd19;
parameter [5:0] STATE_COMPARE_WRITE_PLUS2 = 6'd20;
parameter [5:0] STATE_COMPARE_WRITE_PLUS2_1 = 6'd21;
parameter [5:0] STATE_COMPARE_WRITE_PLUS3 = 6'd22;
parameter [5:0] STATE_COMPARE_WRITE_PLUS4 = 6'd23;
parameter [5:0] STATE_COMPARE_WRITE_PLUS5 = 6'd24;
parameter [5:0] STATE_WRITE_PLUS_DONE0 = 6'd25;
parameter [5:0] STATE_WRITE_PLUS_DONE = 6'd26;
parameter [5:0] STATE_WRITE_PLUS_DONE2 = 6'd27;
parameter [5:0] STATE_PARTIAL_RESET = 6'd29;
parameter [5:0] STATE_PARTIAL_RESET_MINUS1 = 6'd30;
parameter [5:0] STATE_PARTIAL_RESET_MINUS2 = 6'd31;
parameter [5:0] STATE_PARTIAL_RESET_MINUS3 = 6'd33;
parameter [5:0] STATE_PARTIAL_RESET_MINUS4 = 6'd34;
parameter [5:0] STATE_PARTIAL_RESET_MINUS5 = 6'd35;
parameter [5:0] STATE_RESET_MINUS_DONE0 = 6'd36;
parameter [5:0] STATE_RESET_MINUS_DONE = 6'd37;
parameter [5:0] STATE_PARTIAL_RESET_PLUS1 = 6'd39;
parameter [5:0] STATE_PARTIAL_RESET_PLUS2 = 6'd40;
parameter [5:0] STATE_PARTIAL_RESET_PLUS3 = 6'd42;
parameter [5:0] STATE_PARTIAL_RESET_PLUS4 = 6'd43;
parameter [5:0] STATE_PARTIAL_RESET_PLUS5 = 6'd44;
parameter [5:0] STATE_RESET_PLUS_DONE0 = 6'd45;
parameter [5:0] STATE_RESET_PLUS_DONE = 6'd46;


always @(posedge clk, posedge rst) begin
    if (rst) begin
        state <= STATE_IDLE;
        sw_cds <= 0;
        sw_cds_vref <= 0;
        sw_in_sample <= 0;
        sw_integ <= 0;
        sw_ota_loop <= 1;
        sw_feedback <= 0;
        sw_latch_en <= 1;
        vcomp_config <= 2'b00;
        v_level1_config <= 2'b00;
        v_level2_config <= 2'b00;
        ns_pulse_trig <= 0;
        ext_1n <= 0;
        ext_3n <= 0;
        precharge_off <= 0;
        neuron_read_trigger <= 2'b00;
        clk_counter <= 0;
        idle <= 0;
        register_mode <= 0;
        sampling <= 0;
        compare_write <= 0;
		partial_reset <= 0;
        ext_inf_on <= 0;
        pulse_cnt <= 0;
		sel_vfb <= 0;
    end else begin
        state <= next_state;
        case (state)
            STATE_IDLE: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= keep_integ_on;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b00;
                v_level1_config <= 2'b00;
                v_level2_config <= 2'b00;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 1;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_CDS1: begin
                sw_cds <= 1;
                sw_cds_vref <= 1;
                sw_in_sample <= 0;
                sw_integ <= 1;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b11;
                v_level1_config <= 2'b00;
                v_level2_config <= 2'b00;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_CDS2: begin
                sw_cds <= 0;
                sw_cds_vref <= 1;
                sw_in_sample <= 0;
                sw_integ <= 1;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b11;
                v_level1_config <= 2'b00;
                v_level2_config <= 2'b00;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_CDS3: begin
                sw_cds <= 0;
                sw_cds_vref <= 1;
                sw_in_sample <= 0;
                sw_integ <= keep_integ_on;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b11;
                v_level1_config <= 2'b00;
                v_level2_config <= 2'b00;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_CDS4: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= keep_integ_on;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b11;
                v_level1_config <= 2'b00;
                v_level2_config <= 2'b00;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_SAMPLING: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= keep_integ_on;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b00;
                v_level1_config <= 2'b00;
                v_level2_config <= 2'b00;
                ns_pulse_trig <= 0;
                ext_1n <= use_ext_1n;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 1;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= pulse_cnt + 1;
				sel_vfb <= 0;
            end
            STATE_SAMPLING1: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 1;
                sw_integ <= keep_integ_on;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b00;
                v_level1_config <= 2'b00;
                v_level2_config <= 2'b00;
                ns_pulse_trig <= 1;
                ext_1n <= use_ext_1n;
                ext_3n <= use_ext_3n;
                precharge_off <= 1;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 1;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= pulse_cnt;
				sel_vfb <= 0;
            end
            STATE_SAMPLING2: begin
               sw_cds <= 0;
               sw_cds_vref <= 0;
               sw_in_sample <= 0;
               sw_integ <= keep_integ_on;
               sw_ota_loop <= 1;
               sw_feedback <= 0;
               sw_latch_en <= 1;
               vcomp_config <= 2'b00;
               v_level1_config <= 2'b00;
               v_level2_config <= 2'b00;
               ns_pulse_trig <= 0;
               ext_1n <= use_ext_1n;
               ext_3n <= 0;
               precharge_off <= 1;
               neuron_read_trigger <= 2'b00;
               clk_counter <= 0; //clk_counter + 1;
               idle <= 0;
               register_mode <= 0;
               sampling <= 1;
               compare_write <= 0;
				partial_reset <= 0;
               ext_inf_on <= 0;
               pulse_cnt <= pulse_cnt;
				sel_vfb <= 0;
            end
            STATE_INTEG1: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 1;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b00;
                v_level1_config <= 2'b00;
                v_level2_config <= 2'b00;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= pulse_cnt;
				sel_vfb <= 0;
            end
            STATE_COMPARE: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b00;
                v_level1_config <= 2'b00;
                v_level2_config <= 2'b00;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 1;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_MINUS1: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 0;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b01;
                v_level2_config <= 2'b10;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 1;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_MINUS2: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b01;
                v_level2_config <= 2'b10;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 1;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_MINUS2_1: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b01;
                v_level2_config <= 2'b10;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_MINUS3: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b01;
                v_level2_config <= 2'b10;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 1;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_MINUS4: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b01;
                v_level2_config <= 2'b10;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 1;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 1;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_MINUS5: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b01;
                v_level2_config <= 2'b10;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 1;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b01;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 1;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_WRITE_MINUS_DONE0: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b01;
                v_level2_config <= 2'b10;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_WRITE_MINUS_DONE: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b01;
                v_level2_config <= 2'b10;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 1;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_WRITE_MINUS_DONE2: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b01;
                v_level2_config <= 2'b10;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 1;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_PLUS1: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 0;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b10;
                v_level2_config <= 2'b01;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 1;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_PLUS2: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b10;
                v_level2_config <= 2'b01;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 1;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_PLUS2_1: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b10;
                v_level2_config <= 2'b01;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_PLUS3: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b10;
                v_level2_config <= 2'b01;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 1;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_PLUS4: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b10;
                v_level2_config <= 2'b01;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 1;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 1;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_COMPARE_WRITE_PLUS5: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b10;
                v_level2_config <= 2'b01;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 1;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b10;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 1;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_WRITE_PLUS_DONE0: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b10;
                v_level2_config <= 2'b01;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_WRITE_PLUS_DONE: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b10;
                v_level2_config <= 2'b01;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 1;
				partial_reset <= 0;
                ext_inf_on <= 1;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_WRITE_PLUS_DONE2: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b10;
                v_level2_config <= 2'b01;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 0;
                ext_inf_on <= 1;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
			STATE_PARTIAL_RESET: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b00;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_PARTIAL_RESET_MINUS1: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 0;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_PARTIAL_RESET_MINUS2: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
				pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_PARTIAL_RESET_MINUS3: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_PARTIAL_RESET_MINUS4: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 1;
                sw_integ <= 0;
                sw_ota_loop <= 1;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_PARTIAL_RESET_MINUS5: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 1;
                sw_integ <= 0;
                sw_ota_loop <= 1;
                sw_feedback <= 1; // better not to turn off feedback and in_sample at the same tmie
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_RESET_MINUS_DONE0: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 1;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_RESET_MINUS_DONE: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 1;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b01;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
            end
            STATE_PARTIAL_RESET_PLUS1: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 0;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 1;
            end
            STATE_PARTIAL_RESET_PLUS2: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 1;
            end
            STATE_PARTIAL_RESET_PLUS3: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 0;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 1;
            end
            STATE_PARTIAL_RESET_PLUS4: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 1;
                sw_integ <= 0;
                sw_ota_loop <= 1;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 1;
            end
            STATE_PARTIAL_RESET_PLUS5: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 1;
                sw_integ <= 0;
                sw_ota_loop <= 1;
                sw_feedback <= 1;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 1;
            end
            STATE_RESET_PLUS_DONE0: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 1;
            end
            STATE_RESET_PLUS_DONE: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 1;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b10;
                v_level1_config <= 2'b11;
                v_level2_config <= 2'b11;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= clk_counter + 1;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
				partial_reset <= 1;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 1;
            end
            default: begin
                sw_cds <= 0;
                sw_cds_vref <= 0;
                sw_in_sample <= 0;
                sw_integ <= 0;
                sw_ota_loop <= 1;
                sw_feedback <= 0;
                sw_latch_en <= 1;
                vcomp_config <= 2'b00;
                v_level1_config <= 2'b00;
                v_level2_config <= 2'b00;
                ns_pulse_trig <= 0;
                ext_1n <= 0;
                ext_3n <= 0;
                precharge_off <= 0;
                neuron_read_trigger <= 2'b00;
                clk_counter <= 0;
                idle <= 0;
                register_mode <= 0;
                sampling <= 0;
                compare_write <= 0;
                ext_inf_on <= 0;
                pulse_cnt <= 0;
				sel_vfb <= 0;
                partial_reset <= 0;
            end
        endcase
    end
end


always @(*) begin
    case (state)
        STATE_IDLE: begin
            if (cds_trigger) next_state = STATE_CDS1;
            else if (pulse_trigger) next_state = STATE_SAMPLING;
            else if (compare_write_trigger) next_state = STATE_COMPARE;
            else if (integ_trigger) next_state = STATE_INTEG1;
				else if (partial_reset_trigger) next_state = STATE_PARTIAL_RESET;
            else next_state = STATE_IDLE;
        end
        STATE_CDS1: begin
            if (clk_counter == t_opamp) next_state = STATE_CDS2;
            else next_state = STATE_CDS1;
        end
        STATE_CDS2: begin
            next_state = STATE_CDS3;
        end
        STATE_CDS3: begin
            next_state = STATE_CDS4;
        end
        STATE_CDS4: begin
            if (run_all) next_state = STATE_SAMPLING;
            else next_state = STATE_IDLE;
        end
        STATE_SAMPLING: begin
            next_state = STATE_SAMPLING1;
        end
        STATE_SAMPLING1: begin
            if (clk_counter == t_sample) next_state = STATE_SAMPLING2;
            else next_state = STATE_SAMPLING1;
        end
        STATE_SAMPLING2: begin
           next_state = STATE_INTEG1;
        end
        STATE_INTEG1: begin
            if (clk_counter == t_opamp) begin
                if (pulse_cnt < num_pulses) begin
                    next_state = STATE_SAMPLING;
                end else begin
				    if (run_all) begin
                        next_state = STATE_COMPARE;
                    end else begin
                        next_state = STATE_IDLE;
                    end
				end
            end else next_state = STATE_INTEG1;
        end
        STATE_COMPARE: begin
            if (comparison_phase[0]) next_state = STATE_COMPARE_WRITE_MINUS1;
            else if (comparison_phase[1]) next_state = STATE_COMPARE_WRITE_PLUS1;
            else next_state = STATE_IDLE;
        end
        STATE_COMPARE_WRITE_MINUS1: begin
            if (clk_counter == t_opamp) next_state = STATE_COMPARE_WRITE_MINUS2;
            else next_state = STATE_COMPARE_WRITE_MINUS1;
        end
        STATE_COMPARE_WRITE_MINUS2: begin
            next_state = STATE_COMPARE_WRITE_MINUS2_1;
        end
        STATE_COMPARE_WRITE_MINUS2_1: begin
            next_state = STATE_COMPARE_WRITE_MINUS3;
        end
        STATE_COMPARE_WRITE_MINUS3: begin
            next_state = STATE_COMPARE_WRITE_MINUS4;
        end
        STATE_COMPARE_WRITE_MINUS4: begin
            next_state = STATE_COMPARE_WRITE_MINUS5;
        end
        STATE_COMPARE_WRITE_MINUS5: begin
            if (clk_counter == 3) next_state = STATE_WRITE_MINUS_DONE0;
            else next_state = STATE_COMPARE_WRITE_MINUS5;
        end
        STATE_WRITE_MINUS_DONE0: begin
            next_state = STATE_WRITE_MINUS_DONE;
        end
        STATE_WRITE_MINUS_DONE: begin
            next_state = STATE_WRITE_MINUS_DONE2;
        end
        STATE_WRITE_MINUS_DONE2: begin
            if (comparison_phase[1]) next_state = STATE_COMPARE_WRITE_PLUS1;
            else if (run_all_reset && partial_reset_phase[0]) next_state = STATE_PARTIAL_RESET_MINUS2;
            else next_state = STATE_IDLE;
        end
        STATE_COMPARE_WRITE_PLUS1: begin
            if (clk_counter == t_opamp) next_state = STATE_COMPARE_WRITE_PLUS2;
            else next_state = STATE_COMPARE_WRITE_PLUS1;
        end
        STATE_COMPARE_WRITE_PLUS2: begin
            next_state = STATE_COMPARE_WRITE_PLUS2_1;
        end
        STATE_COMPARE_WRITE_PLUS2_1: begin
            next_state = STATE_COMPARE_WRITE_PLUS3;
        end
        STATE_COMPARE_WRITE_PLUS3: begin
            next_state = STATE_COMPARE_WRITE_PLUS4;
        end
        STATE_COMPARE_WRITE_PLUS4: begin
            next_state = STATE_COMPARE_WRITE_PLUS5;
        end
        STATE_COMPARE_WRITE_PLUS5: begin
            if (clk_counter == 3) next_state = STATE_WRITE_PLUS_DONE0;
            else next_state = STATE_COMPARE_WRITE_PLUS5;
        end
        STATE_WRITE_PLUS_DONE0: begin
            next_state = STATE_WRITE_PLUS_DONE;
        end
        STATE_WRITE_PLUS_DONE: begin
            next_state = STATE_WRITE_PLUS_DONE2;
        end
        STATE_WRITE_PLUS_DONE2: begin
            if (run_all_reset) begin
                if (partial_reset_phase[0]) next_state = STATE_PARTIAL_RESET;
                else if (partial_reset_phase[1]) next_state = STATE_PARTIAL_RESET_PLUS2;
                else next_state = STATE_IDLE;
			end
            else next_state = STATE_IDLE;
        end
		STATE_PARTIAL_RESET: begin
            if (partial_reset_phase[0]) next_state = STATE_PARTIAL_RESET_MINUS1;
            else if (partial_reset_phase[1]) next_state = STATE_PARTIAL_RESET_PLUS1;
            else next_state = STATE_IDLE;
        end
        STATE_PARTIAL_RESET_MINUS1: begin
            if (clk_counter == t_opamp) next_state = STATE_PARTIAL_RESET_MINUS2;
            else next_state = STATE_PARTIAL_RESET_MINUS1;
        end
        STATE_PARTIAL_RESET_MINUS2: begin
            next_state = STATE_PARTIAL_RESET_MINUS3;
        end
        STATE_PARTIAL_RESET_MINUS3: begin
            next_state = STATE_PARTIAL_RESET_MINUS4;
        end
        STATE_PARTIAL_RESET_MINUS4: begin
            next_state = STATE_PARTIAL_RESET_MINUS5;
        end
        STATE_PARTIAL_RESET_MINUS5: begin
            if (clk_counter == t_opamp) next_state = STATE_RESET_MINUS_DONE0;
            else next_state = STATE_PARTIAL_RESET_MINUS5;
        end
        STATE_RESET_MINUS_DONE0: begin
            next_state = STATE_RESET_MINUS_DONE;
        end
        STATE_RESET_MINUS_DONE: begin
            if (clk_counter == t_opamp) begin
                if (partial_reset_phase[1]) next_state = STATE_PARTIAL_RESET_PLUS1;
                else next_state = STATE_IDLE;
            end
            else next_state = STATE_RESET_MINUS_DONE;
        end
        STATE_PARTIAL_RESET_PLUS1: begin
            if (clk_counter == t_opamp) next_state = STATE_PARTIAL_RESET_PLUS2;
            else next_state = STATE_PARTIAL_RESET_PLUS1;
        end
        STATE_PARTIAL_RESET_PLUS2: begin
            next_state = STATE_PARTIAL_RESET_PLUS3;
        end
        STATE_PARTIAL_RESET_PLUS3: begin
            next_state = STATE_PARTIAL_RESET_PLUS4;
        end
        STATE_PARTIAL_RESET_PLUS4: begin
            next_state = STATE_PARTIAL_RESET_PLUS5;
        end
        STATE_PARTIAL_RESET_PLUS5: begin
            if (clk_counter == t_opamp) next_state = STATE_RESET_PLUS_DONE0;
            else next_state = STATE_PARTIAL_RESET_PLUS5;
        end
        STATE_RESET_PLUS_DONE0: begin
            next_state = STATE_RESET_PLUS_DONE;
        end
        STATE_RESET_PLUS_DONE: begin
            if (clk_counter == t_opamp) next_state = STATE_IDLE;
            else next_state = STATE_RESET_PLUS_DONE;
        end
        default: begin
            next_state = STATE_IDLE;
        end
    endcase
end

endmodule