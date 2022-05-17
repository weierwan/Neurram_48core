`timescale 1ns / 1ps
//------------------------------------------------------------------------
// lfsr_control.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module lfsr_control
	(
	input wire clk,
	input wire ok_clk,
	input wire rst,

	input wire lfsr_shift_trigger,
	input wire lfsr_pulse_trigger,
	input wire [8:0] shift_cycle,

	output wire [31:0] pipe_out,
	input wire out_fifo_rd_en,
	output wire out_fifo_empty,

	output reg lfsr_clk,
	input wire [1:0] lfsr_in,
	output reg lfsr_pulse,
	output reg lfsr_neuron_on,

	output reg inf_mode_off,
	output reg ext_inf,
	output reg lfsr_mode,

	input wire [3:0] pulse_width,
	
	input wire sw_integ,
	output reg integ_trig
	);


wire out_fifo_full;
reg out_fifo_wr_en;
reg [31:0] out_fifo_din;

FIFO32x32 FIFO_pipe_out(

	// General input
	.wr_clk(clk), // input wr_clk
	.rd_clk(ok_clk), // input rd_clk
	.rst(rst), // input rst

	// FSM interface
	.full(out_fifo_full), // output full
	.wr_en(out_fifo_wr_en), // input wr_en
	.wr_ack(), // output wr_ack
	.din(out_fifo_din), // input [31 : 0] din
	
	// End-Point Pipe interface
	.empty(out_fifo_empty), // output empty
	.rd_en(out_fifo_rd_en), // input rd_en
	.dout(pipe_out), // output [31 : 0] dout	
	.valid() // output valid
);


reg [4:0] state, next_state;
reg [8:0] clk_counter;
reg [4:0] pip_counter;
reg [255:0] shift_lfsr2pip[1:0];

parameter [4:0] STATE_IDLE = 5'd0;
parameter [4:0]	STATE_SHIFT = 5'd1;
parameter [4:0]	STATE_SHIFT2 = 5'd2;
parameter [4:0] STATE_PIPE_OUT = 5'd3;
parameter [4:0] STATE_PREP1 = 5'd4;
parameter [4:0] STATE_PREP2 = 5'd5;
parameter [4:0] STATE_PREP3 = 5'd6;
parameter [4:0] STATE_PREP4 = 5'd7;
parameter [4:0] STATE_PULSE1 = 5'd8;
parameter [4:0] STATE_PULSE2 = 5'd9;
parameter [4:0] STATE_TRIG_INTEG1 = 5'd10;
parameter [4:0] STATE_TRIG_INTEG2 = 5'd11;
parameter [4:0] STATE_PULSE_DONE1 = 5'd12;
parameter [4:0] STATE_PULSE_DONE2 = 5'd13;
parameter [4:0] STATE_PULSE_DONE3 = 5'd14;


always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
		lfsr_clk <= 0;
		out_fifo_wr_en <= 0;
		out_fifo_din <= 0;
		clk_counter <= 0;
		pip_counter <= 0;
		shift_lfsr2pip[0] <= 0;
		shift_lfsr2pip[1] <= 0;
		lfsr_pulse <= 0;
		lfsr_neuron_on <= 0;
		inf_mode_off <= 0;
		ext_inf <= 0;
		lfsr_mode <= 0;
		integ_trig <= 0;
	end else begin
		state <= next_state;
		case (state)
			STATE_IDLE: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 0;
				ext_inf <= 0;
				lfsr_mode <= 0;
				integ_trig <= 0;
			end
			STATE_SHIFT: begin
				lfsr_clk <= 1;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= clk_counter;
				pip_counter <= 0;
				shift_lfsr2pip[0][255] <= lfsr_in[0];
				shift_lfsr2pip[0][254:0] <= shift_lfsr2pip[0][255:1];
				shift_lfsr2pip[1][255] <= lfsr_in[1];
				shift_lfsr2pip[1][254:0] <= shift_lfsr2pip[1][255:1];
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 0;
				ext_inf <= 0;
				lfsr_mode <= 0;
				integ_trig <= 0;
			end
			STATE_SHIFT2: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= clk_counter + 1;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= shift_lfsr2pip[0];
				shift_lfsr2pip[1] <= shift_lfsr2pip[1];
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 0;
				ext_inf <= 0;
				lfsr_mode <= 0;
				integ_trig <= 0;
			end
			STATE_PIPE_OUT: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 1;
				clk_counter <= 0;
				pip_counter <= pip_counter + 1;
				shift_lfsr2pip[0] <= shift_lfsr2pip[0];
				shift_lfsr2pip[1] <= shift_lfsr2pip[1];
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 0;
				ext_inf <= 0;
				lfsr_mode <= 0;
				integ_trig <= 0;
				if (pip_counter < 8) begin
					out_fifo_din <= shift_lfsr2pip[0][32*pip_counter +: 32];
				end else begin
					out_fifo_din <= shift_lfsr2pip[1][32*(pip_counter-8) +: 32];
				end
			end
			STATE_PREP1: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 0;
				ext_inf <= 1;
				lfsr_mode <= 0;
				integ_trig <= 0;
			end
			STATE_PREP2: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 1;
				ext_inf <= 1;
				lfsr_mode <= 0;
				integ_trig <= 0;
			end
			STATE_PREP3: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 1;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 1;
				ext_inf <= 0;
				lfsr_mode <= 0;
				integ_trig <= 0;
			end
			STATE_PREP4: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 1;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 1;
				ext_inf <= 0;
				lfsr_mode <= 1;
				integ_trig <= 0;
			end
			STATE_PULSE1: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= clk_counter + 1;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 1;
				lfsr_neuron_on <= 1;
				inf_mode_off <= 1;
				ext_inf <= 0;
				lfsr_mode <= 1;
				integ_trig <= 0;
			end
			STATE_PULSE2: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 1;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 1;
				ext_inf <= 0;
				lfsr_mode <= 1;
				integ_trig <= 0;
			end
			STATE_TRIG_INTEG1: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 1;
				ext_inf <= 0;
				lfsr_mode <= 1;
				integ_trig <= 1;
			end
			STATE_TRIG_INTEG2: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 1;
				ext_inf <= 0;
				lfsr_mode <= 1;
				integ_trig <= 0;
			end
			STATE_PULSE_DONE1: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 1;
				ext_inf <= 0;
				lfsr_mode <= 0;
				integ_trig <= 0;
			end
			STATE_PULSE_DONE2: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 1;
				ext_inf <= 1;
				lfsr_mode <= 0;
				integ_trig <= 0;
			end
			STATE_PULSE_DONE3: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 0;
				ext_inf <= 1;
				lfsr_mode <= 0;
				integ_trig <= 0;
			end
			default: begin
				lfsr_clk <= 0;
				out_fifo_wr_en <= 0;
				out_fifo_din <= 0;
				clk_counter <= 0;
				pip_counter <= 0;
				shift_lfsr2pip[0] <= 0;
				shift_lfsr2pip[1] <= 0;
				lfsr_pulse <= 0;
				lfsr_neuron_on <= 0;
				inf_mode_off <= 0;
				ext_inf <= 0;
				lfsr_mode <= 0;
				integ_trig <= 0;
			end
		endcase
	end
end

always @(*) begin
	case (state)
		STATE_IDLE: begin
			if (lfsr_shift_trigger) begin
				next_state = STATE_SHIFT;
			end else if (lfsr_pulse_trigger) begin
				next_state = STATE_PREP1;
			end else begin
				next_state = STATE_IDLE;
			end
		end
		STATE_SHIFT: begin 
			next_state = STATE_SHIFT2;
		end
		STATE_SHIFT2: begin
			if (clk_counter == shift_cycle-1) next_state = STATE_PIPE_OUT;
			else next_state = STATE_SHIFT;
		end
		STATE_PIPE_OUT: begin
			if (pip_counter == 15) next_state = STATE_IDLE;
			else next_state = STATE_PIPE_OUT;
		end
		STATE_PREP1: begin
			next_state = STATE_PREP2;
		end
		STATE_PREP2: begin
			next_state = STATE_PREP3;
		end
		STATE_PREP3: begin
			next_state = STATE_PREP4;
		end
		STATE_PREP4: begin
			next_state = STATE_PULSE1;
		end
		STATE_PULSE1: begin
			if (clk_counter == pulse_width) next_state = STATE_PULSE2;
			else next_state = STATE_PULSE1;
		end
		STATE_PULSE2: begin
			next_state = STATE_TRIG_INTEG1;
		end
		STATE_TRIG_INTEG1: begin
			if (sw_integ) next_state = STATE_TRIG_INTEG2;
			else next_state = STATE_TRIG_INTEG1;
		end
		STATE_TRIG_INTEG2: begin
			if (~sw_integ) next_state = STATE_PULSE_DONE1;
			else next_state = STATE_TRIG_INTEG2;
		end
		STATE_PULSE_DONE1: begin
			next_state = STATE_PULSE_DONE2;
		end
		STATE_PULSE_DONE2: begin
			next_state = STATE_PULSE_DONE3;
		end
		STATE_PULSE_DONE3: begin
			next_state = STATE_IDLE;
		end
		default: begin
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule
