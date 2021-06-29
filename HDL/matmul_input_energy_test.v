`timescale 1ns / 1ps
//------------------------------------------------------------------------
// matmul_input_energy_test.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module matmul_input_energy_test(
	input wire clk,
	input wire rst,

	// host interface
	input wire trigger,
	input wire [2:0] num_bits,
	input wire [4:0] pulse_multiplier,
	input wire [31:0] iterations,
	output reg idle,

	// Neurram control module interface
	input wire neuron_idle,
	output reg neuron_sample_trigger,
	output reg neuron_cds_trigger,
	output reg [7:0] num_pulses
	);


reg [2:0] state, next_state;
reg [7:0] next_num_pulses;
reg [2:0] clk_counter, next_clk_counter;
reg [2:0] cycle_counter, next_cycle_counter;
reg [31:0] iter_counter, next_iter_counter;


parameter [2:0] STATE_IDLE = 3'd0;
parameter [2:0] STATE_CDS_TRIG = 3'd1;
parameter [2:0] STATE_CDS_WAIT = 3'd2;
parameter [2:0] STATE_SAMPLE_TRIG = 3'd3;
parameter [2:0] STATE_SAMPLE_WAIT = 3'd4;
parameter [2:0] STATE_CYCLE_DONE = 3'd5;


always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
		idle <= 0;
		neuron_sample_trigger <= 0;
		neuron_cds_trigger <= 0;
		num_pulses <= 0;
		clk_counter <= 0;
		cycle_counter <= 0;
		iter_counter <= 0;
	end else begin
		state <= next_state;
		num_pulses <= next_num_pulses;
		clk_counter <= next_clk_counter;
		cycle_counter <= next_cycle_counter;
		iter_counter <= next_iter_counter;
		case (state)
			STATE_IDLE: begin
				idle <= 1;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
			end
			STATE_CDS_TRIG: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 1;
			end
			STATE_CDS_WAIT: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
			end
			STATE_SAMPLE_TRIG: begin
				idle <= 0;
				neuron_sample_trigger <= 1;
				neuron_cds_trigger <= 0;
			end
			STATE_SAMPLE_WAIT: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
			end
			STATE_CYCLE_DONE: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
			end
			default: begin
				idle <= 0;
				neuron_sample_trigger <= 0;
				neuron_cds_trigger <= 0;
			end
		endcase
	end
end


always @(*) begin
	case (state)
		STATE_IDLE: begin
			next_num_pulses = pulse_multiplier;
			next_clk_counter = 0;
			next_cycle_counter = 0;
			next_iter_counter = 0;
			if (trigger) next_state = STATE_CDS_TRIG;
			else next_state = STATE_IDLE;
		end
		STATE_CDS_TRIG: begin
			next_num_pulses = pulse_multiplier;
			next_clk_counter = clk_counter + 1;
			next_cycle_counter = 0;
			next_iter_counter = iter_counter;
			if (clk_counter == 3) next_state = STATE_CDS_WAIT;
			else next_state = STATE_CDS_TRIG;
		end
		STATE_CDS_WAIT: begin
			next_num_pulses = pulse_multiplier;
			next_clk_counter = 0;
			next_cycle_counter = 0;
			next_iter_counter = iter_counter;
			if (neuron_idle) next_state = STATE_SAMPLE_TRIG;
			else next_state = STATE_CDS_WAIT;
		end
		STATE_SAMPLE_TRIG: begin
			next_num_pulses = num_pulses;
			next_clk_counter = clk_counter + 1;
			next_cycle_counter = cycle_counter;
			next_iter_counter = iter_counter;
			if (clk_counter == 3) next_state = STATE_SAMPLE_WAIT;
			else next_state = STATE_SAMPLE_TRIG;
		end
		STATE_SAMPLE_WAIT: begin
			next_num_pulses = num_pulses;
			next_clk_counter = 0;
			next_cycle_counter = cycle_counter;
			next_iter_counter = iter_counter;
			if (neuron_idle) begin
				next_state = STATE_CYCLE_DONE;
			end else begin
				next_state = STATE_SAMPLE_WAIT;
			end
		end
		STATE_CYCLE_DONE: begin
			next_clk_counter = 0;
			next_cycle_counter = cycle_counter + 1;
			next_num_pulses = num_pulses << 1;
			next_iter_counter = iter_counter;
			if (cycle_counter == num_bits) begin
				next_iter_counter = iter_counter + 1;
				if (iter_counter == iterations - 1) next_state = STATE_IDLE;
				else next_state = STATE_CDS_TRIG;
			end else next_state = STATE_SAMPLE_TRIG;
		end
		default: begin
			next_num_pulses = 0;
			next_clk_counter = 0;
			next_cycle_counter = 0;
			next_iter_counter = 0;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule