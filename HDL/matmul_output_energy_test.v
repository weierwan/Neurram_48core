`timescale 1ns / 1ps
//------------------------------------------------------------------------
// matmul_output_energy_test.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module matmul_output_energy_test(
	input wire clk,
	input wire rst,

	// host interface
	input wire trigger,
	input wire cds,
	input wire [7:0] steps,
	input wire [31:0] iterations,
	output reg idle,

	// Neurram control module interface
	input wire neuron_idle,
	output reg neuron_cds_trigger,
	output reg neuron_reset_trigger
	);


reg [2:0] state, next_state;
reg [2:0] clk_counter, next_clk_counter;
reg [7:0] step_counter, next_step_counter;
reg [31:0] iter_counter, next_iter_counter;

parameter [2:0] STATE_IDLE = 3'd0;
parameter [2:0] STATE_RESET_TRIG = 3'd1;
parameter [2:0] STATE_RESET_WAIT = 3'd2;
parameter [2:0] STATE_CDS_TRIG = 3'd3;
parameter [2:0] STATE_CDS_WAIT = 3'd4;

always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
		idle <= 0;
		neuron_reset_trigger <= 0;
		clk_counter <= 0;
		step_counter <= 0;
		iter_counter <= 0;
	end else begin
		state <= next_state;
		clk_counter <= next_clk_counter;
		step_counter <= next_step_counter;
		iter_counter <= next_iter_counter;
		case (state)
			STATE_IDLE: begin
				idle <= 1;
				neuron_cds_trigger <= 0;
				neuron_reset_trigger <= 0;
			end
			STATE_CDS_TRIG: begin
				idle <= 0;
				neuron_cds_trigger <= 1;
				neuron_reset_trigger <= 0;
			end
			STATE_CDS_WAIT: begin
				idle <= 0;
				neuron_cds_trigger <= 0;
				neuron_reset_trigger <= 0;
			end
			STATE_RESET_TRIG: begin
				idle <= 0;
				neuron_cds_trigger <= 0;
				neuron_reset_trigger <= 1;
			end
			STATE_RESET_WAIT: begin
				idle <= 0;
				neuron_cds_trigger <= 0;
				neuron_reset_trigger <= 0;
			end
			default: begin
				idle <= 0;
				neuron_cds_trigger <= 0;
				neuron_reset_trigger <= 0;
			end
		endcase
	end
end


always @(*) begin
	case (state)
		STATE_IDLE: begin
			next_clk_counter = 0;
			next_step_counter = 0;
			next_iter_counter = 0;
			if (trigger) begin
				if (cds) next_state = STATE_CDS_TRIG;
				else next_state = STATE_RESET_TRIG;
			end else next_state = STATE_IDLE;
		end
		STATE_CDS_TRIG: begin
			next_clk_counter = clk_counter + 1;
			next_step_counter = step_counter;
			next_iter_counter = iter_counter;
			if (clk_counter == 3) next_state = STATE_CDS_WAIT;
			else next_state = STATE_CDS_TRIG;
		end
		STATE_CDS_WAIT: begin
			next_clk_counter = 0;
			next_step_counter = step_counter;
			next_iter_counter = iter_counter;
			if (neuron_idle) next_state = STATE_RESET_TRIG;
			else next_state = STATE_CDS_WAIT;
		end
		STATE_RESET_TRIG: begin
			next_clk_counter = clk_counter + 1;
			next_step_counter = step_counter;
			next_iter_counter = iter_counter;
			if (clk_counter == 3) next_state = STATE_RESET_WAIT;
			else next_state = STATE_RESET_TRIG;
		end
		STATE_RESET_WAIT: begin
			next_clk_counter = 0;
			next_step_counter = step_counter;
			next_iter_counter = iter_counter;
			if (neuron_idle) begin
				next_step_counter = step_counter + 1;
				if (step_counter == steps - 1) begin
					next_step_counter = 0;
					next_iter_counter = iter_counter + 1;
					if (iter_counter == iterations - 1) next_state = STATE_IDLE;
					else if (cds) next_state = STATE_CDS_TRIG;
					else next_state = STATE_RESET_TRIG;
				end else next_state = STATE_RESET_TRIG;
			end else next_state = STATE_RESET_WAIT;
		end
		default: begin
			next_clk_counter = 0;
			next_step_counter = 0;
			next_iter_counter = 0;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule
