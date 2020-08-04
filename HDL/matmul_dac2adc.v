`timescale 1ns / 1ps
//------------------------------------------------------------------------
// matmul_dac2adc.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module matmul_dac2adc(
	input wire clk,
	input wire rst,

	// host interface
	input wire trigger,
	input wire [7:0] iteration,
	output reg idle,
	
	// Neurram interface
	input wire matmul_unsigned_idle,
	input wire nmlo_idle,
	output reg matmul_unsigned_trigger,
	output reg nmlo_trigger
	);

reg [3:0] state, next_state;
reg [7:0] cycle_counter, next_cycle_counter;


parameter [3:0] STATE_IDLE = 4'd0;
parameter [3:0] STATE_UNSIGNED_TRIG = 4'd1;
parameter [3:0] STATE_UNSIGNED_WAIT = 4'd2;
parameter [3:0] STATE_NMLO_TRIG = 4'd3;
parameter [3:0] STATE_NMLO_WAIT = 4'd4;


always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
		idle <= 0;
		matmul_unsigned_trigger <= 0;
		nmlo_trigger <= 0;
		cycle_counter <= 0;
	end else begin
		state <= next_state;
		cycle_counter <= next_cycle_counter;
		case (state)
			STATE_IDLE: begin
				idle <= 1;
				matmul_unsigned_trigger <= 0;
				nmlo_trigger <= 0;
			end
			STATE_UNSIGNED_TRIG: begin
				idle <= 0;
				matmul_unsigned_trigger <= 1;
				nmlo_trigger <= 0;
			end
			STATE_UNSIGNED_WAIT: begin
				idle <= 0;
				matmul_unsigned_trigger <= 0;
				nmlo_trigger <= 0;
			end
			STATE_NMLO_TRIG: begin
				idle <= 0;
				matmul_unsigned_trigger <= 0;
				nmlo_trigger <= 1;
			end
			STATE_NMLO_WAIT: begin
				idle <= 0;
				matmul_unsigned_trigger <= 0;
				nmlo_trigger <= 0;
			end
			default: begin
				idle <= 0;
				matmul_unsigned_trigger <= 0;
				nmlo_trigger <= 0;
			end
		endcase
	end
end

always @(*) begin
	case (state)
		STATE_IDLE: begin
			next_cycle_counter = 0;
			if (trigger) begin
				next_cycle_counter = cycle_counter + 1;
				next_state = STATE_UNSIGNED_TRIG;
			end else next_state = STATE_IDLE;
		end
		STATE_UNSIGNED_TRIG: begin
			next_cycle_counter = cycle_counter;
			if (matmul_unsigned_idle) next_state = STATE_UNSIGNED_TRIG;
			else next_state = STATE_UNSIGNED_WAIT;
		end
		STATE_UNSIGNED_WAIT: begin
			next_cycle_counter = cycle_counter;
			if (matmul_unsigned_idle) next_state = STATE_NMLO_TRIG;
			else next_state = STATE_UNSIGNED_WAIT;
		end
		STATE_NMLO_TRIG: begin
			next_cycle_counter = cycle_counter;
			if (nmlo_idle) next_state = STATE_NMLO_TRIG;
			else next_state = STATE_NMLO_WAIT;
		end
		STATE_NMLO_WAIT: begin
			next_cycle_counter = cycle_counter;
			if (nmlo_idle) begin
				if (cycle_counter >= iteration) begin
					next_state = STATE_IDLE;
				end else begin
					next_state = STATE_UNSIGNED_TRIG;
					next_cycle_counter = cycle_counter + 1;
				end
			end else next_state = STATE_NMLO_WAIT;
		end
		default: begin
			next_cycle_counter = cycle_counter;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule