`timescale 1ns / 1ps
//------------------------------------------------------------------------
// neurram_wupdate_control.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module neurram_wupdate_control(
	input clk,
	input rst,

	// EP interface
	input wire read_trigger,
	input wire read_ack,
	input wire vread_on,

	input wire program_trigger,
	input wire wupdate_mode_trigger,
	input wire [31:0] pulse_width,
	input wire [31:0] wupdate_mode_width,
	input wire program_ack,
	output reg program_done,

	// Neurram interface
	// output wire wupdate_mode, //This can be controlled directly by epwire
	output reg wupdate_pulse,
	output reg wupdate_mode
	);


reg [2:0] state, next_state;
reg [31:0] clk_counter;

parameter [2:0] STATE_IDLE = 3'd0;
parameter [2:0] STATE_READ = 3'd1;
parameter [2:0] STATE_PROGRAM = 3'd2;
parameter [2:0] STATE_WUPDATE_MODE = 3'd3;
parameter [2:0] STATE_PROGRAM_DONE = 3'd4;


always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
		wupdate_pulse <= 0;
		wupdate_mode <= 0;
		clk_counter <= 0;
		program_done <= 0;
	end else begin
		state <= next_state;
		case (state)
			STATE_IDLE: begin
				wupdate_pulse <= 0;
				wupdate_mode <= 0;
				clk_counter <= 0;
				program_done <= 0;
			end
			STATE_READ: begin
				wupdate_pulse <= vread_on;
				wupdate_mode <= vread_on;
				clk_counter <= 0;
				program_done <= 0;
			end
			STATE_PROGRAM: begin
				wupdate_pulse <= 1;
				wupdate_mode <= 1;
				clk_counter <= clk_counter + 1;
				program_done <= 0;
			end
			STATE_WUPDATE_MODE: begin
				wupdate_pulse <= 0;
				wupdate_mode <= 1;
				clk_counter <= clk_counter + 1;
				program_done <= 0;
			end
			STATE_PROGRAM_DONE: begin
				wupdate_pulse <= 0;
				wupdate_mode <= 0;
				clk_counter <= 0;
				program_done <= 1;
			end
			default: begin
				wupdate_pulse <= 0;
				wupdate_mode <= 0;
				clk_counter <= 0;
				program_done <= 0;
			end
		endcase
	end
end

always @(*) begin
	case (state)
		STATE_IDLE: begin
			if (read_trigger) next_state = STATE_READ;
			else if (program_trigger) next_state = STATE_PROGRAM;
			else if (wupdate_mode_trigger) next_state = STATE_WUPDATE_MODE;
			else next_state = STATE_IDLE;				
		end
		STATE_READ: begin
			if (read_ack) next_state = STATE_IDLE;
			else next_state = STATE_READ;
		end
		STATE_PROGRAM: begin
			if (clk_counter == pulse_width) next_state = STATE_PROGRAM_DONE;
			else next_state = STATE_PROGRAM;
		end
		STATE_WUPDATE_MODE: begin
			if (clk_counter == wupdate_mode_width) next_state = STATE_PROGRAM_DONE;
			else next_state = STATE_WUPDATE_MODE;
		end
		STATE_PROGRAM_DONE: begin
			if (program_ack) next_state = STATE_IDLE;
			else next_state = STATE_PROGRAM_DONE;
		end
		default: begin
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule