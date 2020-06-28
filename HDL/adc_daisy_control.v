`timescale 1ns / 1ps
//------------------------------------------------------------------------
// adc_daisy_control.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module adc_daisy_control #(parameter num_adc = 3)
	(
	// General input
	input wire clk,
	input wire rst,

	// End-Point (slave) interface
	input wire trigger,
	input wire ack,
	output reg ready,
	output reg [18*num_adc - 1 : 0] dout,

	// DAC (master) interface
	output reg adc_sck, //20MHz
	output reg convst,
	input wire adc_sdo
	);

// FSM
reg [2:0] state, next_state;
reg [18*num_adc - 1 : 0] next_dout;
reg [3:0] tconv_cnt, next_tconv_cnt;
reg [5:0] dout_cnt, next_dout_cnt;


parameter [2:0] STATE_IDLE = 3'b000;
parameter [2:0] STATE_CONV = 3'b001;
parameter [2:0] STATE_DOUT = 3'b010;
parameter [2:0] STATE_DOUT2 = 3'b011;
parameter [2:0] STATE_READY = 3'b100;



always @(posedge rst, posedge clk) begin
	if (rst) begin
		state <= STATE_IDLE;
		dout <= 0;
		tconv_cnt <= 0;
		dout_cnt <= 0;
	end else begin
		state <= next_state;
		dout <= next_dout;
		tconv_cnt <= next_tconv_cnt;
		dout_cnt <= next_dout_cnt;
	end
end

always @(*) begin
	case(state)
		STATE_IDLE: begin
			ready = 0;
			adc_sck = 0;
			convst = 0;
			next_dout = 0;
			next_tconv_cnt = 0;
			next_dout_cnt = 0;

			if (trigger) next_state = STATE_CONV;
			else next_state = STATE_IDLE;
		end

		STATE_CONV: begin
			ready = 0;
			adc_sck = 0;
			convst = 1;
			next_dout = 0;
			next_tconv_cnt = tconv_cnt + 1;
			next_dout_cnt = 0;

			if (tconv_cnt >= 9) next_state = STATE_DOUT;
			else next_state = STATE_CONV;
		end
		STATE_DOUT: begin
			ready = 0;
			adc_sck = 1;
			convst = 1;
			next_dout[0] = adc_sdo;
			next_dout[18*num_adc - 1 : 1] = dout[18*num_adc - 2 : 0];
			next_tconv_cnt = 0;
			next_dout_cnt = dout_cnt + 1;

			next_state = STATE_DOUT2;
		end
		STATE_DOUT2: begin
			ready = 0;
			adc_sck = 0;
			convst = 1;
			next_dout = dout;
			next_tconv_cnt = 0;
			next_dout_cnt = dout_cnt;

			if (dout_cnt == 18 * num_adc + 1) next_state = STATE_READY;
			else next_state = STATE_DOUT;
		end
		STATE_READY: begin
			ready = 1;
			adc_sck = 0;
			convst = 0;
			next_dout = dout;
			next_tconv_cnt = 0;
			next_dout_cnt = 0;

			if (ack) next_state = STATE_IDLE;
			else next_state = STATE_READY;
		end

		default: begin
			ready = 0;
			adc_sck = 0;
			convst = 0;
			next_dout = 0;
			next_tconv_cnt = 0;
			next_dout_cnt = 0;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule
