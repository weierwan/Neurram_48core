`timescale 1ns / 1ps
//------------------------------------------------------------------------
// dac_daisy_control.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module dac_daisy_control(
	// General input
	input wire clk, //100MHz
	input wire clk_dac, //20MHz
	input wire rst,

	// End-Point (slave) interface
	input wire program,
	input wire update,
	input wire [31:0] ep_din,
	output wire fifo_full,
	output reg fifo_wr_done,
	output wire fifo_empty,
	output reg dac_state_idle,
	
	// Debugging outputs
	// output wire [1:0] dac_ep_state,
	// output wire [2:0] dac_dac_state,

	// DAC (master) interface
	output reg dac_sck,
	output reg dac_cs_b,
	output wire dac_sdi
	);

// assign dac_ep_state = ep_state;
// assign dac_dac_state = dac_state;

wire fifo_wr_ack;
reg fifo_wr_en;

wire fifo_valid;
wire [31:0] fifo_dout;
reg fifo_rd_en;

FIFO32x16 FIFO_daisy0(

	// General input
	.wr_clk(clk), // input wr_clk
	.rd_clk(clk_dac), // input rd_clk
	.rst(rst), // input rst

	// End-Point interface
	.full(fifo_full), // output full
	.wr_en(fifo_wr_en), // input wr_en
	.wr_ack(fifo_wr_ack), // output wr_ack
	.din(ep_din), // input [31 : 0] din
	
	
	// DAC interface
	.empty(fifo_empty), // output empty
	.rd_en(fifo_rd_en), // input rd_en
	.dout(fifo_dout), // output [31 : 0] dout	
	.valid(fifo_valid) // output valid
);


// FSM for End-Point interface
reg [1:0] ep_state, ep_next_state;

parameter [1:0] EP_IDLE = 2'b00;
parameter [1:0] EP_PROGRAM = 2'b01;
parameter [1:0] EP_WAIT_ACK = 2'b10;
parameter [1:0] EP_WR_DONE = 2'b11;


always @(posedge rst, posedge clk) begin
	if (rst) begin
		ep_state <= EP_IDLE;
	end else begin
		ep_state <= ep_next_state;
	end
end

always @(*) begin
	case (ep_state)
		EP_IDLE: begin
			fifo_wr_en = 0;
			fifo_wr_done = 0;

			if ((!fifo_full) && program) ep_next_state = EP_PROGRAM;
			else ep_next_state = EP_IDLE;
		end

		EP_PROGRAM: begin
			fifo_wr_en = 1;
			fifo_wr_done = 0;

			ep_next_state = EP_WAIT_ACK;
		end

		EP_WAIT_ACK: begin
			fifo_wr_en = 0;
			fifo_wr_done = 0;

			if (fifo_wr_ack) ep_next_state = EP_WR_DONE;
			else ep_next_state = EP_WAIT_ACK;
		end

		EP_WR_DONE: begin
			fifo_wr_en = 0;
			fifo_wr_done = 1;

			if ((!fifo_full) && program) ep_next_state = EP_PROGRAM;
			else ep_next_state = EP_WR_DONE;
		end

		default: begin
			fifo_wr_en = 0;
			fifo_wr_done = 0;
			ep_next_state = EP_IDLE;
		end
	endcase
end

// FSM for DAC interface
reg [1:0] dac_state, dac_next_state;
reg [4:0] counter, next_counter;
reg [31:0] shift_reg, next_shift_reg;

assign dac_sdi = shift_reg[31];

parameter [1:0] DAC_IDLE = 2'b00;
parameter [1:0] DAC_RD_EN = 2'b01;
// parameter [2:0] DAC_LOAD = 3'b010;
parameter [1:0] DAC_SHIFT = 2'b10;
parameter [1:0] DAC_SHIFT_DONE = 2'b11;

always @(posedge rst, posedge clk_dac) begin
	if (rst) begin
		dac_state <= DAC_IDLE;
		counter <= 0;
		shift_reg <= 0;
	end else begin
		dac_state <= dac_next_state;
		shift_reg <= next_shift_reg;
		counter <= next_counter;
	end
end

always @(*) begin
	case (dac_state)
		DAC_IDLE: begin
			fifo_rd_en = 0;
			dac_sck = 0;
			dac_cs_b = 1;
			dac_state_idle = 1;
			next_shift_reg = 0;
			next_counter = 0;

			if (!fifo_empty) dac_next_state = DAC_RD_EN;
			else dac_next_state = DAC_IDLE;
		end

		DAC_RD_EN: begin
			fifo_rd_en = 1;
			dac_sck = 0;
			dac_cs_b = 0;
			dac_state_idle = 0;
			next_counter = 0;

			if (fifo_valid) begin 
				dac_next_state = DAC_SHIFT;
				next_shift_reg = fifo_dout;
			end else begin
				dac_next_state = DAC_RD_EN;
				next_shift_reg = 0;
			end
		end

		DAC_SHIFT: begin
			fifo_rd_en = 0;
			dac_sck = ~clk_dac;
			dac_cs_b = 0;
			dac_state_idle = 0;
			next_shift_reg = shift_reg << 1;
			next_counter = counter + 1;

			if (counter >= 31) dac_next_state = DAC_SHIFT_DONE;
			else dac_next_state = DAC_SHIFT;
		end

		DAC_SHIFT_DONE: begin
			fifo_rd_en = 0;
			dac_sck = 0;
			dac_cs_b = 0;
			dac_state_idle = 0;
			next_shift_reg = shift_reg;
			next_counter = counter;

			if (!fifo_empty) dac_next_state = DAC_RD_EN;
			else if (update) dac_next_state = DAC_IDLE;
			else dac_next_state = DAC_SHIFT_DONE;
		end

		default: begin
			fifo_rd_en = 0;
			dac_sck = 0;
			dac_cs_b = 1;
			dac_state_idle = 0;
			next_shift_reg = 0;
			next_counter = 0;
			dac_next_state = DAC_IDLE;
		end
	endcase
end

endmodule