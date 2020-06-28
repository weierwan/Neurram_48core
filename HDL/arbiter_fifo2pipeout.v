`timescale 1ns / 1ps
//------------------------------------------------------------------------
// arbiter_fifo2pipeout.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module arbiter_fifo2pipeout #(parameter FIFO_SIZE = 256)(
	input wire clk,
	input wire ok_clk,
	input wire rst,

	// Pipe-out interface
	output wire [31:0] pipe_out,
	input wire pipe_out_read,
	input wire [7:0] core_select,
	input wire [7:0] num_words,
	output reg idle,
	output wire pipe_out_empty,

	// Neurram I/O FIFO interface
	input wire [255:0] data_from_fifo,
	input wire [7:0] empty_from_fifo,
	input wire [7:0] valid_from_fifo,
	output reg [7:0] rd_en_from_ok
);

reg fifo_wr_en;
reg [31:0] fifo_din;
wire fifo_full;


generate
	if (FIFO_SIZE == 512) begin: fifo_gen_512
		FIFO32x512 FIFO_pipe_out(
			// General input
			.wr_clk(clk), // input wr_clk
			.rd_clk(ok_clk), // input rd_clk
			.rst(rst), // input rst

			// FSM interface
			.full(fifo_full), // output full
			.wr_en(fifo_wr_en), // input wr_en
			.wr_ack(), // output wr_ack
			.din(fifo_din), // input [31 : 0] din
			
			// Pipe-out interface
			.empty(pipe_out_empty), // output empty
			.rd_en(pipe_out_read), // input rd_en
			.dout(pipe_out), // output [31 : 0] dout	
			.valid() // output valid
		);
	end else begin: fifo_gen_256
		FIFO32x256 FIFO_pipe_out(
			// General input
			.wr_clk(clk), // input wr_clk
			.rd_clk(ok_clk), // input rd_clk
			.rst(rst), // input rst

			// FSM interface
			.full(fifo_full), // output full
			.wr_en(fifo_wr_en), // input wr_en
			.wr_ack(), // output wr_ack
			.din(fifo_din), // input [31 : 0] din
			
			// Pipe-out interface
			.empty(pipe_out_empty), // output empty
			.rd_en(pipe_out_read), // input rd_en
			.dout(pipe_out), // output [31 : 0] dout	
			.valid() // output valid
		);
	end
endgenerate



reg [1:0] state, next_state;
reg [7:0] write_counter, next_write_counter;
reg [3:0] addr_counter, next_addr_counter;

parameter [1:0] STATE_IDLE = 2'd1;
parameter [1:0] STATE_CHECK_ADDR = 2'd2;
parameter [1:0] STATE_TRANSFER = 2'd3;


always @(posedge clk, posedge rst) begin
	if (rst) begin
		state <= STATE_IDLE;
		write_counter <= 0;
		addr_counter <= 0;
	end else begin
		state <= next_state;
		write_counter <= next_write_counter;
		addr_counter <= next_addr_counter;
	end
end


always @(*) begin
	case (state)
		STATE_IDLE: begin
			idle = 1;
			fifo_din = 0;
			fifo_wr_en = 0;
			rd_en_from_ok = 0;
			next_write_counter = 0;
			next_addr_counter = 0;

			if (~((& empty_from_fifo) | fifo_full)) next_state = STATE_CHECK_ADDR;
			else next_state = STATE_IDLE;
		end
		STATE_CHECK_ADDR: begin
			idle = 0;
			fifo_din = 0;
			fifo_wr_en = 0;
			rd_en_from_ok = 0;
			next_write_counter = 0;
			if (addr_counter >= 8) begin
				next_addr_counter = addr_counter;
				next_state = STATE_IDLE;
			end else if (core_select[addr_counter] == 1) begin
				next_addr_counter = addr_counter;
				next_state = STATE_TRANSFER;
			end else begin
				next_addr_counter = addr_counter + 1;
				next_state = STATE_CHECK_ADDR;
			end
		end
		STATE_TRANSFER: begin
			idle = 0;
			fifo_din = 0;
			fifo_wr_en = 0;
			rd_en_from_ok = 0;
			if (write_counter == num_words) begin
				next_write_counter = write_counter;
				next_addr_counter = addr_counter + 1;
				next_state = STATE_CHECK_ADDR;
			end else begin
				fifo_din = data_from_fifo[addr_counter*32 +: 32];
				fifo_wr_en = valid_from_fifo[addr_counter];
				rd_en_from_ok[addr_counter] = ~fifo_full;
				if (fifo_wr_en && (~fifo_full)) next_write_counter = write_counter + 1;
				else next_write_counter = write_counter;
				next_addr_counter = addr_counter;
				next_state = STATE_TRANSFER;
			end
		end
		default: begin
			idle = 0;
			fifo_din = 0;
			fifo_wr_en = 0;
			rd_en_from_ok = 0;
			next_write_counter = 0;
			next_addr_counter = 0;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule
