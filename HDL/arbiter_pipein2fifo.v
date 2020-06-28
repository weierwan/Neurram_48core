`timescale 1ns / 1ps
//------------------------------------------------------------------------
// arbiter_pipein2fifo.v
// Author: Weier Wan
//
//
//
//------------------------------------------------------------------------

module arbiter_pipein2fifo #(parameter FIFO_SIZE = 1024)(
	input wire clk,
	input wire ok_clk,
	input wire rst,

	// Pipe-in interface
	input wire [31:0] pipe_in,
	input wire pipe_in_write,
	input wire [7:0] core_select,
	input wire [9:0] num_words,
	output reg idle,
	output wire pipe_in_full,

	// Neurram I/O FIFO interface
	output reg [255:0] data2fifo,
	output reg [7:0] wr_en_2fifo,
	input wire [7:0] rd_en_2ok
);

reg fifo_rd_en;
wire [31:0] fifo_dout;
wire fifo_valid, fifo_empty;


generate
if (FIFO_SIZE == 64) begin: fifo_gen_64
	FIFO32x64FWFT FIFO_pipe_in(
		// General input
		.wr_clk(ok_clk), // input wr_clk
		.rd_clk(clk), // input rd_clk
		.rst(rst), // input rst

		// End-Point Pipe interface
		.full(pipe_in_full), // output full
		.wr_en(pipe_in_write), // input wr_en
		.wr_ack(), // output wr_ack
		.din(pipe_in), // input [31 : 0] din
		
		// FSM interface
		.empty(fifo_empty), // output empty
		.rd_en(fifo_rd_en), // input rd_en
		.dout(fifo_dout), // output [31 : 0] dout	
		.valid(fifo_valid) // output valid
	);
end else begin: fifo_gen_1024
	FIFO32x1024FWFT FIFO_pipe_in(
		// General input
		.wr_clk(ok_clk), // input wr_clk
		.rd_clk(clk), // input rd_clk
		.rst(rst), // input rst

		// End-Point Pipe interface
		.full(pipe_in_full), // output full
		.wr_en(pipe_in_write), // input wr_en
		.wr_ack(), // output wr_ack
		.din(pipe_in), // input [31 : 0] din
		
		// FSM interface
		.empty(fifo_empty), // output empty
		.rd_en(fifo_rd_en), // input rd_en
		.dout(fifo_dout), // output [31 : 0] dout	
		.valid(fifo_valid) // output valid
	);
end
endgenerate


reg [1:0] state, next_state;
reg [9:0] write_counter, next_write_counter;
reg [3:0] addr_counter, next_addr_counter;


parameter [1:0] STATE_IDLE = 2'b00;
parameter [1:0] STATE_CHECK_ADDR = 2'b01;
parameter [1:0] STATE_TRANSFER = 2'b11;


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
			fifo_rd_en = 0;
			data2fifo = 0;
			wr_en_2fifo = 0;
			next_write_counter = 0;
			next_addr_counter = 0;

			if (~fifo_empty) next_state = STATE_CHECK_ADDR;
			else next_state = STATE_IDLE;
		end
		STATE_CHECK_ADDR: begin
			idle = 0;
			data2fifo = 0;
			wr_en_2fifo = 0;
			fifo_rd_en = 0;
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
			data2fifo = 0;
			wr_en_2fifo = 0;
			fifo_rd_en = 0;
			if (write_counter == num_words) begin
				next_write_counter = write_counter;
				next_addr_counter = addr_counter + 1;
				next_state = STATE_CHECK_ADDR;
			end else begin
				data2fifo[addr_counter*32 +: 32] = fifo_dout;
				wr_en_2fifo[addr_counter] = fifo_valid;
				fifo_rd_en = rd_en_2ok[addr_counter];
				if (fifo_valid && rd_en_2ok[addr_counter]) next_write_counter = write_counter + 1;
				else next_write_counter = write_counter;
				next_addr_counter = addr_counter;
				next_state = STATE_TRANSFER;
			end
		end
		default: begin
			idle = 0;
			fifo_rd_en = 0;
			data2fifo = 0;
			wr_en_2fifo = 0;
			next_write_counter = 0;
			next_addr_counter = 0;
			next_state = STATE_IDLE;
		end
	endcase
end

endmodule

